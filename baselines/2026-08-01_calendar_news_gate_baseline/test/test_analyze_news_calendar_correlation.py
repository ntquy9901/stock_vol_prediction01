"""Unit tests for the per-point delta_QLIKE / calendar-correlation EDA logic.

No trained checkpoints needed here -- these test the PURE logic (qlike_pointwise math, date
extraction alignment, groupby/correlation aggregation) with synthetic data, so they run fast and
don't depend on `models/har_only_ablation_ref_.../best.pt` existing on disk.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
_DUAL_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_DUAL_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from analyze_news_calendar_correlation import (  # noqa: E402
    qlike_pointwise, extract_target_dates, build_dataframe, analyze,
)
from dataset_dual_news import MultiStockDatasetWithDualNews, HAR_COLS  # noqa: E402
from src.common.evaluation import qlike_loss  # noqa: E402
from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402

pytestmark = pytest.mark.smoke

STOCKS = ["AAA", "BBB"]
N_DAYS = 60


def _make_har_df(seed: int, start="2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=N_DAYS, freq="D")
    return pd.DataFrame({
        "date": dates.astype(str),
        "parkinson_volatility": rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[0]: rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[1]: rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[2]: rng.uniform(0.001, 0.05, N_DAYS),
    })


class TestQlikePointwise:
    def test_mean_matches_aggregate_qlike_loss(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0.001, 0.05, 200)
        y_pred = rng.uniform(0.001, 0.05, 200)
        pointwise = qlike_pointwise(y_true, y_pred)
        assert pointwise.shape == (200,)
        assert np.mean(pointwise) == pytest.approx(qlike_loss(y_true, y_pred), rel=1e-6)

    def test_perfect_prediction_is_zero(self):
        y = np.array([0.01, 0.02, 0.03])
        assert np.allclose(qlike_pointwise(y, y), 0.0, atol=1e-9)

    def test_handles_near_zero_without_nan_or_inf(self):
        y_true = np.array([0.0, 1e-10, 0.02])
        y_pred = np.array([0.0, 0.02, 1e-10])
        out = qlike_pointwise(y_true, y_pred)
        assert np.all(np.isfinite(out))


class TestExtractTargetDates:
    def test_length_matches_windows_times_stocks(self):
        stock_data = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
        config = LSTMGATConfig()
        config.num_features_per_stock = 3
        ds = MultiStockDatasetWithDualNews(
            stock_data, stock_data, STOCKS,
            seq_length=10, forecast_horizon=5, graph_method="correlation",
            normalize=False, config=config, news_panel_path=None,
        )
        dates = extract_target_dates(ds)
        assert len(dates) == len(ds) * len(STOCKS)

    def test_first_window_target_date_matches_manual_computation(self):
        stock_data = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
        config = LSTMGATConfig()
        config.num_features_per_stock = 3
        ds = MultiStockDatasetWithDualNews(
            stock_data, stock_data, STOCKS,
            seq_length=10, forecast_horizon=5, graph_method="correlation",
            normalize=False, config=config, news_panel_path=None,
        )
        dates = extract_target_dates(ds)
        # window 0: target_idx = 0 + 10 + 5 - 1 = 14
        expected_aaa = str(stock_data["AAA"]["date"].iloc[14])[:10]
        expected_bbb = str(stock_data["BBB"]["date"].iloc[14])[:10]
        assert dates[0] == expected_aaa
        assert dates[1] == expected_bbb

    def test_dates_advance_monotonically_per_ticker_across_windows(self):
        stock_data = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
        config = LSTMGATConfig()
        config.num_features_per_stock = 3
        ds = MultiStockDatasetWithDualNews(
            stock_data, stock_data, STOCKS,
            seq_length=10, forecast_horizon=5, graph_method="correlation",
            normalize=False, config=config, news_panel_path=None,
        )
        dates = extract_target_dates(ds)
        n_stocks = len(STOCKS)
        aaa_dates = [dates[i] for i in range(0, len(dates), n_stocks)]  # ticker-0 slot each window
        assert aaa_dates == sorted(aaa_dates)
        assert len(set(aaa_dates)) == len(aaa_dates)  # no duplicate target dates across windows


class TestBuildDataframeAndAnalyze:
    def test_build_dataframe_shapes_and_month_extraction(self):
        dates = ["2024-01-05", "2024-01-05", "2024-07-15", "2024-07-15"]
        tickers = ["AAA", "BBB", "AAA", "BBB"]
        delta = np.array([-0.1, -0.05, 0.2, 0.15])
        y_true = np.array([0.01, 0.02, 0.03, 0.04])
        df = build_dataframe(dates, tickers, delta, y_true)
        assert len(df) == 4
        assert df["month"].tolist() == [1, 1, 7, 7]
        assert set(df.columns) >= {"ticker", "date", "month", "y_true", "delta_qlike",
                                    "tet_proximity", "in_tet_window",
                                    "earnings_proximity", "in_earnings_window"}

    def test_analyze_month_bucketing_and_stats(self):
        # January points: strongly negative delta (news helps); July points: strongly positive
        dates = ["2024-01-05"] * 10 + ["2024-07-15"] * 10
        tickers = ["AAA"] * 20
        delta = np.array([-0.5] * 10 + [0.5] * 10)
        y_true = np.full(20, 0.02)
        df = build_dataframe(dates, tickers, delta, y_true)
        result = analyze(df)

        assert result["by_month"][1]["n"] == 10
        assert result["by_month"][1]["mean_delta_qlike"] == pytest.approx(-0.5)
        assert result["by_month"][7]["mean_delta_qlike"] == pytest.approx(0.5)
        assert result["by_month"][2]["n"] == 0
        assert result["by_month"][2]["mean_delta_qlike"] is None

    def test_analyze_tet_window_split_detects_known_difference(self):
        # 2024-02-10 is Tet 2024 (inside window); 2024-08-01 is far from Tet (outside)
        dates = ["2024-02-10"] * 15 + ["2024-08-01"] * 15
        tickers = ["AAA"] * 30
        delta = np.concatenate([np.full(15, -1.0), np.full(15, 1.0)])
        y_true = np.full(30, 0.02)
        df = build_dataframe(dates, tickers, delta, y_true)
        result = analyze(df)

        tw = result["tet_window"]
        assert tw["n_inside"] == 15
        assert tw["n_outside"] == 15
        assert tw["mean_delta_inside"] == pytest.approx(-1.0)
        assert tw["mean_delta_outside"] == pytest.approx(1.0)
        assert tw["welch_p_value"] < 0.01  # obviously different by construction

    def test_analyze_correlation_direction(self):
        # delta_qlike perfectly anti-correlated with tet_proximity by construction:
        # use dates with varying distance from Tet 2024-02-10
        dates = ["2024-02-10", "2024-02-15", "2024-02-20", "2024-03-01", "2024-05-01"]
        tickers = ["AAA"] * 5
        delta = np.array([-1.0, -0.7, -0.4, -0.1, 0.5])  # decreasing "news helps" as we move away
        y_true = np.full(5, 0.02)
        df = build_dataframe(dates, tickers, delta, y_true)
        result = analyze(df)
        assert result["tet_proximity_corr"]["pearson_r"] < 0  # farther from Tet (lower proximity) -> higher delta
