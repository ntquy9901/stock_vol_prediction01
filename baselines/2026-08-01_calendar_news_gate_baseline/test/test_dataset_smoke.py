"""Smoke tests for MultiStockDatasetWithCalendarNews.

Two layers, per CLAUDE.md Testing quality rules:
  1. Synthetic tmp_path panels (fast, exercises every code branch: coverage counting, missing-
     panel fallback, calendar-broadcast-across-tickers) -- mirrors
     `2026-07-25_macro_news_baseline/test/test_dataset_smoke.py`'s structure.
  2. A REAL-data-sample smoke (small slice of actual `data/processed/*.csv` + the actual
     `data/features/dual_group_news_panel.parquet`) -- catches things synthetic fixtures can't
     (date-format mismatch between real panel and real price data, unexpected NaN, etc.), per
     CLAUDE.md "Data-pipeline test phải có real-data-sample smoke."

Run: pytest baselines/2026-08-01_calendar_news_gate_baseline/test/test_dataset_smoke.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset_calendar_news import (  # noqa: E402
    MultiStockDatasetWithCalendarNews, N_CALENDAR_FEAT, HAR_COLS,
)
from calendar_features import compute_calendar_vector  # noqa: E402
from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402

pytestmark = pytest.mark.smoke

STOCKS = ["AAA", "BBB"]
N_DAYS = 40
N_DUAL_FEAT = 5


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


def _make_dual_panel(tmp_path) -> Path:
    rows = []
    for ticker in STOCKS:
        for d in ["2024-01-05", "2024-01-10", "2024-01-20"]:
            rows.append({"ticker": ticker, "date": d,
                        **{f"kq_emb_{i}": float(i + hash(ticker + d) % 7) for i in range(N_DUAL_FEAT)}})
    df = pd.DataFrame(rows)
    path = tmp_path / "dual_group_news_panel.parquet"
    df.to_parquet(path, index=False)
    return path


def _build_ds(tmp_path, news_panel_path):
    stock_data_with_har = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return MultiStockDatasetWithCalendarNews(
        stock_data_with_har, stock_data_with_har, STOCKS,
        seq_length=10, forecast_horizon=5, graph_method="correlation",
        normalize=False, config=config, news_panel_path=news_panel_path,
    )


class TestSyntheticShapesAndCoverage:
    def test_n_feat_is_dual_plus_calendar(self, tmp_path):
        ds = _build_ds(tmp_path, _make_dual_panel(tmp_path))
        assert ds._n_feat == N_DUAL_FEAT + N_CALENDAR_FEAT

    def test_shapes(self, tmp_path):
        ds = _build_ds(tmp_path, _make_dual_panel(tmp_path))
        S = len(STOCKS)
        x_har, adj, x_news, y = ds[0]
        assert x_har.shape == (10, S, 3)
        assert adj.shape == (S, S)
        assert x_news.shape == (10, S, N_DUAL_FEAT + N_CALENDAR_FEAT)
        assert y.shape == (S,)
        assert torch_isfinite_all(x_news)

    def test_partial_dual_coverage_tracked(self, tmp_path):
        ds = _build_ds(tmp_path, _make_dual_panel(tmp_path))
        assert ds._matched_dual_cells > 0
        assert ds._matched_dual_cells < ds._total_cells  # panel only covers 3/40 dates on purpose

    def test_calendar_part_never_zero_even_without_dual_panel(self):
        """Unlike the dual-group/macro vectors (which fall back to all-zero when no panel is
        given), the calendar vector is a pure function of the date -- it must be REAL, non-zero
        even in the no-panel ("smoke") path. This is the key behavioral difference from the
        macro_news_baseline sibling's `test_missing_panels_fall_back_to_dummy_zero_features`."""
        stock_data_with_har = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
        config = LSTMGATConfig()
        config.num_features_per_stock = 3
        ds = MultiStockDatasetWithCalendarNews(
            stock_data_with_har, stock_data_with_har, STOCKS,
            seq_length=10, forecast_horizon=5, graph_method="correlation",
            normalize=False, config=config, news_panel_path=None,
        )
        x_har, adj, x_news, y = ds[0]
        assert x_news.shape == (10, len(STOCKS), 146 + N_CALENDAR_FEAT)
        dual_part = x_news[:, :, :146]
        cal_part = x_news[:, :, 146:]
        assert (dual_part.numpy() == 0).all()
        assert not (cal_part.numpy() == 0).all(), "calendar part must be real, non-dummy"

    def test_calendar_vector_identical_across_tickers_for_same_date(self, tmp_path):
        ds = _build_ds(tmp_path, _make_dual_panel(tmp_path))
        x_har, adj, x_news, y = ds[0]
        cal_part = x_news[:, :, N_DUAL_FEAT:]
        for t in range(cal_part.shape[0]):
            np.testing.assert_array_almost_equal(
                cal_part[t, 0].numpy(), cal_part[t, 1].numpy(),
                err_msg="calendar vector must be identical across tickers at a given date")

    def test_calendar_values_match_pure_function(self, tmp_path):
        """Cross-check: the calendar slice of x_news[0] must equal compute_calendar_vector applied
        directly to the corresponding window date (catches off-by-one / wrong-column bugs)."""
        ds = _build_ds(tmp_path, _make_dual_panel(tmp_path))
        stock_feats = ds.stock_data_with_har["AAA"]
        window_dates = stock_feats["date"].iloc[0:10].astype(str).tolist()
        x_har, adj, x_news, y = ds[0]
        for t, d in enumerate(window_dates):
            expected = compute_calendar_vector(d)
            np.testing.assert_array_almost_equal(x_news[t, 0, N_DUAL_FEAT:].numpy(), expected)


class TestCalendarFeatureSubsetting:
    """2026-08-01 ablation feature: `calendar_feature_names` lets a train run include only a
    GROUP of the 10 calendar columns (e.g. only Tet-related), reusing this same dataset class."""

    def test_default_includes_all_10(self, tmp_path):
        ds = _build_ds(tmp_path, _make_dual_panel(tmp_path))
        assert ds._n_feat == N_DUAL_FEAT + N_CALENDAR_FEAT
        assert len(ds._calendar_feature_names) == N_CALENDAR_FEAT

    def test_subset_reduces_n_feat_and_column_count(self, tmp_path):
        stock_data_with_har = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
        config = LSTMGATConfig()
        config.num_features_per_stock = 3
        ds = MultiStockDatasetWithCalendarNews(
            stock_data_with_har, stock_data_with_har, STOCKS,
            seq_length=10, forecast_horizon=5, graph_method="correlation",
            normalize=False, config=config, news_panel_path=_make_dual_panel(tmp_path),
            calendar_feature_names=["tet_proximity", "in_tet_window"],
        )
        assert ds._n_feat == N_DUAL_FEAT + 2
        x_har, adj, x_news, y = ds[0]
        assert x_news.shape[-1] == N_DUAL_FEAT + 2

    def test_subset_values_match_correct_columns_of_full_vector(self, tmp_path):
        stock_data_with_har = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
        config = LSTMGATConfig()
        config.num_features_per_stock = 3
        subset_names = ["earnings_proximity", "dow_sin"]
        ds = MultiStockDatasetWithCalendarNews(
            stock_data_with_har, stock_data_with_har, STOCKS,
            seq_length=10, forecast_horizon=5, graph_method="correlation",
            normalize=False, config=config, news_panel_path=None,
            calendar_feature_names=subset_names,
        )
        x_har, adj, x_news, y = ds[0]
        window_dates = stock_data_with_har["AAA"]["date"].iloc[0:10].astype(str).tolist()
        for t, d in enumerate(window_dates):
            full = compute_calendar_vector(d)
            idx = {"dow_sin": 0, "dow_cos": 1, "month_sin": 2, "month_cos": 3,
                   "tet_proximity": 4, "in_tet_window": 5, "is_month_end": 6,
                   "is_quarter_end": 7, "earnings_proximity": 8, "in_earnings_window": 9}
            expected = np.array([full[idx[n]] for n in subset_names])
            np.testing.assert_array_almost_equal(
                x_news[t, 0, 146:].numpy(), expected)  # no dual panel -> 146 dummy zero cols first

    def test_unknown_feature_name_raises(self, tmp_path):
        stock_data_with_har = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
        config = LSTMGATConfig()
        config.num_features_per_stock = 3
        with pytest.raises(ValueError, match="unknown calendar feature"):
            MultiStockDatasetWithCalendarNews(
                stock_data_with_har, stock_data_with_har, STOCKS,
                seq_length=10, forecast_horizon=5, graph_method="correlation",
                normalize=False, config=config, news_panel_path=None,
                calendar_feature_names=["not_a_real_feature"],
            )


def torch_isfinite_all(t):
    return bool(np.isfinite(t.numpy()).all())


class TestRealDataSlice:
    """Real-data-sample smoke (CLAUDE.md Testing quality rules) -- small slice of ACTUAL
    data/processed CSVs + the ACTUAL dual_group_news_panel.parquet built by the sibling baseline.
    Skips gracefully if either real file is absent (e.g. a fresh checkout without data/)."""

    REAL_PRICE_FILE = _ROOT / "data" / "processed" / "ACB_processed.csv"
    REAL_PANEL_FILE = _ROOT / "data" / "features" / "dual_group_news_panel.parquet"

    @pytest.mark.skipif(not REAL_PRICE_FILE.exists(), reason="real price CSV not present")
    def test_real_price_slice_runs_without_exception(self):
        sys.path.insert(0, str(_ROOT))
        from src.common.feature_engineering import create_har_features

        raw = pd.read_csv(self.REAL_PRICE_FILE).head(120)  # small real slice, not the full file
        har = create_har_features(raw["parkinson_volatility"])
        combined = pd.concat([raw[["date", "parkinson_volatility"]], har], axis=1).dropna()
        combined = combined.rename(columns={
            "har_daily_vol": HAR_COLS[0], "har_weekly_vol": HAR_COLS[1], "har_monthly_vol": HAR_COLS[2],
        })
        assert len(combined) > 30

        stock_data_with_har = {"ACB": combined.reset_index(drop=True),
                                "ACB2": combined.reset_index(drop=True)}
        config = LSTMGATConfig()
        config.num_features_per_stock = 3
        news_panel = self.REAL_PANEL_FILE if self.REAL_PANEL_FILE.exists() else None
        ds = MultiStockDatasetWithCalendarNews(
            stock_data_with_har, stock_data_with_har, ["ACB", "ACB2"],
            seq_length=10, forecast_horizon=5, graph_method="correlation",
            normalize=False, config=config, news_panel_path=news_panel,
        )
        assert len(ds) > 0
        x_har, adj, x_news, y = ds[0]
        assert torch_isfinite_all(x_news)
        assert torch_isfinite_all(x_har)
        assert torch_isfinite_all(y)
