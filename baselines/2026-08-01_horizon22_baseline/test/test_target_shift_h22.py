"""Proves the target shifts to 22-days-ahead (not 5 or 10), using the UNCHANGED sibling dataset
class `MultiStockDatasetWithDualNews` (read-only import) -- mirrors
`2026-08-01_horizon10_baseline/test/test_target_shift.py`, adjusted for horizon=22.

Written BEFORE the two train scripts exist (test-first, CLAUDE.md §5 SDD Implement phase).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_DUAL_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_DUAL_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from dataset_dual_news import MultiStockDatasetWithDualNews, HAR_COLS  # noqa: E402
from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402

pytestmark = pytest.mark.smoke

STOCKS = ["AAA", "BBB"]
N_DAYS = 90  # needs to comfortably exceed seq_length(10) + forecast_horizon(22) = 32 for this test
SEQ_LENGTH = 10


def _make_har_df(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
    return pd.DataFrame({
        "date": dates.astype(str),
        "parkinson_variance": np.arange(N_DAYS, dtype=np.float64),  # known values for exact checks
        HAR_COLS[0]: rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[1]: rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[2]: rng.uniform(0.001, 0.05, N_DAYS),
    })


def _build_ds(forecast_horizon: int) -> MultiStockDatasetWithDualNews:
    stock_data = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return MultiStockDatasetWithDualNews(
        stock_data, stock_data, STOCKS,
        seq_length=SEQ_LENGTH, forecast_horizon=forecast_horizon, graph_method="correlation",
        normalize=False, config=config, news_panel_path=None,
    )


class TestTargetShift22Day:
    def test_window0_target_matches_22day_formula(self):
        ds = _build_ds(forecast_horizon=22)
        _x_har, _adj, _x_news, y = ds[0]
        expected_target_idx = 0 + SEQ_LENGTH + 22 - 1  # = 31
        assert y[0].item() == pytest.approx(31.0)
        assert y[1].item() == pytest.approx(31.0)

    def test_window0_target_does_NOT_match_5day_or_10day_formula(self):
        ds = _build_ds(forecast_horizon=22)
        _x_har, _adj, _x_news, y = ds[0]
        assert y[0].item() != pytest.approx(0 + SEQ_LENGTH + 5 - 1)    # 14
        assert y[0].item() != pytest.approx(0 + SEQ_LENGTH + 10 - 1)   # 19

    def test_window_count_shrinks_relative_to_5day_and_10day(self):
        ds5 = _build_ds(forecast_horizon=5)
        ds10 = _build_ds(forecast_horizon=10)
        ds22 = _build_ds(forecast_horizon=22)
        assert len(ds22) == len(ds5) - 17   # 22-5
        assert len(ds22) == len(ds10) - 12  # 22-10
        assert len(ds22) > 0

    def test_second_window_target_advances_by_one_day(self):
        ds = _build_ds(forecast_horizon=22)
        _x_har, _adj, _x_news, y0 = ds[0]
        _x_har, _adj, _x_news, y1 = ds[1]
        assert y1[0].item() == pytest.approx(y0[0].item() + 1.0)
