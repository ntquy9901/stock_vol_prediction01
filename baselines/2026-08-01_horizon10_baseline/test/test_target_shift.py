"""Proves the target actually shifts to 10-days-ahead (not the project default of 5), using the
UNCHANGED sibling dataset class `MultiStockDatasetWithDualNews` (read-only import) -- this
baseline adds NO new dataset code, only a `forecast_horizon` kwarg passed differently (design.md
§1). This test is the acceptance criterion that proves the kwarg actually took effect, not just
"the code ran without crashing."

Written BEFORE the two train scripts exist (test-first, CLAUDE.md §5 SDD Implement phase) --
importing the sibling dataset class directly, so this test can run standalone.
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
N_DAYS = 60
SEQ_LENGTH = 10


def _make_har_df(seed: int) -> pd.DataFrame:
    """parkinson_variance is a KNOWN, distinct sequence (0, 1, 2, ..., 59) so the exact target
    value at any index can be checked by hand, not just "some plausible float"."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
    return pd.DataFrame({
        "date": dates.astype(str),
        "parkinson_variance": np.arange(N_DAYS, dtype=np.float64),  # 0.0, 1.0, 2.0, ...
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


class TestTargetShift10Day:
    def test_window0_target_matches_10day_formula(self):
        ds = _build_ds(forecast_horizon=10)
        _x_har, _adj, _x_news, y = ds[0]
        expected_target_idx = 0 + SEQ_LENGTH + 10 - 1  # = 19
        # parkinson_variance is np.arange(N_DAYS), so value at index 19 is exactly 19.0
        assert y[0].item() == pytest.approx(19.0)
        assert y[1].item() == pytest.approx(19.0)  # same for both tickers (same synthetic series)

    def test_window0_target_does_NOT_match_5day_formula(self):
        """Guards against silently falling back to the project's forecast_horizon=5 default --
        if this ever passes, the horizon=10 kwarg is being ignored somewhere upstream."""
        ds = _build_ds(forecast_horizon=10)
        _x_har, _adj, _x_news, y = ds[0]
        target_idx_5day = 0 + SEQ_LENGTH + 5 - 1  # = 14
        assert y[0].item() != pytest.approx(14.0)

    def test_default_horizon_still_5_when_unspecified(self):
        """Sanity check: the sibling dataset's OWN default is untouched (still 5) -- this
        baseline only ever passes forecast_horizon=10 explicitly, never relies on/changes the
        shared default (CLAUDE.md hard-isolation rule)."""
        stock_data = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
        config = LSTMGATConfig()
        config.num_features_per_stock = 3
        ds = MultiStockDatasetWithDualNews(
            stock_data, stock_data, STOCKS,
            seq_length=SEQ_LENGTH, graph_method="correlation",  # forecast_horizon NOT passed
            normalize=False, config=config, news_panel_path=None,
        )
        _x_har, _adj, _x_news, y = ds[0]
        expected_target_idx_5day = 0 + SEQ_LENGTH + 5 - 1  # = 14
        assert y[0].item() == pytest.approx(14.0)

    def test_window_count_shrinks_relative_to_5day(self):
        ds5 = _build_ds(forecast_horizon=5)
        ds10 = _build_ds(forecast_horizon=10)
        assert len(ds10) == len(ds5) - 5
        assert len(ds10) > 0  # still usable with this synthetic 60-day series

    def test_second_window_target_advances_by_one_day(self):
        ds = _build_ds(forecast_horizon=10)
        _x_har, _adj, _x_news, y0 = ds[0]
        _x_har, _adj, _x_news, y1 = ds[1]
        assert y1[0].item() == pytest.approx(y0[0].item() + 1.0)
