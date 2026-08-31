"""Smoke test: MultiStockDatasetWithMacroNews shapes + coverage against tiny synthetic dual-group
and macro panel parquets (written to tmp_path — exercises the real load_news_panel/
load_macro_panel/_norm_date/fillna(0.0)/concat code path end to end).

Run: pytest baselines/2026-07-25_macro_news_baseline/test/test_dataset_smoke.py -v
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

from dataset_macro_news import (  # noqa: E402
    MultiStockDatasetWithMacroNews, load_macro_panel, HAR_COLS,
)
from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402

pytestmark = pytest.mark.smoke

STOCKS = ["AAA", "BBB"]
N_DAYS = 40
N_DUAL_FEAT = 5
N_MACRO_FEAT = 3


def _make_har_df(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
    return pd.DataFrame({
        "date": dates.astype(str),
        "parkinson_variance": rng.uniform(0.001, 0.05, N_DAYS),
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


def _make_macro_panel(tmp_path) -> Path:
    """Date-only (no ticker column) — a FEW dated rows so the fillna(0.0)/no-match path is
    exercised too."""
    rows = []
    for d in ["2024-01-05", "2024-01-15"]:  # deliberately different dates than the dual panel
        rows.append({"date": d, **{f"macro_emb_{i}": float(i + 1) for i in range(N_MACRO_FEAT)}})
    df = pd.DataFrame(rows)
    path = tmp_path / "macro_news_panel.parquet"
    df.to_parquet(path, index=False)
    return path


def test_load_macro_panel_shapes_and_fillna(tmp_path):
    path = _make_macro_panel(tmp_path)
    by_date, cols = load_macro_panel(path)
    assert cols == [f"macro_emb_{i}" for i in range(N_MACRO_FEAT)]
    assert set(by_date.keys()) == {"2024-01-05", "2024-01-15"}
    assert by_date["2024-01-05"].shape == (N_MACRO_FEAT,)


def test_shapes_and_coverage(tmp_path):
    stock_data_with_har = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
    dual_panel_path = _make_dual_panel(tmp_path)
    macro_panel_path = _make_macro_panel(tmp_path)

    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    ds = MultiStockDatasetWithMacroNews(
        stock_data_with_har, stock_data_with_har, STOCKS,
        seq_length=10, forecast_horizon=5, graph_method="correlation",
        normalize=False, config=config,
        news_panel_path=dual_panel_path, macro_panel_path=macro_panel_path,
    )

    assert len(ds) > 0
    assert ds._n_feat == N_DUAL_FEAT + N_MACRO_FEAT

    x_har, adj, x_news, y = ds[0]
    S = len(STOCKS)
    assert x_har.shape == (10, S, 3)
    assert adj.shape == (S, S)
    assert x_news.shape == (10, S, N_DUAL_FEAT + N_MACRO_FEAT)
    assert y.shape == (S,)
    assert ds._matched_dual_cells > 0
    assert ds._matched_macro_cells > 0
    assert ds._matched_dual_cells < ds._total_cells
    assert ds._matched_macro_cells < ds._total_cells


def test_macro_vector_is_identical_across_tickers_for_same_date(tmp_path):
    """The whole point of the macro feature: broadcast the SAME vector to every ticker at a
    given date (unlike the dual-group vector, which differs per ticker)."""
    stock_data_with_har = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
    dual_panel_path = _make_dual_panel(tmp_path)
    macro_panel_path = _make_macro_panel(tmp_path)

    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    ds = MultiStockDatasetWithMacroNews(
        stock_data_with_har, stock_data_with_har, STOCKS,
        seq_length=10, forecast_horizon=5, graph_method="correlation",
        normalize=False, config=config,
        news_panel_path=dual_panel_path, macro_panel_path=macro_panel_path,
    )
    x_har, adj, x_news, y = ds[0]
    # last N_MACRO_FEAT cols are the macro part; must be identical across the stock dimension
    macro_part = x_news[:, :, N_DUAL_FEAT:]
    for t in range(macro_part.shape[0]):
        assert (macro_part[t, 0] == macro_part[t, 1]).all(), \
            "macro vector must be broadcast identically across all tickers at a given date"


def test_missing_panels_fall_back_to_dummy_zero_features():
    stock_data_with_har = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    ds = MultiStockDatasetWithMacroNews(
        stock_data_with_har, stock_data_with_har, STOCKS,
        seq_length=10, forecast_horizon=5, graph_method="correlation",
        normalize=False, config=config, news_panel_path=None, macro_panel_path=None,
    )
    x_har, adj, x_news, y = ds[0]
    assert x_news.shape == (10, len(STOCKS), 146 + 66)
    assert (x_news == 0).all()


if __name__ == "__main__":
    print("Run with pytest for tmp_path fixture support.")
