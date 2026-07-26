"""Smoke test: MultiStockDatasetWithDualNews shapes + date-matching against a tiny synthetic
news panel parquet (written to tmp_path — exercises the real load_news_panel/_norm_date/
fillna(0.0) code path end to end, not just a mocked dict).

Run: pytest baselines/2026-07-25_dual_group_news_embedding_baseline/test/test_dataset_smoke.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset_dual_news import MultiStockDatasetWithDualNews, HAR_COLS  # noqa: E402
from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402

pytestmark = pytest.mark.smoke

STOCKS = ["AAA", "BBB"]
N_DAYS = 40
N_FEAT = 5


def _make_har_df(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
    return pd.DataFrame({
        "date": dates.astype(str),
        "parkinson_volatility": rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[0]: rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[1]: rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[2]: rng.uniform(0.001, 0.05, N_DAYS),
    })


def _make_news_panel(tmp_path) -> Path:
    """Tiny 2-ticker panel with a FEW dated rows (not every day) so the fillna(0.0)/no-match
    path is also exercised, matching the real panel's sparsity (NaN where no news)."""
    rows = []
    for ticker in STOCKS:
        for d in ["2024-01-05", "2024-01-10", "2024-01-20"]:
            rows.append({"ticker": ticker, "date": d,
                        **{f"kq_emb_{i}": float(i + hash(ticker + d) % 7) for i in range(N_FEAT)}})
    df = pd.DataFrame(rows)
    path = tmp_path / "dual_group_news_panel.parquet"
    df.to_parquet(path, index=False)
    return path


def test_shapes_and_coverage(tmp_path):
    stock_data_with_har = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
    news_panel_path = _make_news_panel(tmp_path)

    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    ds = MultiStockDatasetWithDualNews(
        stock_data_with_har, stock_data_with_har, STOCKS,
        seq_length=10, forecast_horizon=5, graph_method="correlation",
        normalize=False, config=config, news_panel_path=news_panel_path,
    )

    assert len(ds) > 0, "expected at least one sequence"
    assert ds._n_feat == N_FEAT

    x_har, adj, x_news, y = ds[0]
    S = len(STOCKS)
    assert x_har.shape == (10, S, 3)
    assert adj.shape == (S, S)
    assert x_news.shape == (10, S, N_FEAT)
    assert y.shape == (S,)
    assert ds._matched_cells > 0, "expected >=1 real date match against the synthetic panel"
    assert ds._matched_cells < ds._total_cells, "expected >=1 no-news (zero-filled) day too"


def test_missing_panel_falls_back_to_dummy_zero_features():
    """No panel file -> dummy n_feat, all-zero x_news, and no crash (smoke-mode parity with
    the original embedding_baseline's dummy emb_dim fallback)."""
    stock_data_with_har = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    ds = MultiStockDatasetWithDualNews(
        stock_data_with_har, stock_data_with_har, STOCKS,
        seq_length=10, forecast_horizon=5, graph_method="correlation",
        normalize=False, config=config, news_panel_path=None,
    )
    x_har, adj, x_news, y = ds[0]
    assert x_news.shape == (10, len(STOCKS), 146)
    assert (x_news == 0).all()


if __name__ == "__main__":
    print("Run with pytest for tmp_path fixture support.")
