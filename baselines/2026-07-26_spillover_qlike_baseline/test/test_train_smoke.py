"""Smoke tests: MultiStockDatasetWithSpilloverNews (directed graph end-to-end) + one real
train_epoch/validate step using the combined MSE+QLIKE loss, on synthetic data (no real
data/processed or news cache needed). Mirrors the sibling baseline's
test_dataset_smoke.py/test_model_smoke.py conventions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from dataset_spillover_news import MultiStockDatasetWithSpilloverNews, HAR_COLS  # noqa: E402
from losses import build_denorm_tensors, combined_loss  # noqa: E402
from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402
from src.common.data_normalization import VolatilityNormalizer  # noqa: E402
from model_dual_news import DualGroupNewsBaseline  # noqa: E402

pytestmark = pytest.mark.smoke

STOCKS = ["AAA", "BBB", "CCC"]
N_DAYS = 60
N_FEAT = 5


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


def _make_dataset(graph_method="spillover", normalize=False):
    stock_data = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return MultiStockDatasetWithSpilloverNews(
        stock_data, stock_data, STOCKS,
        seq_length=10, forecast_horizon=5, graph_method=graph_method,
        k_neighbors=2, normalize=normalize, config=config, news_panel_path=None,
    )


def test_spillover_graph_shapes_end_to_end():
    ds = _make_dataset()
    assert len(ds) > 0
    x_har, adj, x_news, y = ds[0]
    S = len(STOCKS)
    assert x_har.shape == (10, S, 3)
    assert adj.shape == (S, S)
    assert x_news.shape == (10, S, 146)  # dummy n_feat, no panel
    assert y.shape == (S,)


def test_spillover_graph_can_be_asymmetric_across_dataset():
    """At least one sequence window in a realistic-length series should produce an asymmetric
    adjacency matrix -- this is the whole point of graph_method='spillover' vs. the sibling's
    always-symmetric 'correlation'/'knn'."""
    ds = _make_dataset()
    found_asymmetric = False
    for idx in range(len(ds)):
        _, adj, _, _ = ds[idx]
        adj_np = adj.numpy()
        if not np.allclose(adj_np, adj_np.T):
            found_asymmetric = True
            break
    assert found_asymmetric, "expected at least one asymmetric adjacency across all windows"


def test_correlation_graph_method_still_available_for_comparison():
    """graph_method='correlation' (sibling's original) still works via this dataset class --
    used for direct comparison runs, not just 'spillover'."""
    ds = _make_dataset(graph_method="correlation")
    _, adj, _, _ = ds[0]
    assert adj.shape == (len(STOCKS), len(STOCKS))


def test_unknown_graph_method_raises():
    with pytest.raises(ValueError):
        _make_dataset(graph_method="not_a_real_method")


def test_train_epoch_smoke_with_combined_loss():
    """One real forward+backward+optimizer.step() using the actual combined_loss (MSE+QLIKE),
    directed spillover graph, and DualGroupNewsBaseline model -- end-to-end smoke."""
    ds = _make_dataset(normalize=True)
    for sn in STOCKS:
        ds.feature_normalizers[sn] = VolatilityNormalizer()
        ds.target_normalizers[sn] = VolatilityNormalizer()
    for s_idx, sn in enumerate(ds.stock_names):
        feats = np.concatenate([x_har[:, s_idx, :] for x_har, _, _, _ in ds.sequences], axis=0)
        targs = np.array([y[s_idx] for _, _, _, y in ds.sequences])
        ds.feature_normalizers[sn].fit(feats)
        ds.target_normalizers[sn].fit(targs.reshape(-1, 1))

    loader = DataLoader(ds, batch_size=4, shuffle=False)
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = DualGroupNewsBaseline(config, n_feat=146, d_news=8, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mean_t, std_t = build_denorm_tensors(ds, device="cpu")

    x_har, adj, x_news, y = next(iter(loader))
    pred = model(x_har, adj, x_news)
    loss = combined_loss(pred, y, mean_t, std_t, qlike_weight=0.1)
    assert torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"train_epoch smoke OK: loss={loss.item():.6f}")


if __name__ == "__main__":
    print("Run with pytest for full fixture/tmp_path support.")
