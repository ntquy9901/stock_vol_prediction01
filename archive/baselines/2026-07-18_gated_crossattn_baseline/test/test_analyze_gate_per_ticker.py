"""Smoke test: extract_gate_per_ticker aggregation (analyze_gate_per_ticker.py).

Run: pytest baselines/2026-07-18_gated_crossattn_baseline/test/test_analyze_gate_per_ticker.py -v
"""
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.lstm_gat_hybrid.config import LSTMGATConfig
from model_gated_crossattn import GatedCrossAttnBaseline
from analyze_gate_per_ticker import extract_gate_per_ticker

pytestmark = pytest.mark.smoke


def _build(emb_dim=16, d_news=16, num_heads=2):
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return GatedCrossAttnBaseline(config, emb_dim=emb_dim, d_news=d_news, num_heads=num_heads,
                                  dropout=0.0)


def _make_loader(n_windows=6, T=4, S=3, A=5, D=16, batch_size=2):
    x_har = torch.randn(n_windows, T, S, 3)
    adj = torch.rand(n_windows, S, S)
    x_emb = torch.randn(n_windows, T, S, A, D)
    mask = torch.ones(n_windows, T, S, A)
    y = torch.randn(n_windows, S)
    ds = TensorDataset(x_har, adj, x_emb, mask, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def test_extract_gate_per_ticker_shape_and_range():
    model = _build().eval()
    stock_names = ["AAA", "BBB", "CCC"]
    loader = _make_loader(n_windows=6, S=len(stock_names))

    gates = extract_gate_per_ticker(model, loader, stock_names, device="cpu")

    assert set(gates.keys()) == set(stock_names)
    for s in stock_names:
        assert gates[s].shape == (6,), f"expected 6 windows for {s}, got {gates[s].shape}"
        assert (gates[s] >= 0.0).all() and (gates[s] <= 1.0).all(), \
            f"gate out of [0,1] for {s}"


def test_extract_gate_per_ticker_differs_across_tickers_when_inputs_differ():
    """Different HAR/news inputs per ticker should generally produce different gate values
    (regression guard: aggregation must not silently collapse all tickers to the same column)."""
    torch.manual_seed(0)
    model = _build().eval()
    stock_names = ["AAA", "BBB"]
    T, S, A, D = 4, 2, 5, 16
    x_har = torch.randn(4, T, S, 3)
    x_har[:, :, 1, :] += 50.0  # make ticker BBB's HAR input wildly different
    adj = torch.rand(4, S, S)
    x_emb = torch.randn(4, T, S, A, D)
    mask = torch.ones(4, T, S, A)
    y = torch.randn(4, S)
    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(TensorDataset(x_har, adj, x_emb, mask, y), batch_size=2)

    gates = extract_gate_per_ticker(model, loader, stock_names, device="cpu")
    assert not torch.allclose(torch.tensor(gates["AAA"]), torch.tensor(gates["BBB"])), \
        "gate values identical across tickers with very different inputs — aggregation bug"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
