"""Smoke test: full PureMarketBaseline forward + backward with dummy tensors (no data needed).

Verifies: HAR branch + market branch (broadcast, no ticker match) produce correct output
shape, no NaN, gradients flow, and the market contribution is IDENTICAL across the stock
dimension (the core architectural property of "broadcast, not gated/per-stock").

Run: pytest baselines/2026-07-15_pure_market_baseline/test/test_smoke.py -v
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
from src.lstm_gat_hybrid.config import LSTMGATConfig
from model_pure_market import PureMarketBaseline

pytestmark = pytest.mark.smoke


def test_forward_shape_and_backward():
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = PureMarketBaseline(config, emb_dim=16, d_news=16, dropout=0.0)
    model.train()

    B, T, S, MAX_M, D = 2, 4, 3, 15, 16
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_market = torch.randn(B, T, MAX_M, D)
    market_mask = torch.ones(B, T, MAX_M)

    pred = model(x_har, adj, x_market, market_mask)
    assert pred.shape == (B, S), f"expected {(B, S)}, got {pred.shape}"
    assert not torch.isnan(pred).any(), "NaN in predictions"

    pred.sum().backward()
    assert model.market_pool.query.grad is not None, "no grad on market attention query"
    assert model.market_temporal.lstm.weight_ih_l0.grad is not None, "no grad on market LSTM"
    assert model.fusion[0].weight.grad is not None, "no grad on fusion"


def test_zero_market_day_does_not_break_forward():
    """All-masked market news (every day has 0 articles) must still forward cleanly."""
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = PureMarketBaseline(config, emb_dim=8, d_news=8, dropout=0.0).eval()
    B, T, S, MAX_M, D = 1, 3, 2, 15, 8
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_market = torch.randn(B, T, MAX_M, D)
    market_mask = torch.zeros(B, T, MAX_M)   # NO market news anywhere
    pred = model(x_har, adj, x_market, market_mask)
    assert not torch.isnan(pred).any(), "NaN when all market news masked"


def test_market_contribution_identical_across_stocks():
    """Core property: the market vector concatenated per-stock must be IDENTICAL for every
    stock (broadcast, no gate, no per-stock routing) — only h_lstm/h_gnn differ per stock.

    Verified by hooking the REAL fusion input inside forward() (not a hand-rolled duplicate
    of the broadcast logic) so a bug in forward()'s actual wiring would be caught here."""
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = PureMarketBaseline(config, emb_dim=8, d_news=8, dropout=0.0).eval()
    B, T, S, MAX_M, D = 1, 3, 4, 15, 8
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_market = torch.randn(B, T, MAX_M, D)
    market_mask = torch.ones(B, T, MAX_M)

    captured = {}

    def _capture(module, args):
        captured["fusion_input"] = args[0].detach().clone()

    handle = model.fusion.register_forward_pre_hook(_capture)
    try:
        model(x_har, adj, x_market, market_mask)
    finally:
        handle.remove()

    fusion_in = captured["fusion_input"]     # [B, S, 64+256+8]
    market_slice = fusion_in[..., -8:]        # last d_news=8 columns = market_bc
    for s in range(1, S):
        assert torch.allclose(market_slice[:, 0], market_slice[:, s], atol=1e-6), \
            f"market broadcast differs at stock {s} — should be identical for all stocks"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
