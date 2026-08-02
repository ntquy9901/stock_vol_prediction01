"""Smoke test: full MarketFallbackBaseline forward + backward with dummy 7-tuple (no data).

Run: pytest baselines/2026-07-08_market_fallback/test/test_smoke.py -v
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from src.lstm_gat_hybrid.config import LSTMGATConfig
from market_fallback_model_embedding import MarketFallbackBaseline


def test_forward_shape_and_backward():
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = MarketFallbackBaseline(config, emb_dim=16, d_news=16, dropout=0.0)
    model.train()

    B, T, S, MAX_A, MAX_M, D = 2, 4, 3, 10, 15, 16
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_emb = torch.randn(B, T, S, MAX_A, D)
    mask = torch.ones(B, T, S, MAX_A)
    # half the stock-days have no news (mask=0) -> exercises the gate fallback
    mask[:, :, 0, :] = 0
    x_market = torch.randn(B, T, MAX_M, D)
    market_mask = torch.ones(B, T, MAX_M)

    pred = model(x_har, adj, x_emb, mask, x_market, market_mask)
    assert pred.shape == (B, S), f"expected {(B, S)}, got {pred.shape}"
    assert not torch.isnan(pred).any(), "NaN in predictions"

    pred.sum().backward()
    assert model.market_branch.pool.query.grad is not None, "no grad on market query"
    assert model.news_pool.query.grad is not None, "no grad on per-stock query"
    assert model.news_temporal.lstm.weight_ih_l0.grad is not None, "no grad on news LSTM"
    assert model.fusion[0].weight.grad is not None, "no grad on fusion"
    print(f"forward shape OK: {pred.shape}; backward OK (grads flow into market + per-stock + fusion)")


def test_all_markets_zero_news():
    """Every stock-day blind + market present -> output must still be finite (market fallback works)."""
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = MarketFallbackBaseline(config, emb_dim=8, d_news=8, dropout=0.0).eval()
    B, T, S, MAX_A, MAX_M, D = 1, 3, 2, 10, 15, 8
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_emb = torch.randn(B, T, S, MAX_A, D)
    mask = torch.zeros(B, T, S, MAX_A)        # NO per-stock news anywhere
    x_market = torch.randn(B, T, MAX_M, D)    # market present
    market_mask = torch.ones(B, T, MAX_M)
    pred = model(x_har, adj, x_emb, mask, x_market, market_mask)
    assert not torch.isnan(pred).any(), "NaN when all per-stock news is masked"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
