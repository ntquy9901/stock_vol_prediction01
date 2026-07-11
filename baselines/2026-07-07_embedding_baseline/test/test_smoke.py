"""Smoke test: full EmbeddingBaseline forward + backward with dummy tensors (no data needed).

Verifies: HAR branch reuse + news branch + concat fusion produce correct output shape,
no NaN, and gradients flow into the news branch.

Run: python baselines/2026-07-07_embedding_baseline/test/test_smoke.py
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
from model_embedding import EmbeddingBaseline

# Module-level smoke marker for the happy-path tests below (CLAUDE.md smoke gate)
pytestmark = pytest.mark.smoke


def test_forward_shape_and_backward():
    config = LSTMGATConfig()
    config.num_features_per_stock = 3   # HAR only
    model = EmbeddingBaseline(config, emb_dim=16, d_news=16, dropout=0.0)
    model.train()

    B, T, S_stocks, A = 2, 4, 3, 5
    x_har = torch.randn(B, T, S_stocks, 3)
    adj = torch.rand(B, S_stocks, S_stocks)
    x_emb = torch.randn(B, T, S_stocks, A, 16)
    mask = torch.ones(B, T, S_stocks, A)

    pred = model(x_har, adj, x_emb, mask)
    assert pred.shape == (B, S_stocks), f"expected {(B, S_stocks)}, got {pred.shape}"
    assert not torch.isnan(pred).any(), "NaN in predictions"

    # Backward + gradient flow check on news branch
    pred.sum().backward()
    assert model.news_pool.query.grad is not None, "no grad on attention query"
    assert model.news_temporal.lstm.weight_ih_l0.grad is not None, "no grad on news LSTM"
    assert model.fusion[0].weight.grad is not None, "no grad on fusion"
    print(f"forward shape OK: {pred.shape}; backward OK (grads flow into news branch)")


def test_zero_news_day_does_not_break_forward():
    """All-masked news (every stock-day has 0 articles) must still forward cleanly."""
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    model = EmbeddingBaseline(config, emb_dim=8, d_news=8, dropout=0.0).eval()
    B, T, S_stocks, A = 1, 3, 2, 4
    x_har = torch.randn(B, T, S_stocks, 3)
    adj = torch.rand(B, S_stocks, S_stocks)
    x_emb = torch.randn(B, T, S_stocks, A, 8)
    mask = torch.zeros(B, T, S_stocks, A)   # NO news anywhere
    pred = model(x_har, adj, x_emb, mask)
    assert not torch.isnan(pred).any(), "NaN when all news masked"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
