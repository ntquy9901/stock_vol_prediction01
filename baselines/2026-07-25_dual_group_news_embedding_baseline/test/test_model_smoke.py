"""Smoke test: full DualGroupNewsBaseline forward + backward with dummy tensors (no data needed).

Verifies: HAR branch reuse + news branch (NewsFeatureLSTM, no pooling/mask needed) + concat
fusion produce correct output shape, no NaN, and gradients flow into the news branch.

Run: pytest baselines/2026-07-25_dual_group_news_embedding_baseline/test/test_model_smoke.py -v
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
from model_dual_news import DualGroupNewsBaseline

pytestmark = pytest.mark.smoke


def test_forward_shape_and_backward():
    config = LSTMGATConfig()
    config.num_features_per_stock = 3   # HAR only
    n_feat = 146
    model = DualGroupNewsBaseline(config, n_feat=n_feat, d_news=16, dropout=0.0)
    model.train()

    B, T, S_stocks = 2, 4, 3
    x_har = torch.randn(B, T, S_stocks, 3)
    adj = torch.rand(B, S_stocks, S_stocks)
    x_news = torch.randn(B, T, S_stocks, n_feat)

    pred = model(x_har, adj, x_news)
    assert pred.shape == (B, S_stocks), f"expected {(B, S_stocks)}, got {pred.shape}"
    assert not torch.isnan(pred).any(), "NaN in predictions"

    pred.sum().backward()
    assert model.news_branch.lstm.weight_ih_l0.grad is not None, "no grad on news LSTM"
    assert model.news_branch.proj.weight.grad is not None, "no grad on news proj"
    assert model.fusion[0].weight.grad is not None, "no grad on fusion"
    print(f"forward shape OK: {pred.shape}; backward OK (grads flow into news branch)")


def test_all_zero_news_day_does_not_break_forward():
    """All-zero news vectors (every stock-day has no news, per the fillna(0.0) convention in
    dataset_dual_news.load_news_panel) must still forward cleanly."""
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    n_feat = 32
    model = DualGroupNewsBaseline(config, n_feat=n_feat, d_news=8, dropout=0.0).eval()
    B, T, S_stocks = 1, 3, 2
    x_har = torch.randn(B, T, S_stocks, 3)
    adj = torch.rand(B, S_stocks, S_stocks)
    x_news = torch.zeros(B, T, S_stocks, n_feat)   # NO news anywhere
    pred = model(x_har, adj, x_news)
    assert not torch.isnan(pred).any(), "NaN when all news is zero"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
