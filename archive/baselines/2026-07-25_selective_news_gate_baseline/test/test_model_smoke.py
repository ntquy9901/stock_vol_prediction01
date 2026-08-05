"""Smoke test: full SelectiveGateNewsBaseline forward + backward, correct shape, no NaN, and the
news branch still receives gradient (from NEWS_ON stocks) despite some stocks being masked.

Run: pytest baselines/2026-07-25_selective_news_gate_baseline/test/test_model_smoke.py -v
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
from model_selective_gate import SelectiveGateNewsBaseline

pytestmark = pytest.mark.smoke


def test_forward_shape_and_backward():
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    stock_names = ["ACB", "SHB", "GAS", "VIB"]   # 2 ON (ACB, VIB), 2 OFF (SHB, GAS)
    n_feat = 32
    model = SelectiveGateNewsBaseline(config, n_feat=n_feat, stock_names=stock_names,
                                      d_news=16, dropout=0.0)
    model.train()

    B, T, S = 2, 4, len(stock_names)
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_news = torch.randn(B, T, S, n_feat)

    pred = model(x_har, adj, x_news)
    assert pred.shape == (B, S)
    assert not torch.isnan(pred).any()

    pred.sum().backward()
    assert model.news_branch.lstm.weight_ih_l0.grad is not None, "no grad on news LSTM (should still train on NEWS_ON stocks)"
    assert model.fusion[0].weight.grad is not None
    assert model.stock_mask.grad is None, "stock_mask must be a fixed buffer, not learnable"


def test_all_stocks_masked_off_does_not_break_forward():
    """Degenerate case: every stock in this batch is NEWS_OFF -> news_rep fully zeroed, model
    must still forward cleanly (falls back to pure HAR signal)."""
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    stock_names = ["SHB", "GAS"]   # both OFF
    model = SelectiveGateNewsBaseline(config, n_feat=8, stock_names=stock_names,
                                      d_news=8, dropout=0.0).eval()
    assert model.stock_mask.sum().item() == 0.0

    B, T, S = 1, 3, 2
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_news = torch.randn(B, T, S, 8)
    pred = model(x_har, adj, x_news)
    assert not torch.isnan(pred).any()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
