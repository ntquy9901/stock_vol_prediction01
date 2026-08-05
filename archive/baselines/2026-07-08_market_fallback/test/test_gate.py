"""Tests for GatedNewsFusion: deterministic availability gate.

Run: pytest baselines/2026-07-08_market_fallback/test/test_gate.py -v
  or: python baselines/2026-07-08_market_fallback/test/test_gate.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from market_fallback_model_embedding import GatedNewsFusion


def test_has_news_uses_stock():
    """g=1 -> output = stock_daily (market ignored)."""
    fuse = GatedNewsFusion()
    B, T, S, D = 1, 1, 2, 3
    stock = torch.randn(B, T, S, D)
    market = torch.randn(B, T, D) * 100   # deliberately huge
    has_news = torch.ones(B, T, S, 1)
    out = fuse(stock, market, has_news)
    assert torch.allclose(out, stock), "g=1 should pass stock_daily through unchanged"


def test_no_news_uses_market():
    """g=0 -> output = market broadcast to all stocks (stock ignored)."""
    fuse = GatedNewsFusion()
    B, T, S, D = 1, 1, 3, 4
    stock = torch.randn(B, T, S, D) * 100   # deliberately huge
    market = torch.randn(B, T, D)
    has_news = torch.zeros(B, T, S, 1)
    out = fuse(stock, market, has_news)
    expected = market.unsqueeze(2).expand(B, T, S, D)
    assert torch.allclose(out, expected), "g=0 should broadcast market to all stocks"


def test_mixed_gate():
    """One stock has news, another doesn't -> each gets the right source."""
    fuse = GatedNewsFusion()
    B, T, S, D = 1, 1, 2, 2
    stock = torch.tensor([[[[1.0, 1.0], [2.0, 2.0]]]])     # stock0=(1,1), stock1=(2,2)
    market = torch.tensor([[[9.0, 9.0]]])                  # market=(9,9)
    has_news = torch.tensor([[[[1.0], [0.0]]]])            # stock0 has, stock1 not
    out = fuse(stock, market, has_news)
    assert torch.allclose(out[0, 0, 0], torch.tensor([1.0, 1.0])), "stock0 should use stock"
    assert torch.allclose(out[0, 0, 1], torch.tensor([9.0, 9.0])), "stock1 should use market"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} gate tests passed.")
