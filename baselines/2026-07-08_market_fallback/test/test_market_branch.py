"""Tests for MarketBranch: shape, permutation invariance, 0-news day.

Run: pytest baselines/2026-07-08_market_fallback/test/test_market_branch.py -v
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from model_embedding import MarketBranch


def test_shape():
    br = MarketBranch(emb_dim=16, d_news=8)
    B, T, M, D = 2, 4, 10, 16
    out = br(torch.randn(B, T, M, D), torch.ones(B, T, M))
    assert out.shape == (B, T, 8), f"expected {(B, T, 8)}, got {out.shape}"


def test_permutation_invariance():
    """Shuffling articles within a day must not change that day's market vector."""
    br = MarketBranch(emb_dim=8, d_news=4).eval()
    torch.manual_seed(0)
    M, D = 6, 8
    emb = torch.randn(1, 1, M, D)
    mask = torch.ones(1, 1, M)
    out1 = br(emb, mask)
    perm = torch.randperm(M)
    out2 = br(emb[:, :, perm, :], mask[:, :, perm])
    assert torch.allclose(out1, out2, atol=1e-6), "market branch not permutation-invariant"


def test_zero_news_day_uses_token():
    """A day with 0 articles must emit the no_news_token (and no NaN)."""
    br = MarketBranch(emb_dim=8, d_news=4).eval()
    emb = torch.randn(1, 1, 3, 8) * 100
    mask = torch.zeros(1, 1, 3)
    out = br(emb, mask)
    assert not torch.isnan(out).any(), "NaN in 0-article market day"
    tok = br.pool.no_news_token.detach()
    assert torch.allclose(out[0, 0], tok, atol=1e-6), "0-article day did not return no_news_token"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} market-branch tests passed.")
