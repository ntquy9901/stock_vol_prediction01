"""Tests for ArticleSetAttentionPooling: shape, permutation invariance, 0-news, mask.

Run: python baselines/2026-07-07_embedding_baseline/test/test_attention_pooling.py
  or: pytest baselines/2026-07-07_embedding_baseline/test/test_attention_pooling.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"   # sibling code/ dir
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from model_embedding import ArticleSetAttentionPooling


def test_shape():
    pool = ArticleSetAttentionPooling(emb_dim=64, d_news=32)
    B, S, T, A, D = 2, 3, 4, 5, 64
    out = pool(torch.randn(B, S, T, A, D), torch.ones(B, S, T, A))
    assert out.shape == (B, S, T, 32), f"expected {(B, S, T, 32)}, got {out.shape}"


def test_permutation_invariance():
    """Shuffling articles within a (stock, day) must not change the output."""
    pool = ArticleSetAttentionPooling(emb_dim=8, d_news=4).eval()
    torch.manual_seed(0)
    A, D = 5, 8
    emb = torch.randn(1, 1, 1, A, D)
    mask = torch.ones(1, 1, 1, A)
    out1 = pool(emb, mask)
    perm = torch.randperm(A)
    out2 = pool(emb[:, :, :, perm, :], mask[:, :, :, perm])
    assert torch.allclose(out1, out2, atol=1e-6), \
        f"not permutation-invariant: max diff {(out1 - out2).abs().max()}"


def test_zero_news_uses_token():
    """A day with 0 real articles must emit the learned no_news_token (no NaN)."""
    pool = ArticleSetAttentionPooling(emb_dim=8, d_news=4).eval()
    emb = torch.randn(1, 1, 1, 3, 8) * 100
    mask = torch.zeros(1, 1, 1, 3)
    out = pool(emb, mask)
    assert not torch.isnan(out).any(), "NaN in 0-news output"
    assert torch.allclose(out[0, 0, 0], pool.no_news_token.detach(), atol=1e-6), \
        "0-news day did not return no_news_token"


def test_masked_article_ignored():
    """A masked article (mask=0) must not influence the output, even if huge."""
    pool = ArticleSetAttentionPooling(emb_dim=8, d_news=4).eval()
    torch.manual_seed(1)
    emb = torch.randn(1, 1, 1, 3, 8)
    mask = torch.tensor([[[[1.0, 1.0, 0.0]]]])   # 3rd article masked
    out_ref = pool(emb, mask)
    emb2 = emb.clone()
    emb2[:, :, :, 2, :] = 1e6                      # poison the masked slot
    out_poison = pool(emb2, mask)
    assert torch.allclose(out_ref, out_poison, atol=1e-3), \
        "masked article leaked into output"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} attention-pooling tests passed.")
