"""Smoke test: AlignmentLossBaseline forward + alignment_loss properties.

Run: pytest baselines/2026-07-18_alignment_loss_baseline/test/test_smoke.py -v
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
from model_alignment import AlignmentLossBaseline, alignment_loss

pytestmark = pytest.mark.smoke


def _build(dropout=0.0, emb_dim=16, d_news=16, d_align=8):
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return AlignmentLossBaseline(config, emb_dim=emb_dim, d_news=d_news, d_align=d_align,
                                 dropout=dropout)


def test_forward_shapes():
    model = _build().train()
    B, T, S, A, D = 2, 4, 3, 5, 16
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_emb = torch.randn(B, T, S, A, D)
    mask = torch.ones(B, T, S, A)

    pred, proj_har, proj_news = model(x_har, adj, x_emb, mask)
    assert pred.shape == (B, S)
    assert proj_har.shape == (B, S, 8)
    assert proj_news.shape == (B, S, 8)
    assert not torch.isnan(pred).any()


def test_alignment_loss_range_and_identity():
    """Cosine-based alignment loss must be in [0, 2]; identical vectors -> loss ~0."""
    v = torch.nn.functional.normalize(torch.randn(2, 3, 8), dim=-1)
    loss_same = alignment_loss(v, v)
    assert loss_same.item() < 1e-5, f"identical projections should give ~0 loss, got {loss_same.item()}"

    v2 = torch.nn.functional.normalize(torch.randn(2, 3, 8), dim=-1)
    loss_rand = alignment_loss(v, v2)
    assert 0.0 <= loss_rand.item() <= 2.0


def test_gradient_flows_into_both_projection_heads():
    model = _build().train()
    B, T, S, A, D = 1, 3, 2, 4, 16
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_emb = torch.randn(B, T, S, A, D)
    mask = torch.ones(B, T, S, A)

    pred, proj_har, proj_news = model(x_har, adj, x_emb, mask)
    loss = alignment_loss(proj_har, proj_news)
    loss.backward()

    assert model.align_har.weight.grad is not None, "align_har got no gradient"
    assert model.align_news.weight.grad is not None, "align_news got no gradient"
    # prediction path must NOT have been touched by an align-only backward
    assert model.fusion[0].weight.grad is None, \
        "alignment-only loss leaked gradient into the prediction fusion path"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
