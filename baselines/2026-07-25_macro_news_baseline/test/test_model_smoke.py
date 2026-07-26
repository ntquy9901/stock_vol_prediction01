"""Smoke test: DualGroupNewsBaseline (reused UNCHANGED from the sibling baseline, read-only
import) with the WIDER n_feat this baseline actually uses (146 dual + macro dims) — verifies the
"reuse instead of reinvent" design decision (design.md §5) actually works end to end.

Run: pytest baselines/2026-07-25_macro_news_baseline/test/test_model_smoke.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]
_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_dual_news import DualGroupNewsBaseline  # noqa: E402
from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402

pytestmark = pytest.mark.smoke

N_FEAT = 146 + 66  # dual-group (146) + macro (32 + 32 ewma + 1 norm + 1 ewma_norm = 66)


def _build(n_feat=N_FEAT, dropout=0.0):
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return DualGroupNewsBaseline(config, n_feat=n_feat, dropout=dropout)


def test_forward_shape_and_backward_with_wide_macro_augmented_nfeat():
    model = _build().train()
    B, T, S, F = 2, 22, 3, N_FEAT
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_news = torch.randn(B, T, S, F)

    pred = model(x_har, adj, x_news)
    assert pred.shape == (B, S)
    assert not torch.isnan(pred).any()

    pred.sum().backward()
    assert model.news_branch.proj.weight.grad is not None
    assert model.news_branch.proj.weight.shape == (64, N_FEAT)
    assert model.fusion[0].weight.grad is not None


def test_all_zero_macro_and_dual_news_does_not_break_forward():
    """Both panels missing (smoke-mode dataset fallback) -> all-zero x_news must not crash or
    NaN the model."""
    model = _build().eval()
    B, T, S, F = 1, 22, 2, N_FEAT
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_news = torch.zeros(B, T, S, F)
    pred = model(x_har, adj, x_news)
    assert not torch.isnan(pred).any()
