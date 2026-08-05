"""Smoke test: RestTsBaseline forward + REST-TS loss composition (residual detach).

Run: pytest baselines/2026-07-18_resttext_baseline/test/test_smoke.py -v
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
from model_resttext import RestTsBaseline

pytestmark = pytest.mark.smoke


def _build(dropout=0.0, emb_dim=16, d_news=16):
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return RestTsBaseline(config, emb_dim=emb_dim, d_news=d_news, dropout=dropout)


def test_forward_shapes():
    model = _build().train()
    B, T, S, A, D = 2, 4, 3, 5, 16
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_emb = torch.randn(B, T, S, A, D)
    mask = torch.ones(B, T, S, A)

    har_pred, news_pred = model(x_har, adj, x_emb, mask)
    assert har_pred.shape == (B, S)
    assert news_pred.shape == (B, S)
    assert not torch.isnan(har_pred).any() and not torch.isnan(news_pred).any()


def test_both_heads_get_gradient():
    model = _build().train()
    B, T, S, A, D = 1, 3, 2, 4, 16
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_emb = torch.randn(B, T, S, A, D)
    mask = torch.ones(B, T, S, A)
    y = torch.randn(B, S)

    har_pred, news_pred = model(x_har, adj, x_emb, mask)
    residual_target = (y - har_pred).detach()
    loss = ((har_pred - y) ** 2).mean() + ((news_pred - residual_target) ** 2).mean()
    loss.backward()

    assert model.har_head[0].weight.grad is not None, "har_head got no gradient"
    assert model.news_head[0].weight.grad is not None, "news_head got no gradient"
    assert model.news_pool.query.grad is not None, "news pooling got no gradient"


def test_residual_detach_blocks_gradient_into_har_head():
    """Core REST-TS property: the residual (news) loss must NOT backprop into har_head —
    only loss_har (the primary task loss) should update har_head."""
    model = _build().train()
    B, T, S, A, D = 1, 3, 2, 4, 16
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_emb = torch.randn(B, T, S, A, D)
    mask = torch.ones(B, T, S, A)
    y = torch.randn(B, S)

    har_pred, news_pred = model(x_har, adj, x_emb, mask)
    residual_target = (y - har_pred).detach()   # [REST-TS] stop-gradient
    loss_news_only = ((news_pred - residual_target) ** 2).mean()
    loss_news_only.backward()

    assert model.har_head[0].weight.grad is None, \
        "residual loss leaked gradient into har_head — .detach() is not working as intended"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
