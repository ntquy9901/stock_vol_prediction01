"""Smoke test: GatedCrossAttnBaseline forward + gate/backward properties.

Run: pytest baselines/2026-07-18_gated_crossattn_baseline/test/test_smoke.py -v
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
from model_gated_crossattn import GatedCrossAttnBaseline

pytestmark = pytest.mark.smoke


def _build(dropout=0.0, emb_dim=16, d_news=16, num_heads=2):
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    return GatedCrossAttnBaseline(config, emb_dim=emb_dim, d_news=d_news, num_heads=num_heads,
                                  dropout=dropout)


def test_forward_shape_and_backward():
    model = _build().train()
    B, T, S, A, D = 2, 4, 3, 5, 16
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_emb = torch.randn(B, T, S, A, D)
    mask = torch.ones(B, T, S, A)

    pred = model(x_har, adj, x_emb, mask)
    assert pred.shape == (B, S)
    assert not torch.isnan(pred).any()

    pred.sum().backward()
    assert model.cross_attn.out_proj.weight.grad is not None, "no grad on cross-attention"
    assert model.gate_mlp[0].weight.grad is not None, "no grad on gate"
    assert model.fusion[0].weight.grad is not None, "no grad on fusion"


def test_zero_news_day_does_not_break_forward():
    model = _build().eval()
    B, T, S, A, D = 1, 3, 2, 4, 16
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_emb = torch.randn(B, T, S, A, D)
    mask = torch.zeros(B, T, S, A)   # NO news anywhere
    pred = model(x_har, adj, x_emb, mask)
    assert not torch.isnan(pred).any(), "NaN when all news masked"


def test_attended_output_depends_on_query():
    """[Regression test, code review 2026-07-18] `attended` must actually depend on har_embed
    (the query). An earlier buggy version collapsed news to 1 K/V token, making softmax always
    1.0 regardless of the query — this test fails on that version and passes on the fix
    (attending over the un-collapsed 22-day sequence, seq_len>1)."""
    model = _build().eval()
    B, T, S, A, D = 1, 5, 2, 4, 16
    adj = torch.rand(B, S, S)
    x_emb = torch.randn(B, T, S, A, D)
    mask = torch.ones(B, T, S, A)

    x_har_a = torch.randn(B, T, S, 3)
    x_har_b = torch.randn(B, T, S, 3) * 5 + 3   # very different HAR input -> different query

    pred_a = model(x_har_a, adj, x_emb, mask)
    pred_b = model(x_har_b, adj, x_emb, mask)
    # if attended were query-independent (the bug), only har_embed's own direct contribution to
    # `fused` would differ; with a real cross-attention, changing the query should also change
    # gate*attended, giving a LARGER divergence than HAR-embedding-only would explain — check the
    # attention module's output directly instead of relying on the full forward's confound.
    captured = {}

    def _hook(module, inp, out):
        captured.setdefault("outs", []).append(out[0].detach().clone())

    handle = model.cross_attn.register_forward_hook(_hook)
    try:
        model(x_har_a, adj, x_emb, mask)
        model(x_har_b, adj, x_emb, mask)
    finally:
        handle.remove()

    attended_a, attended_b = captured["outs"]
    assert not torch.allclose(attended_a, attended_b, atol=1e-6), \
        "cross-attention output is identical for two very different queries — degenerate softmax bug"


def test_gate_is_in_unit_interval():
    """The learned gate must stay in (0,1) — sigmoid output, checked via a forward hook."""
    model = _build().eval()
    B, T, S, A, D = 1, 3, 2, 4, 16
    x_har = torch.randn(B, T, S, 3)
    adj = torch.rand(B, S, S)
    x_emb = torch.randn(B, T, S, A, D) * 10  # exaggerate to stress-test range
    mask = torch.ones(B, T, S, A)

    captured = {}

    def _hook(module, inp, out):
        captured["gate"] = out.detach().clone()

    handle = model.gate_mlp.register_forward_hook(_hook)
    try:
        model(x_har, adj, x_emb, mask)
    finally:
        handle.remove()

    gate = captured["gate"]
    assert (gate >= 0.0).all() and (gate <= 1.0).all(), \
        f"gate out of [0,1] range: min={gate.min().item()}, max={gate.max().item()}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed.")
