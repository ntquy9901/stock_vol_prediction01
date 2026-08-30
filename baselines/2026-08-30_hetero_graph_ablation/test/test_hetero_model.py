"""Architecture tests for HeteroRichNet (hetero_model.py) — no training, CPU only.

Independently verifies (per the task): (b) the two relations have DIFFERENT learned parameters (independent
conv weights) — distinct Parameter objects AND grads that diverge when the two adjacencies differ; plus SUM
aggregation dim, that the nonlinear relation is actually consumed, and finite forward. UNIQUE basename
(test_hetero_model.py) to avoid the pytest prepend-import duplicate-basename shadowing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import hetero_model as HM  # noqa: E402

N, SEQ, F, HID, HEADS = 5, 4, HM.MR.N_FEAT, 8, 2


def _adj(n, seed):
    """Symmetric [n,n] adjacency with self-loop 1 and a few random weighted edges."""
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                a[i, j] = a[j, i] = float(rng.uniform(0.2, 1.0))
    np.fill_diagonal(a, 1.0)
    return a


def _inputs(b=3, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.normal(size=(b, N, SEQ, F)).astype(np.float32))
    al = torch.from_numpy(_adj(N, 1)).unsqueeze(0).repeat(b, 1, 1)
    an = torch.from_numpy(_adj(N, 2)).unsqueeze(0).repeat(b, 1, 1)
    return x, al, an


def test_forward_shape_and_finite():
    net = HM.HeteroRichNet(HID, HEADS, 0.0).eval()
    x, al, an = _inputs()
    out = net(x, al, an)
    assert out.shape == (x.shape[0], N)
    assert torch.isfinite(out).all()


def test_sum_aggregation_head_input_dim():
    """SUM aggregation -> head input = hidden + hidden*heads (same as the single-relation MaskedRichNet)."""
    net = HM.HeteroRichNet(HID, HEADS, 0.0)
    assert net.head[0].in_features == HID + HID * HEADS


def test_relations_have_distinct_parameter_objects():
    net = HM.HeteroRichNet(HID, HEADS, 0.0)
    # (b) INDEPENDENT conv weights: distinct Parameter objects (not a shared/tied weight)
    assert net.gat_lin1.W.weight is not net.gat_nl1.W.weight
    assert net.gat_lin2.W.weight is not net.gat_nl2.W.weight
    assert net.gat_lin1.a_src is not net.gat_nl1.a_src


def test_relations_grads_diverge_when_adjacencies_differ():
    """(b) With the two branches initialised IDENTICALLY but fed DIFFERENT adjacencies, their gradients differ
    -> the relations are independent params receiving relation-specific signal (a tied weight would share one
    summed grad)."""
    net = HM.HeteroRichNet(HID, HEADS, 0.0)
    with torch.no_grad():                                  # copy linear-branch params into the nonlinear branch
        net.gat_nl1.W.weight.copy_(net.gat_lin1.W.weight)
        net.gat_nl1.a_src.copy_(net.gat_lin1.a_src)
        net.gat_nl1.a_dst.copy_(net.gat_lin1.a_dst)
        net.gat_nl1.edge_bias.copy_(net.gat_lin1.edge_bias)
    x, al, an = _inputs(seed=7)
    assert not torch.allclose(al, an)                      # the two relations really differ
    out = net(x, al, an)
    out.pow(2).mean().backward()
    gl = net.gat_lin1.W.weight.grad
    gn = net.gat_nl1.W.weight.grad
    assert gl is not None and gn is not None
    assert not torch.allclose(gl, gn)                      # divergent grads -> independent, relation-specific


def test_nonlinear_relation_is_consumed():
    """Changing ONLY adj_nl changes the output -> the nonlinear relation actually propagates (not ignored)."""
    net = HM.HeteroRichNet(HID, HEADS, 0.0).eval()
    x, al, an = _inputs(seed=3)
    an2 = torch.from_numpy(_adj(N, 99)).unsqueeze(0).repeat(x.shape[0], 1, 1)
    with torch.no_grad():
        o1 = net(x, al, an)
        o2 = net(x, al, an2)
    assert not torch.allclose(o1, o2)


def test_one_hop_variant_shapes():
    net = HM.HeteroRichNet(HID, HEADS, 0.0, gat_layers=1).eval()
    x, al, an = _inputs()
    assert net(x, al, an).shape == (x.shape[0], N)
