"""TDD for the MTGNN graph-learning layer + learned-adjacency wrapper.

Asserts the paper properties: output [N,N]; directed/asymmetric; top-k sparse (<= k outgoing edges/node);
self-loop on the wrapper's adjacency; differentiable with finite gradients; and equivalence of the module
matmuls to an independent recompute of Eqs. (1)-(3).
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

from mtgnn_graph import GraphConstructor, LearnedGraphNet  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def test_output_shape_is_NxN():
    gc = GraphConstructor(n_nodes=12, subgraph_size=4, node_dim=8)
    a = gc(torch.arange(12))
    assert a.shape == (12, 12)


def test_topk_sparsity_at_most_k_outgoing_edges_per_node():
    n, k = 30, 5
    gc = GraphConstructor(n, subgraph_size=k, node_dim=16)
    a = gc(torch.arange(n)).detach().numpy()
    per_row_nonzero = (a > 0).sum(axis=1)
    assert per_row_nonzero.max() <= k, per_row_nonzero.max()


def test_k_capped_at_n_nodes():
    gc = GraphConstructor(n_nodes=6, subgraph_size=50, node_dim=8)
    assert gc.k == 6
    a = gc(torch.arange(6))
    assert a.shape == (6, 6) and torch.isfinite(a).all()


def test_directed_asymmetric():
    gc = GraphConstructor(n_nodes=20, subgraph_size=6, node_dim=16)
    a = gc(torch.arange(20)).detach().numpy()
    assert not np.allclose(a, a.T), "MTGNN adjacency must be directed/asymmetric"


def test_relu_nonnegative_and_finite():
    gc = GraphConstructor(n_nodes=15, subgraph_size=4, node_dim=8)
    a = gc(torch.arange(15)).detach().numpy()
    assert (a >= 0).all() and np.isfinite(a).all()


def test_differentiable_finite_gradients():
    gc = GraphConstructor(n_nodes=10, subgraph_size=3, node_dim=8)
    a = gc(torch.arange(10))
    a.sum().backward()
    g = gc.emb1.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_matches_paper_equations_independent_recompute():
    """Recompute Eqs.(1)-(3) from the layer's own weights (no top-k) and match the pre-sparsified adj."""
    n, d, alpha = 8, 8, 3.0
    gc = GraphConstructor(n, subgraph_size=n, node_dim=d, alpha=alpha)  # k=n -> no sparsification effect on values
    idx = torch.arange(n)
    with torch.no_grad():
        e1, e2 = gc.emb1(idx), gc.emb2(idx)
        m1 = torch.tanh(alpha * gc.theta1(e1))            # Eq.(1)
        m2 = torch.tanh(alpha * gc.theta2(e2))            # Eq.(2)
        a_full = torch.relu(torch.tanh(alpha * (m1 @ m2.T - m2 @ m1.T)))  # Eq.(3)
        # with k=n every column is selectable; the only difference is the rand tie-break can drop exact-0
        # columns, which are 0 anyway -> the NONZERO structure and values must equal the reference.
        got = gc(idx)
    nz = a_full > 0
    assert torch.allclose(got[nz], a_full[nz], atol=1e-6)


def test_alpha_default_is_three():
    assert GraphConstructor(4, 2, 4).alpha == 3.0


def test_invalid_subgraph_size_raises():
    with pytest.raises(ValueError):
        GraphConstructor(5, subgraph_size=0, node_dim=4)


def test_wrapper_learned_adjacency_has_unit_self_loop():
    net = LearnedGraphNet(n_nodes=14, hidden=8, heads=2, subgraph_size=4, node_dim=8)
    a = net.learned_adjacency().detach().numpy()
    assert a.shape == (14, 14)
    assert np.allclose(np.diag(a), 1.0), "self-loop diagonal must be 1.0"


def test_wrapper_forward_shape_and_finite():
    torch.manual_seed(1)
    n, b, seq = 9, 3, 10
    net = LearnedGraphNet(n_nodes=n, hidden=8, heads=2, subgraph_size=3, node_dim=8).eval()
    x = torch.randn(b, n, seq, 5)
    nmask = torch.ones(b, n)
    with torch.no_grad():
        out = net(x, nmask)
    assert out.shape == (b, n) and torch.isfinite(out).all()


def test_wrapper_node_mask_zeros_invalid_source_columns():
    """build_adj must zero every column for an invalid source node (matches the training-loop masking)."""
    n = 8
    net = LearnedGraphNet(n_nodes=n, hidden=8, heads=2, subgraph_size=3, node_dim=8)
    nmask = torch.ones(1, n); nmask[0, 3] = 0.0            # node 3 invalid as a source
    adj_b = net.build_adj(nmask).detach().numpy()[0]
    assert np.allclose(adj_b[:, 3], 0.0), "invalid source column must be fully zeroed"
