import torch

import model as md


def test_forward_shape_and_graph_toggle():
    full = md.HARLSTMGAT(use_graph=True)
    nog = md.HARLSTMGAT(use_graph=False)
    x = torch.randn(4, 6, 10, 3)
    adj = torch.eye(6)
    assert full(x, adj).shape == (4, 6)
    assert nog(x, adj).shape == (4, 6)
    # the GAT branch adds parameters
    n_full = sum(p.numel() for p in full.parameters())
    n_nog = sum(p.numel() for p in nog.parameters())
    assert n_full > n_nog


def test_no_graph_ignores_adjacency():
    nog = md.HARLSTMGAT(use_graph=False).eval()
    x = torch.randn(2, 5, 10, 3)
    with torch.no_grad():
        a = nog(x, torch.eye(5))
        b = nog(x, torch.ones(5, 5))
    assert torch.allclose(a, b)   # adjacency has no effect when use_graph=False


def test_negative_partial_corr_edges_are_used():
    """A negative-weight edge is still an edge: use_graph must NOT drop it (MAJOR-1 fix: mask on !=0)."""
    full = md.HARLSTMGAT(use_graph=True).eval()
    x = torch.randn(1, 4, 10, 3)
    eye = torch.eye(4)
    neg = eye.clone(); neg[0, 1] = -0.7; neg[1, 0] = -0.7   # a negative partial-correlation edge
    with torch.no_grad():
        out_selfonly = full(x, eye)       # node 0 sees only itself
        out_with_neg = full(x, neg)       # node 0 now also attends to node 1 via the negative edge
    assert not torch.allclose(out_selfonly, out_with_neg)   # the negative edge changed node 0's output


def test_batched_adjacency_accepted():
    full = md.HARLSTMGAT(use_graph=True).eval()
    x = torch.randn(3, 4, 10, 3)
    adj = torch.eye(4).unsqueeze(0).expand(3, 4, 4)
    with torch.no_grad():
        out = full(x, adj)
    assert out.shape == (3, 4) and torch.isfinite(out).all()
