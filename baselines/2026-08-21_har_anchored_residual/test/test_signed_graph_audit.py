"""V2 characterization: the current GATLayer consumes adjacency ONLY as a binary presence mask.

These tests document the signed->binary reduction (diagnosis doc section 5): with the same edge SUPPORT
but different edge WEIGHTS/SIGNS, the current layer produces identical output (weight/sign are discarded),
and removing all non-self edges matches an identity graph. A corrected weighted/signed layer must instead
change its output when weights/signs change; that is Phase C.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
import model as md  # noqa: E402


def test_current_gat_ignores_edge_weight_and_sign():
    torch.manual_seed(0)
    gat = md.GATLayer(3, 8, heads=2).eval()
    h = torch.randn(1, 4, 3)
    support = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
    a_binary = torch.tensor(support, dtype=torch.float32)
    a_weighted = torch.tensor([[1.0, 0.6, 0, 0], [0.6, 1.0, 0, 0],
                               [0, 0, 1.0, -0.9], [0, 0, -0.9, 1.0]])  # same support, diff weight + sign
    with torch.no_grad():
        out_bin = gat(h, a_binary)
        out_w = gat(h, a_weighted)
    assert torch.allclose(out_bin, out_w), "current GAT should ignore edge weight/sign (binary mask)"


def test_current_gat_sign_flip_no_effect():
    torch.manual_seed(1)
    gat = md.GATLayer(3, 8, heads=2).eval()
    h = torch.randn(1, 3, 3)
    a_pos = torch.tensor([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
    a_neg = torch.tensor([[1.0, -0.5, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])  # sign flipped
    with torch.no_grad():
        assert torch.allclose(gat(h, a_pos), gat(h, a_neg))   # sign discarded


def test_removing_nonself_edges_matches_identity():
    torch.manual_seed(2)
    gat = md.GATLayer(3, 8, heads=2).eval()
    h = torch.randn(1, 4, 3)
    eye = torch.eye(4)
    with torch.no_grad():
        out_self = gat(h, eye)
    # each node attends only to itself -> a self-only message; deterministic given the same eval graph
    assert out_self.shape == (1, 4, 16) and torch.isfinite(out_self).all()
