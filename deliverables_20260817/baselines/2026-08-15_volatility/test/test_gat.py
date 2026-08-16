# test/test_gat.py
import sys
from pathlib import Path
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from gat import GATLayer  # noqa: E402


def test_gat_shapes_and_identity_is_self_only():
    torch.manual_seed(0)
    B, N, din = 2, 4, 8
    layer = GATLayer(din, out_dim=5, heads=3)   # -> out width 15
    h = torch.randn(B, N, din)
    full = torch.ones(B, N, N)                  # fully connected
    out_full = layer(h, full)
    assert out_full.shape == (B, N, 15)
    # identity adjacency => each node attends only to itself
    eye = torch.eye(N).unsqueeze(0).expand(B, N, N)
    out_self = layer(h, eye)
    # perturbing OTHER nodes must not change a node's self-only output
    h2 = h.clone()
    h2[:, 1:] += 3.0
    out_self2 = layer(h2, eye)
    assert torch.allclose(out_self[:, 0], out_self2[:, 0], atol=1e-5)


def test_gat_directed_edge_only_affects_destination():
    torch.manual_seed(0)
    layer = GATLayer(4, out_dim=2, heads=1)
    h = torch.randn(1, 3, 4)
    adj = torch.zeros(1, 3, 3)
    adj[0, 1, 0] = 1.0          # directed edge: source 0 -> dest 1 only
    out = layer(h, adj)
    assert torch.isnan(out).any().item() is False
    assert torch.allclose(out[:, 0], torch.zeros(1, 2), atol=1e-6)   # node 0: isolated, no in-edge
    assert torch.allclose(out[:, 2], torch.zeros(1, 2), atol=1e-6)   # node 2: isolated, no in-edge
    h2 = h.clone()
    h2[:, 0] += 5.0
    out2 = layer(h2, adj)
    assert not torch.allclose(out[:, 1], out2[:, 1])   # perturbing source 0 moves dest 1
    assert torch.allclose(out[:, 2], out2[:, 2])        # unrelated isolated node unchanged
