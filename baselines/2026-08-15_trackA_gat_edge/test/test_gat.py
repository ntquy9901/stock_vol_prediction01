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
