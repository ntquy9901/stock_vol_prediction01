"""Graph WaveNet component tests: the self-adaptive adjacency is recomputed INDEPENDENTLY from the raw
node-embedding parameters (softmax(relu(E1 @ E2)), NOT reusing the module forward) and matched to the
module -- the CLAUDE.md named-formula rule for arXiv:1906.00121 (official
``adp = softmax(relu(mm(nodevec1, nodevec2)), dim=1)``). Plus nconv/gcn propagation + forward shapes.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

from gwn_model import GCN, GraphWaveNet, Linear1x1, NConv  # noqa: E402


def test_adaptive_adjacency_matches_paper_formula_independent_recompute():
    """A_adp = SoftMax(ReLU(E1 . E2^T)), softmax over dim=1 -- recomputed with numpy from the raw params."""
    torch.manual_seed(0)
    net = GraphWaveNet(n_nodes=7, adaptive=True, node_dim=6)
    e1 = net.nodevec1.detach().numpy()          # [N, c]  (E1)
    e2 = net.nodevec2.detach().numpy()          # [c, N]  (E2^T)
    raw = np.maximum(e1 @ e2, 0.0)              # ReLU(E1 . E2^T)
    ex = np.exp(raw - raw.max(axis=1, keepdims=True))
    expected = ex / ex.sum(axis=1, keepdims=True)   # softmax over dim=1 (per source row)
    got = net.adaptive_adjacency().detach().numpy()
    assert got.shape == (7, 7)
    assert np.allclose(got, expected, atol=1e-5)
    assert np.allclose(got.sum(axis=1), 1.0, atol=1e-5)   # each source row sums to 1


def test_adaptive_adjacency_raises_when_not_adaptive():
    net = GraphWaveNet(n_nodes=5, adaptive=False)
    with pytest.raises(RuntimeError):
        net.adaptive_adjacency()


def test_nconv_fixed_adjacency_propagation():
    """out[..., w, :] = sum_v x[..., v, :] * A[v, w] -- verified against an independent matmul."""
    torch.manual_seed(1)
    x = torch.randn(2, 3, 4, 5)                 # [B, C, N, T]
    a = torch.randn(4, 4)
    got = NConv()(x, a)
    exp = torch.matmul(x.permute(0, 1, 3, 2), a).permute(0, 1, 3, 2)   # sum over node dim, independent
    assert got.shape == x.shape
    assert torch.allclose(got, exp, atol=1e-5)


def test_nconv_batched_adjacency_propagation():
    torch.manual_seed(2)
    x = torch.randn(2, 3, 4, 5)
    a = torch.randn(2, 4, 4)                    # [B, N, N]
    got = NConv()(x, a)
    exp = torch.matmul(x.permute(0, 1, 3, 2), a.unsqueeze(1)).permute(0, 1, 3, 2)
    assert torch.allclose(got, exp, atol=1e-5)


def test_gcn_and_linear_shapes():
    torch.manual_seed(3)
    x = torch.randn(2, 8, 4, 5)
    a = torch.rand(2, 4, 4)
    gcn = GCN(c_in=8, c_out=16, dropout=0.0, support_len=1, order=2)
    out = gcn(x, [a])
    assert out.shape == (2, 16, 4, 5)
    lin = Linear1x1(8, 3)
    assert lin(x).shape == (2, 3, 4, 5)


def test_masked_adp_zeros_invalid_source_rows():
    net = GraphWaveNet(n_nodes=4, adaptive=True, node_dim=6)
    nmask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])         # node 1 invalid
    a_b = net._masked_adp(nmask)
    assert a_b.shape == (1, 4, 4)
    assert torch.allclose(a_b[0, 1, :], torch.zeros(4))  # invalid SOURCE row zeroed
    assert torch.all(a_b[0, 0, :] > 0)                   # valid source untouched (softmax > 0)


def test_receptive_field_default_is_13():
    net = GraphWaveNet(n_nodes=5, blocks=4, layers=2, kernel_size=2)
    assert net.receptive_field == 13                     # 1 + 4*(1+2)


@pytest.mark.parametrize("adaptive", [True, False])
def test_forward_shape_and_finite_padded(adaptive):
    """seq(10) < receptive_field(13) -> input is left-padded; both variants return finite [B, N]."""
    torch.manual_seed(4)
    net = GraphWaveNet(n_nodes=6, adaptive=adaptive).eval()
    x = torch.randn(3, 6, 10, 5)
    nmask = torch.ones(3, 6)
    out = net(x, nmask)
    assert out.shape == (3, 6)
    assert torch.isfinite(out).all()


def test_forward_no_padding_branch_when_seq_ge_receptive_field():
    """blocks=1, layers=1 -> receptive_field=2; seq=5 >= 2 exercises the no-pad path."""
    net = GraphWaveNet(n_nodes=4, blocks=1, layers=1, adaptive=True).eval()
    assert net.receptive_field == 2
    out = net(torch.randn(2, 4, 5, 5), torch.ones(2, 4))
    assert out.shape == (2, 4) and torch.isfinite(out).all()


def test_forward_finite_with_partially_invalid_nmask():
    """A snapshot with some invalid (nmask=0) nodes stays finite; masking invalid sources does not blow up."""
    torch.manual_seed(5)
    net = GraphWaveNet(n_nodes=6, adaptive=True).eval()
    nmask = torch.ones(2, 6)
    nmask[0, 1] = 0.0; nmask[0, 4] = 0.0; nmask[1, 0] = 0.0   # partially invalid nodes per sample
    x = torch.randn(2, 6, 10, 5)
    x[0, 1] = 0.0; x[0, 4] = 0.0; x[1, 0] = 0.0               # zero-filled like the masked panel
    out = net(x, nmask)
    assert out.shape == (2, 6) and torch.isfinite(out).all()


def test_adaptive_and_no_adaptive_differ_and_param_presence():
    """The adaptive variant owns node embeddings + graph-conv layers and yields a DIFFERENT output; the
    no-adaptive variant has neither (pure TCN)."""
    torch.manual_seed(6)
    x = torch.randn(2, 5, 10, 5); nmask = torch.ones(2, 5)
    a = GraphWaveNet(n_nodes=5, adaptive=True).eval()
    n = GraphWaveNet(n_nodes=5, adaptive=False).eval()
    assert hasattr(a, "nodevec1") and len(a.gconv) > 0
    assert not hasattr(n, "nodevec1") and len(n.gconv) == 0
    assert not torch.allclose(a(x, nmask), n(x, nmask))       # the adaptive graph changes the prediction


def test_forward_differentiable_finite_gradients():
    net = GraphWaveNet(n_nodes=5, adaptive=True)
    out = net(torch.randn(2, 5, 10, 5), torch.ones(2, 5))
    out.sum().backward()
    g1 = net.nodevec1.grad
    assert g1 is not None and torch.isfinite(g1).all()
