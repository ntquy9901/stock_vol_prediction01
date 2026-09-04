"""Unit tests for the PatchTST encoder + PatchTSTRichNet (CPU-only; see conftest).

Covers: (a) encoder shape [B,N,seq,5] -> [B,N,hidden]; (b) patch-count math
num_patches = floor((seq-patch_len)/stride)+1; (c) full-net forward -> [B,N] for both variants;
plus config validation and the recency-preservation property of the default patch/stride at seq=22.
"""
import pytest
import torch

from patchtst_config import PatchTSTHParams, num_patches
from patchtst_net import PatchTSTEncoder, PatchTSTRichNet


@pytest.mark.smoke
def test_encoder_shape_maps_to_hidden():
    """(a) [B,N,22,5] -> [B,N,64]."""
    torch.manual_seed(0)
    enc = PatchTSTEncoder(seq_len=22, n_feat=5, out_dim=64, hp=PatchTSTHParams(), dropout=0.0)
    x = torch.randn(3, 8, 22, 5)
    out = enc(x)
    assert out.shape == (3, 8, 64)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("seq,patch_len,stride", [(22, 6, 4), (22, 8, 4), (10, 6, 4), (30, 8, 8)])
def test_patch_count_matches_formula(seq, patch_len, stride):
    """(b) patch count == floor((seq-patch_len)/stride)+1, verified against the encoder's own patching."""
    expected = (seq - patch_len) // stride + 1
    assert num_patches(seq, patch_len, stride) == expected
    hp = PatchTSTHParams(patch_len=patch_len, stride=stride)
    enc = PatchTSTEncoder(seq_len=seq, n_feat=5, out_dim=16, hp=hp, dropout=0.0)
    assert enc.num_patches == expected
    # confirm the actual unfold produces exactly `expected` patch tokens
    z = torch.randn(2 * 4 * 5, seq)
    assert z.unfold(-1, patch_len, stride).shape[1] == expected


def test_default_patch_stride_preserve_last_day_at_lookback22():
    """Property: defaults (patch_len=6, stride=4) tile lookback=22 so the last patch covers index 21
    (the most recent day) -> no trailing days dropped (design §2 recency preservation)."""
    hp = PatchTSTHParams()
    p = num_patches(22, hp.patch_len, hp.stride)
    last_start = (p - 1) * hp.stride
    last_covered_end = last_start + hp.patch_len - 1        # inclusive last index inside the last patch
    assert last_covered_end == 21


def test_forward_both_variants_return_BN():
    """(c) full net -> [B,N] predictions for no-graph and graph variants."""
    torch.manual_seed(0)
    B, N, seq = 4, 6, 22
    x = torch.randn(B, N, seq, 5)
    adj = torch.eye(N).unsqueeze(0).repeat(B, 1, 1)        # self-loop only; valid adjacency
    for use_graph in (False, True):
        net = PatchTSTRichNet(seq_len=seq, hidden=64, heads=4, dropout=0.0, use_graph=use_graph)
        out = net(x, adj)
        assert out.shape == (B, N)
        assert torch.isfinite(out).all()


def test_graph_variant_depends_on_adjacency():
    """The graph branch must actually consume the adjacency (sign/weight change -> output change),
    otherwise the leave-one-out graph contrast would be meaningless."""
    torch.manual_seed(1)
    B, N, seq = 2, 5, 22
    x = torch.randn(B, N, seq, 5)
    net = PatchTSTRichNet(seq_len=seq, hidden=32, heads=2, dropout=0.0, use_graph=True)
    net.eval()
    a1 = torch.eye(N).unsqueeze(0).repeat(B, 1, 1)
    a2 = a1.clone()
    a2[:, 0, 1] = 0.9                                      # add a real (source 1 -> target 0) edge
    with torch.no_grad():
        assert not torch.allclose(net(x, a1), net(x, a2))


def test_config_rejects_bad_hparams():
    with pytest.raises(ValueError):
        PatchTSTHParams(d_model=65, n_heads=4)             # not divisible
    with pytest.raises(ValueError):
        PatchTSTHParams(pool="max")
    with pytest.raises(ValueError):
        PatchTSTEncoder(seq_len=4, n_feat=5, out_dim=8, hp=PatchTSTHParams(patch_len=6))  # seq<patch_len


def test_encoder_channel_count_guard():
    enc = PatchTSTEncoder(seq_len=22, n_feat=5, out_dim=8, hp=PatchTSTHParams(), dropout=0.0)
    with pytest.raises(ValueError):
        enc(torch.randn(1, 2, 22, 3))                      # wrong channel count
