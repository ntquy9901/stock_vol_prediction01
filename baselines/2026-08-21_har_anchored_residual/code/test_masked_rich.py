"""Smoke + property tests for the rich masked-panel graph experiment (pytest).

Covers the two new pieces:
  * ``WeightedGATLayer`` MUST consume edge weight AND sign (the whole point vs the binary GATLayer),
    and MUST stay mask-aware (a non-neighbour node cannot influence a target's output).
  * ``build_masked_rich`` produces a 5-feature masked panel + two square, self-looped adjacencies on a
    small real VN30 data slice (real-data smoke, per project rule).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "submission" / "soict_lstm_gat"))

import masked_rich as MR  # noqa: E402
from run_masked_rich import WeightedGATLayer, MaskedRichNet, train_masked_rich  # noqa: E402
from config import SMOKE  # noqa: E402

_VN30 = sorted(glob.glob(str(REPO / "submission" / "soict_lstm_gat" / "data" / "vn30" / "*_processed.csv")))
_PRICE = REPO / "data" / "raw" / "prices"


def _layer(seed=0):
    torch.manual_seed(seed)
    return WeightedGATLayer(in_dim=5, out_dim=8, heads=2).eval()


def test_wgat_consumes_edge_sign():
    """Flipping an edge's SIGN must change the output (binary GATLayer would not)."""
    lay = _layer()
    torch.manual_seed(1)
    h = torch.randn(1, 4, 5)
    A = torch.zeros(1, 4, 4)
    for i in range(4):
        A[0, i, i] = 1.0
    A[0, 0, 1] = 0.7                       # edge source 1 -> target 0
    out_pos = lay(h, A)
    A_neg = A.clone(); A_neg[0, 0, 1] = -0.7
    out_neg = lay(h, A_neg)
    assert not torch.allclose(out_pos[0, 0], out_neg[0, 0], atol=1e-6), "sign flip did not change output"


def test_wgat_consumes_edge_weight():
    """Changing an edge's WEIGHT magnitude must change the output."""
    lay = _layer()
    torch.manual_seed(2)
    h = torch.randn(1, 4, 5)
    A = torch.eye(4).unsqueeze(0).clone()
    A[0, 0, 1] = 0.2
    out_a = lay(h, A)
    A2 = A.clone(); A2[0, 0, 1] = 0.9
    out_b = lay(h, A2)
    assert not torch.allclose(out_a[0, 0], out_b[0, 0], atol=1e-6), "weight change did not change output"


def test_wgat_mask_aware():
    """A node that is NOT a neighbour of target 0 cannot influence target 0's output."""
    lay = _layer()
    torch.manual_seed(3)
    h = torch.randn(1, 4, 5)
    A = torch.eye(4).unsqueeze(0).clone()
    A[0, 0, 1] = 0.5                        # target 0's only non-self neighbour is node 1
    out1 = lay(h, A)
    h2 = h.clone(); h2[0, 3] = torch.randn(5)   # perturb node 3 (a non-neighbour of target 0)
    out2 = lay(h2, A)
    assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6), "non-neighbour leaked into target output"


@pytest.mark.skipif(len(_VN30) < 5, reason="VN30 processed data not available")
def test_build_masked_rich_shapes():
    D = MR.build_masked_rich(_VN30, _PRICE, lookback=10, horizon=5)
    assert D.X_tr.shape[-1] == MR.N_FEAT == 5
    assert D.X_tr.shape[1] == D.N and D.adj_vol2pk.shape == (D.N, D.N)
    assert D.adj_corr.shape == (D.N, D.N)
    assert np.allclose(np.diag(D.adj_vol2pk), 1.0)         # self-loop
    assert np.allclose(np.diag(D.adj_corr), 1.0)
    assert D.har_tr.shape[-1] == 3                          # HAR baseline still 3-feat
    # masks: a scored target cell must also have a valid input window
    assert (D.tmask_te <= D.nmask_te).all()
    # vol->PK is DIRECTED (asymmetric) whereas corr edge is symmetric in support
    assert not np.allclose(D.adj_vol2pk, D.adj_vol2pk.T)
    # features finite after nan_to_num
    assert np.isfinite(D.X_tr).all() and np.isfinite(D.y_te).all()


@pytest.mark.skipif(len(_VN30) < 5, reason="VN30 processed data not available")
def test_train_masked_rich_smoke():
    """End-to-end: build + train (2 epochs, 1 seed) both no-graph and weighted-GAT paths run + return
    finite floored predictions of the right shape."""
    D = MR.build_masked_rich(_VN30, _PRICE, lookback=10, horizon=5)
    p_lstm = train_masked_rich(D, SMOKE, seed=42, use_graph=False, adj=D.adj_vol2pk)
    p_gat = train_masked_rich(D, SMOKE, seed=42, use_graph=True, adj=D.adj_vol2pk)
    assert p_lstm.shape == D.y_te.shape == p_gat.shape
    assert np.isfinite(p_gat).all() and (p_gat > 0).all()


def test_masked_rich_net_forward():
    net = MaskedRichNet(hidden=8, heads=2, use_graph=True).eval()
    assert net.gat_layers == 2 and hasattr(net, "gat2")   # 2-hop default (matches deliverable)
    x = torch.randn(3, 6, 10, 5)
    adj = torch.eye(6).unsqueeze(0).expand(3, 6, 6).contiguous()
    with torch.no_grad():
        out = net(x, adj)
    assert out.shape == (3, 6)


def test_two_hop_net_consumes_edge_sign_and_weight():
    """Through BOTH hops, flipping an edge's sign OR changing its weight must change the net output."""
    torch.manual_seed(7)
    net = MaskedRichNet(hidden=8, heads=2, use_graph=True).eval()
    x = torch.randn(1, 5, 10, 5)
    A = torch.eye(5).unsqueeze(0).clone()
    A[0, 0, 1] = 0.6                                       # source 1 -> target 0
    with torch.no_grad():
        base = net(x, A)
        A_neg = A.clone(); A_neg[0, 0, 1] = -0.6
        out_neg = net(x, A_neg)
        A_w = A.clone(); A_w[0, 0, 1] = 0.1
        out_w = net(x, A_w)
    assert not torch.allclose(base[0, 0], out_neg[0, 0], atol=1e-6), "2-hop: sign flip ignored"
    assert not torch.allclose(base[0, 0], out_w[0, 0], atol=1e-6), "2-hop: weight change ignored"


def test_two_hop_mask_aware():
    """2-hop must stay mask-aware: with only self-loops, perturbing another node cannot change target
    0's output (no path reaches it through either hop)."""
    torch.manual_seed(8)
    net = MaskedRichNet(hidden=8, heads=2, use_graph=True).eval()
    x = torch.randn(1, 5, 10, 5)
    A = torch.eye(5).unsqueeze(0).clone()                 # self-loops only -> disconnected nodes
    with torch.no_grad():
        out1 = net(x, A)
        x2 = x.clone(); x2[0, 3] = torch.randn(10, 5)     # perturb an unconnected node
        out2 = net(x2, A)
    assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6), "2-hop leaked a non-neighbour into target"


def test_har5_exposed_and_harx_ols_runs():
    """HAR-X fair baseline: build exposes the raw 5-feature vector at t, its first 3 cols are the HAR
    features, and a 5-feature linear OLS fits + yields finite predictions (isolates the extra-feature
    contribution so a deep/graph win over HAR is not just the 2 extra node features)."""
    D = MR.build_masked_rich(_VN30, _PRICE, lookback=10, horizon=5)
    assert D.har5_tr.shape[-1] == 5 and D.har5_te.shape[-1] == 5
    assert np.isfinite(D.har5_tr).all() and np.isfinite(D.har5_te).all()
    assert np.allclose(D.har5_tr[..., :3], D.har_tr)          # first 3 = daily/weekly/monthly HAR
    m = D.tmask_tr.astype(bool)
    Xb = np.column_stack([np.ones(int(m.sum())), D.har5_tr[m]])
    coef = np.linalg.lstsq(Xb, D.y_tr[m], rcond=None)[0]
    pred = np.column_stack([np.ones(len(D.har5_te.reshape(-1, 5))), D.har5_te.reshape(-1, 5)]) @ coef
    assert np.isfinite(pred).all()
