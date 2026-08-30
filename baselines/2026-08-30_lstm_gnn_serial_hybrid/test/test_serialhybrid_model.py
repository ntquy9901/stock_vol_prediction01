"""SERIAL LSTM->GNN model + trainer coverage. UNIQUE basename (test_serialhybrid_model.py) to avoid the
pytest prepend-import duplicate-basename collision that has cost gate cycles.

Pins the SERIAL contract: the GAT's input tensor IS the LSTM embedding (not the raw features), and SEQ
lookback changes that input (impossible in the delivered PARALLEL model). Also exercises the full
``train_serial`` loop on tiny CPU data (real net + a constant-net stub that forces the early-stop arcs).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import serial_hybrid_net as SH  # noqa: E402
from config import Config       # noqa: E402


def _adj_b(batch, n, nmask=None):
    base = torch.eye(n).unsqueeze(0).repeat(batch, 1, 1)
    if nmask is None:
        nmask = torch.ones(batch, n)
    return base * nmask.unsqueeze(1)


# ------------------------- device resolver (both arcs) -------------------------

def test_resolve_device_both_arcs(monkeypatch):
    monkeypatch.setattr(SH.torch.cuda, "is_available", lambda: True)
    assert SH.resolve_device().type == "cuda"
    monkeypatch.setattr(SH.torch.cuda, "is_available", lambda: False)
    assert SH.resolve_device().type == "cpu"


# ------------------------- model forward (graph + no-graph) -------------------------

def test_forward_shapes_graph_and_nograph():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 6, 5)
    adj = _adj_b(3, 5)
    for use_graph in (True, False):
        net = SH.SerialLSTMGNN(hidden=8, heads=2, dropout=0.0, use_graph=use_graph, in_dim=5).eval()
        with torch.no_grad():
            out = net(x, adj)
        assert out.shape == (3, 5)
        assert torch.isfinite(out).all()


def test_gnn_input_is_the_lstm_embedding():
    """SERIAL contract: the GAT receives h = LSTM embedding, NOT the raw features."""
    torch.manual_seed(1)
    net = SH.SerialLSTMGNN(hidden=8, heads=2, dropout=0.0, use_graph=True, in_dim=5).eval()
    x = torch.randn(2, 4, 6, 5)
    adj = _adj_b(2, 4)
    captured = {}
    net.gat.register_forward_hook(lambda m, inp, out: captured.__setitem__("in", inp[0]))
    with torch.no_grad():
        net(x, adj)
    # the tensor fed to the GAT equals the model's LSTM embedding, and does NOT equal the raw day-t features
    assert torch.allclose(captured["in"], net.lstm_embed(x), atol=1e-6)
    assert captured["in"].shape == (2, 4, 8)                 # [B,N,hidden], not [B,N,5] raw day-t features


def test_seq_history_changes_the_gnn_input():
    """Changing an EARLIER timestep changes the LSTM embedding (the GAT input) -- the property the PARALLEL
    model lacks (its GAT sees only day t = index -1)."""
    torch.manual_seed(2)
    net = SH.SerialLSTMGNN(hidden=8, heads=2, dropout=0.0, use_graph=True, in_dim=5).eval()
    x = torch.randn(2, 4, 6, 5)
    x_early = x.clone()
    x_early[:, :, 0, :] += 5.0                               # perturb only the earliest timestep
    with torch.no_grad():
        assert not torch.allclose(net.lstm_embed(x), net.lstm_embed(x_early), atol=1e-5)


def test_mask_awareness_isolated_node_finite():
    """A node whose column is zeroed by the node mask (invalid neighbour) still yields finite output."""
    torch.manual_seed(3)
    net = SH.SerialLSTMGNN(hidden=8, heads=2, dropout=0.0, use_graph=True, in_dim=5).eval()
    x = torch.randn(2, 4, 6, 5)
    nm = torch.ones(2, 4)
    nm[:, 2] = 0.0                                           # node 2 invalid -> its column (incl self-loop) zeroed
    with torch.no_grad():
        out = net(x, _adj_b(2, 4, nm))
    assert torch.isfinite(out).all()


# ------------------------- train_serial (real + stub) -------------------------

class _TinyD:
    def __init__(self, n=12, N=5, seq=6):
        rng = np.random.default_rng(0)
        self.tickers = [f"T{i}" for i in range(N)]
        mk = lambda m: rng.standard_normal((m, N, seq, 5)).astype(np.float32)
        one = lambda m: np.ones((m, N), dtype=np.float32)
        yv = lambda m: (np.abs(rng.standard_normal((m, N))) * 1e-3 + 1e-3).astype(float)
        ntr, nva, nte = n, 4, 4
        self.X_tr, self.X_va, self.X_te = mk(ntr), mk(nva), mk(nte)
        self.nmask_tr, self.nmask_va, self.nmask_te = one(ntr), one(nva), one(nte)
        self.tmask_tr, self.tmask_va, self.tmask_te = one(ntr), one(nva), one(nte)
        self.y_tr, self.y_va, self.y_te = yv(ntr), yv(nva), yv(nte)
        self.t_mean = np.full(N, 1e-3)
        self.t_std = np.full(N, 1e-3)
        self.d_te = [f"2020-01-{i + 1:02d}" for i in range(nte)]

    @property
    def N(self):
        return len(self.tickers)


def _cfg(epochs=1):
    return Config(epochs=epochs, seeds=(42,), min_epochs=1, patience=1, batch_size=4, hidden=8, heads=2)


def test_train_serial_real_net_finite(monkeypatch):
    """Real SERIAL net trains one epoch on tiny CPU data and returns finite, positive predictions."""
    monkeypatch.setattr(SH.torch.cuda, "is_available", lambda: False)   # force CPU (polite + deterministic)
    D = _TinyD()
    adj = np.eye(D.N, dtype=np.float32)
    pred = SH.train_serial(D, _cfg(1), seed=42, use_graph=True, adj=adj)
    assert pred.shape == (D.X_te.shape[0], D.N)
    assert np.isfinite(pred).all() and (pred > 0).all()


class _ConstNet(nn.Module):
    """Constant-output stub (grad flows to a dummy param but the output never changes) so val MSE is flat
    after epoch 0 -> forces the no-improvement + early-stop-break arcs of train_serial."""

    def __init__(self, *a, **k):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj_b):
        return self.p * 0.0 + torch.zeros(x.shape[0], x.shape[1])


def test_train_serial_earlystop_and_splits(monkeypatch):
    """Stub the net so val loss is flat -> covers improve(ep0)/no-improve(ep1)/break + return_splits arcs."""
    monkeypatch.setattr(SH.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(SH, "SerialLSTMGNN", lambda *a, **k: _ConstNet())
    D = _TinyD()
    adj = np.eye(D.N, dtype=np.float32)
    out = SH.train_serial(D, _cfg(3), seed=7, use_graph=True, adj=adj, return_splits=True)
    assert set(out) == {"test", "val", "train", "train_curve", "val_curve", "best_epoch"}
    assert len(out["val_curve"]) >= 2 and out["best_epoch"] == 1     # improved only at epoch 1, then broke
    assert len(out["val_curve"]) < 3                                 # early-stop broke before epoch 3
    assert np.isfinite(out["test"]).all()
    # return_splits=False arc
    te = SH.train_serial(D, _cfg(2), seed=7, use_graph=False, adj=adj, return_splits=False)
    assert te.shape == (D.X_te.shape[0], D.N)
