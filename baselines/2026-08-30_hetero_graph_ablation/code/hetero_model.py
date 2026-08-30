"""Heterogeneous 2-relation LSTM+GAT model + its batched, mask-aware GPU training loop.

``HeteroRichNet`` = the shared LSTM temporal branch (identical to ``MaskedRichNet``) + TWO relation-specific
weighted-GAT branches with INDEPENDENT parameters:

  * ``gat_lin{1,2}`` consume the ``linear_corr`` adjacency,
  * ``gat_nl{1,2}``  consume the ``nonlinear_assoc`` adjacency.

Two separate ``WeightedGATLayer`` instances = heterogeneous message passing: PyTorch clones the conv weights
per instance, so the network learns WHEN to weight the linear vs the non-linear signal. The two relation node
updates are SUM-aggregated (PyG ``HeteroConv`` default) before the fusion head, keeping the head input dim
identical to the single-relation ``MaskedRichNet`` -> a clean controlled comparison. Both GAT branches read
the RAW node features at day t (``x[:,:,-1,:]``), PARALLEL to the LSTM (matching the delivered architecture).

``train_hetero_rich`` mirrors ``run_masked_rich.train_masked_rich`` (per-node zscore target scaler, 1e-2*mean
QLIKE floor, Adam + ReduceLROnPlateau, early stop on val MSE, per-epoch learning curves for over/under-fit
evidence) but drives the two adjacencies and supports ONLY the delivered ``zscore_floor`` output param
(minimum code). Fully batched ``[B,N,...]`` on GPU; both adjacencies masked per batch via
``base * nmask.unsqueeze(1)`` (mask-aware); no per-item Python loop in the hot path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_HAR = Path(__file__).resolve().parents[2] / "2026-08-21_har_anchored_residual" / "code"
sys.path.insert(0, str(_HAR))

import masked_rich as MR          # noqa: E402  (N_FEAT)
import run_masked_rich as RMR     # noqa: E402  (WeightedGATLayer, _batches -- read-only reuse)


class HeteroRichNet(nn.Module):
    """LSTM(5-feat) + two INDEPENDENT 2-hop weighted-GAT branches (linear_corr / nonlinear_assoc), SUM-agg."""

    def __init__(self, hidden: int = 64, heads: int = 4, dropout: float = 0.2,
                 in_dim: int = MR.N_FEAT, gat_layers: int = 2):
        super().__init__()
        self.hidden, self.gat_layers = hidden, gat_layers
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
        gdim = hidden * heads
        # relation linear_corr (independent params)
        self.gat_lin1 = RMR.WeightedGATLayer(in_dim, hidden, heads)
        self.gat_lin2 = RMR.WeightedGATLayer(gdim, hidden, heads)
        # relation nonlinear_assoc (independent params -- distinct instances -> cloned weights)
        self.gat_nl1 = RMR.WeightedGATLayer(in_dim, hidden, heads)
        self.gat_nl2 = RMR.WeightedGATLayer(gdim, hidden, heads)
        self.head = nn.Sequential(nn.Linear(hidden + gdim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def _relation(self, gat1, gat2, node_raw, adj_b):
        out = gat1(node_raw, adj_b)
        if self.gat_layers == 2:
            out = gat2(out, adj_b)      # 2nd hop, same masked relation adjacency
        return out

    def forward(self, x, adj_lin_b, adj_nl_b):      # x [B,N,seq,5]; adj_*_b [B,N,N] (invalid source cols zeroed)
        b, n, seq, d = x.shape
        out, _ = self.lstm(x.reshape(b * n, seq, d))
        h = out[:, -1].reshape(b, n, self.hidden)
        node_raw = x[:, :, -1, :]
        g_lin = self._relation(self.gat_lin1, self.gat_lin2, node_raw, adj_lin_b)
        g_nl = self._relation(self.gat_nl1, self.gat_nl2, node_raw, adj_nl_b)
        g = g_lin + g_nl                            # SUM aggregation across relations
        return self.head(torch.cat([h, g], -1)).squeeze(-1)


def train_hetero_rich(D, cfg, seed, adj_lin, adj_nl, return_splits=False):
    """Train HeteroRichNet on the masked panel with the two relation adjacencies (zscore_floor output).

    Returns the TEST prediction array, or (return_splits) a dict with test/val/train predictions + per-epoch
    train/val MSE curves + best_epoch (over/under-fit evidence)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    net = HeteroRichNet(cfg.hidden, cfg.heads, cfg.dropout).to(dev)
    base_l = torch.from_numpy(adj_lin).to(dev)
    base_n = torch.from_numpy(adj_nl).to(dev)
    tmean = torch.from_numpy(D.t_mean.astype(np.float32)).to(dev)
    tstd = torch.from_numpy(D.t_std.astype(np.float32)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    Xtr = torch.from_numpy(D.X_tr).to(dev); nmtr = torch.from_numpy(D.nmask_tr).to(dev)
    tmtr = torch.from_numpy(D.tmask_tr).to(dev)
    ytr_n = (torch.from_numpy(D.y_tr.astype(np.float32)).to(dev) - tmean) / tstd
    bs = cfg.batch_size

    def adj_batch(nm):                              # both relations masked by the SAME valid-neighbour mask
        m = nm.unsqueeze(1)
        return base_l.unsqueeze(0) * m, base_n.unsqueeze(0) * m

    def infer(X_np, nm_np):
        net.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(X_np), bs):
                xb = torch.from_numpy(X_np[i:i + bs]).to(dev)
                nmb = torch.from_numpy(nm_np[i:i + bs]).to(dev)
                al, an = adj_batch(nmb)
                outs.append(net(xb, al, an).cpu().numpy())
        pn = np.concatenate(outs)
        return np.maximum(pn * D.t_std + D.t_mean, 1e-2 * D.t_mean + 1e-12)

    best = np.inf; best_state = None; wait = 0; best_ep = 0
    train_curve = []; val_curve = []
    for ep in range(cfg.epochs):
        net.train()
        for idx in RMR._batches(len(Xtr), bs, True, seed + ep):
            xb = Xtr[idx]; nmb = nmtr[idx]; tmb = tmtr[idx]; yb = ytr_n[idx]
            al, an = adj_batch(nmb)
            opt.zero_grad(); pred = net(xb, al, an)
            loss = (((pred - yb) ** 2) * tmb).sum() / tmb.sum().clamp(min=1)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip); opt.step()
        pva = infer(D.X_va, D.nmask_va); m = D.tmask_va.astype(bool)
        vmse = float(np.mean((pva[m] - D.y_va[m]) ** 2))
        ptr = infer(D.X_tr, D.nmask_tr); mtr = D.tmask_tr.astype(bool)
        train_curve.append(float(np.mean((ptr[mtr] - D.y_tr[mtr]) ** 2)))
        val_curve.append(vmse)
        sched.step(vmse)
        if vmse < best - 1e-12:
            best = vmse; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            wait = 0; best_ep = ep + 1
        else:
            wait += 1
        if ep + 1 >= cfg.min_epochs and wait >= cfg.patience:
            break
    if best_state:
        net.load_state_dict(best_state)
    te = infer(D.X_te, D.nmask_te)
    if return_splits:
        return {"test": te, "val": infer(D.X_va, D.nmask_va), "train": infer(D.X_tr, D.nmask_tr),
                "train_curve": train_curve, "val_curve": val_curve, "best_epoch": best_ep}
    return te
