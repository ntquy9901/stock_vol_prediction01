"""SERIAL LSTM->GNN hybrid network + trainer (Sonani, Badii & Moin 2025, arXiv:2502.15813 sec 2.3).

The whole point (distinct from the delivered PARALLEL ``MaskedRichNet``): the GNN's node features ARE the
LSTM's per-stock temporal embeddings, not the raw day-t features.

  Stage 1  h_i = LSTM(x_i[SEQ,5])[-1]                 # per-stock temporal embedding  [B,N,hidden]
  Stage 2  g_i = WeightedGAT(h, A_b)                  # GNN INPUT = h  (the LSTM embedding)  [B,N,hidden*heads]
  Head     y_i = head( concat[h_i, g_i] )             # residual skip on h; see design.md sec 3

Contrast: the delivered PARALLEL model feeds ``GAT(x[:, :, -1, :], A_b)`` -- the RAW day-t features, NOT h
(CLAUDE.md "GAT uses raw features"). Here the graph only ever sees temporal embeddings, so the SEQ lookback
propagates through the graph (it cannot in the parallel model, whose GAT sees a single day).

The ``WeightedGATLayer`` (weight/sign-aware, mask-aware multi-head GAT) is reused READ-ONLY from
``run_masked_rich``; only the wiring (its INPUT is h, one hop) is new here. The trainer mirrors
``train_masked_rich``'s zscore-floor path (per-node target StandardScaler, linear denorm, shared 1e-2*mean
positivity floor, Adam + ReduceLROnPlateau, grad-clip, early stop on val MSE, per-epoch learning curves),
swapping only the network. Batched, mask-aware, GPU when available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "scripts" / "eda",
           REPO / "scripts" / "quality_gate",
           str(_HERE)):
    sys.path.insert(0, str(_p))

import masked_rich as MR                      # noqa: E402  (read-only: N_FEAT)
from run_masked_rich import WeightedGATLayer  # noqa: E402  (read-only: reuse the GAT layer)


class SerialLSTMGNN(nn.Module):
    """LSTM temporal encoder feeding a 1-hop weighted-GAT over the LSTM embeddings (SERIAL).

    ``forward(x, adj_b)`` with ``x`` ``[B,N,SEQ,in_dim]`` and ``adj_b`` ``[B,N,N]`` (invalid-neighbour
    columns already zeroed by the caller). Returns ``[B,N]``. When ``use_graph=False`` the head sees only the
    LSTM embedding -> the plain temporal (no-graph) baseline with the SAME architecture.
    """

    def __init__(self, hidden: int = 64, heads: int = 4, dropout: float = 0.2, use_graph: bool = True,
                 in_dim: int = MR.N_FEAT):
        super().__init__()
        self.use_graph, self.hidden = use_graph, hidden
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True, dropout=dropout)
        gdim = hidden * heads if use_graph else 0
        if use_graph:
            self.gat = WeightedGATLayer(hidden, hidden, heads)   # INPUT dim = hidden (the LSTM embedding)
        self.head = nn.Sequential(nn.Linear(hidden + gdim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def lstm_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Per-stock temporal embedding h [B,N,hidden] = last LSTM hidden state. Exposed so a test can pin
        that the GAT's input equals THIS (the serial contract) and that SEQ-history changes it."""
        b, n, seq, d = x.shape
        out, _ = self.lstm(x.reshape(b * n, seq, d))
        return out[:, -1].reshape(b, n, self.hidden)

    def forward(self, x: torch.Tensor, adj_b: torch.Tensor) -> torch.Tensor:
        h = self.lstm_embed(x)
        parts = [h]
        if self.use_graph:
            parts.append(self.gat(h, adj_b))          # SERIAL: GNN consumes the LSTM embedding h
        return self.head(torch.cat(parts, -1)).squeeze(-1)


def resolve_device() -> torch.device:
    """Training device -- torch.cuda.is_available() is the single source of truth (design.md sec 5.3)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches(nrow, bs, seed):
    idx = np.arange(nrow)
    np.random.default_rng(seed).shuffle(idx)
    for i in range(0, nrow, bs):
        yield idx[i:i + bs]


def train_serial(D, cfg, seed, use_graph, adj, return_splits=False):
    """Train the SERIAL model on the masked panel (zscore-floor parameterisation). Returns the test
    prediction ``[n,N]`` (or, with ``return_splits``, a dict with test/val/train predictions + per-epoch
    train/val MSE learning curves + best_epoch for the over/under-fit evidence)."""
    dev = resolve_device()
    torch.manual_seed(seed); np.random.seed(seed)
    net = SerialLSTMGNN(cfg.hidden, cfg.heads, cfg.dropout, use_graph).to(dev)
    base = torch.from_numpy(adj).to(dev)
    tmean = torch.from_numpy(D.t_mean.astype(np.float32)).to(dev)
    tstd = torch.from_numpy(D.t_std.astype(np.float32)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    Xtr = torch.from_numpy(D.X_tr).to(dev); nmtr = torch.from_numpy(D.nmask_tr).to(dev)
    tmtr = torch.from_numpy(D.tmask_tr).to(dev)
    ytr_n = (torch.from_numpy(D.y_tr.astype(np.float32)).to(dev) - tmean) / tstd
    bs = cfg.batch_size

    def adj_batch(nm):                              # [b,N,N] = base * valid-neighbour (source) mask
        return base.unsqueeze(0) * nm.unsqueeze(1)

    def infer(X_np, nm_np):
        net.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(X_np), bs):
                xb = torch.from_numpy(X_np[i:i + bs]).to(dev)
                nmb = torch.from_numpy(nm_np[i:i + bs]).to(dev)
                outs.append(net(xb, adj_batch(nmb)).cpu().numpy())
        pn = np.concatenate(outs)
        return np.maximum(pn * D.t_std + D.t_mean, 1e-2 * D.t_mean + 1e-12)

    best = np.inf
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    wait = 0; best_ep = 0
    train_curve = []; val_curve = []
    for ep in range(cfg.epochs):
        net.train()
        for idx in _batches(len(Xtr), bs, seed + ep):
            xb = Xtr[idx]; nmb = nmtr[idx]; tmb = tmtr[idx]; yb = ytr_n[idx]
            opt.zero_grad()
            pred = net(xb, adj_batch(nmb))
            loss = (((pred - yb) ** 2) * tmb).sum() / tmb.sum().clamp(min=1)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip); opt.step()
        pva = infer(D.X_va, D.nmask_va); mva = D.tmask_va.astype(bool)
        vmse = float(np.mean((pva[mva] - D.y_va[mva]) ** 2))
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
    net.load_state_dict(best_state)
    te = infer(D.X_te, D.nmask_te)
    if return_splits:
        return {"test": te, "val": infer(D.X_va, D.nmask_va), "train": infer(D.X_tr, D.nmask_tr),
                "train_curve": train_curve, "val_curve": val_curve, "best_epoch": best_ep}
    return te
