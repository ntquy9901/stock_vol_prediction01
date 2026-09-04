"""Batched trainer for PatchTSTRichNet — a mirror of ``run_masked_rich.train_masked_rich`` with the
LSTM net swapped for ``PatchTSTRichNet``. Same scaling, masked pooled MSE, ReduceLROnPlateau, early
stop, learning curves, positivity floor, and ``zscore_floor``/``ratio_exp`` output params, so the
result schema and over/under-fit evidence match VolGA exactly.

GPU-first (uses ``cuda`` when available); the tiny CPU smokes in this baseline force CPU via
``CUDA_VISIBLE_DEVICES=`` because the GPU is busy with an overnight run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "submission" / "soict_lstm_gat",
           _REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           _REPO / "scripts" / "quality_gate", str(Path(__file__).resolve().parent)):
    sys.path.insert(0, str(_p))

import pipeline_config as pc  # noqa: E402  (single source of truth for shared tunables)
from run_masked_rich import _batches  # noqa: E402  (identical mini-batch index generator)

from patchtst_config import PatchTSTHParams  # noqa: E402
from patchtst_net import PatchTSTRichNet  # noqa: E402


def train_patchtst(D, cfg, seed, use_graph, adj, output_param="zscore_floor",
                   return_splits=False, hp: PatchTSTHParams | None = None):
    """Train one PatchTSTRichNet (mirror of train_masked_rich). ``hp`` = PatchTST knobs (defaults from
    patchtst_config). ``output_param`` as in train_masked_rich."""
    hp = hp or PatchTSTHParams()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    seq_len = D.X_tr.shape[2]
    net = PatchTSTRichNet(seq_len, cfg.hidden, cfg.heads, cfg.dropout, use_graph, hp=hp).to(dev)
    if output_param == "ratio_exp":                 # bias-match: exp(0)=1 starts at the mean ratio
        with torch.no_grad():
            net.head[-1].bias.fill_(0.0)
    base = torch.from_numpy(adj).to(dev)
    tmean = torch.from_numpy(D.t_mean.astype(np.float32)).to(dev)
    tstd = torch.from_numpy(D.t_std.astype(np.float32)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    Xtr = torch.from_numpy(D.X_tr).to(dev); nmtr = torch.from_numpy(D.nmask_tr).to(dev)
    tmtr = torch.from_numpy(D.tmask_tr).to(dev)
    if output_param == "ratio_exp":
        ytr_n = torch.from_numpy(D.y_tr.astype(np.float32)).to(dev) / (tmean + 1e-12)
    else:
        ytr_n = ((torch.from_numpy(D.y_tr.astype(np.float32)).to(dev) - tmean) / tstd)
    bs = cfg.batch_size
    tiny = float(np.finfo(np.float32).tiny)

    def adj_batch(nm):                              # [b,N,N] = base * valid-neighbour (source) mask
        return base.unsqueeze(0) * nm.unsqueeze(1)

    def _apply(pn):                                 # network output -> prediction in the training space
        return torch.exp(pn.clamp(max=15.0)) if output_param == "ratio_exp" else pn

    def infer(X_np, nm_np):
        net.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(X_np), bs):
                xb = torch.from_numpy(X_np[i:i + bs]).to(dev)
                nmb = torch.from_numpy(nm_np[i:i + bs]).to(dev)
                outs.append(net(xb, adj_batch(nmb)).cpu().numpy())
        pn = np.concatenate(outs)
        if output_param == "ratio_exp":             # exp(pn)*mean: positive by construction
            return np.maximum(np.exp(np.minimum(pn, 15.0)) * D.t_mean, tiny)
        return np.maximum(pn * D.t_std + D.t_mean, pc.POS_FLOOR_FRAC * D.t_mean + pc.POS_FLOOR_EPS)

    best = np.inf; best_state = None; wait = 0; best_ep = 0
    train_curve = []; val_curve = []                         # per-epoch masked MSE -> learning curve
    for ep in range(cfg.epochs):
        net.train()
        for idx in _batches(len(Xtr), bs, True, seed + ep):
            xb = Xtr[idx]; nmb = nmtr[idx]; tmb = tmtr[idx]; yb = ytr_n[idx]
            opt.zero_grad(); pred = _apply(net(xb, adj_batch(nmb)))
            loss = (((pred - yb) ** 2) * tmb).sum() / tmb.sum().clamp(min=1)
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip); opt.step()
        pva = infer(D.X_va, D.nmask_va)
        m = D.tmask_va.astype(bool)
        vmse = float(np.mean((pva[m] - D.y_va[m]) ** 2))
        ptr = infer(D.X_tr, D.nmask_tr); mtr = D.tmask_tr.astype(bool)   # train-fit curve (underfit evidence)
        train_curve.append(float(np.mean((ptr[mtr] - D.y_tr[mtr]) ** 2)))
        val_curve.append(vmse)
        sched.step(vmse)
        if vmse < best - 1e-12:
            best = vmse; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}; wait = 0; best_ep = ep + 1
        else:
            wait += 1
        if ep + 1 >= cfg.min_epochs and wait >= cfg.patience:
            break
    if best_state:
        net.load_state_dict(best_state)
    te = infer(D.X_te, D.nmask_te)
    if return_splits:                                        # train/val preds + learning curves for the fit verdict
        return {"test": te, "val": infer(D.X_va, D.nmask_va), "train": infer(D.X_tr, D.nmask_tr),
                "train_curve": train_curve, "val_curve": val_curve, "best_epoch": best_ep}
    return te
