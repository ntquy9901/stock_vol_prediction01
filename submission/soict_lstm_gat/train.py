"""Pooled training loop for the graph model over common-date snapshots (MSE loss, early-stop on val MSE)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from model import HARLSTMGAT  # noqa: E402


def _stack(snaps: list) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    x = torch.from_numpy(np.stack([s["x"] for s in snaps])).float()      # [B,N,seq,3]
    y = torch.from_numpy(np.stack([s["y_norm"] for s in snaps])).float() # [B,N]
    y_raw = np.stack([s["y_raw"] for s in snaps])                        # [B,N]
    return x, y, y_raw


def _plot_curve(history: dict, path: Path) -> None:
    plt.figure(figsize=(5, 3))
    plt.plot(history["train_mse"], label="train MSE")
    plt.plot(history["val_mse"], label="val MSE")
    plt.xlabel("epoch"); plt.ylabel("MSE (norm scale)"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=110); plt.close()


def train_pooled(snap, adj: np.ndarray, cfg, seed: int, use_graph: bool, out_dir: Path,
                 log=None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    adj_t = torch.from_numpy(adj).float().to(device)

    Xtr, Ytr, _ = _stack(snap.train)
    Xva, Yva, _ = _stack(snap.val)
    Xva, Yva = Xva.to(device), Yva.to(device)
    model = HARLSTMGAT(price_dim=3, hidden=cfg.hidden, heads=cfg.heads,
                       dropout=cfg.dropout, use_graph=use_graph).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    lossf = nn.MSELoss()
    n = len(Xtr); bs = cfg.batch_size
    hist = {"train_mse": [], "val_mse": []}
    best_val = np.inf; best_state = None; wait = 0

    def _log(msg):
        print(msg, flush=True)
        if log is not None:
            log.write(msg + "\n"); log.flush()

    _log(f"[hyperparams] seed={seed} use_graph={use_graph} nodes={snap.num_nodes} "
         f"lookback={snap.lookback} horizon={snap.horizon} epochs={cfg.epochs} lr={cfg.lr} "
         f"hidden={cfg.hidden} dropout={cfg.dropout} wd={cfg.weight_decay} device={device} "
         f"train_snaps={len(snap.train)} val_snaps={len(snap.val)} test_snaps={len(snap.test)}")

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        tr_loss = 0.0; nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx].to(device, non_blocking=True)
            yb = Ytr[idx].to(device, non_blocking=True)
            opt.zero_grad()
            out = model(xb, adj_t)
            loss = lossf(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_loss += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vp = model(Xva, adj_t)
            val_mse = lossf(vp, Yva).item()
        tr_mse = tr_loss / max(nb, 1)
        hist["train_mse"].append(tr_mse); hist["val_mse"].append(val_mse)
        sched.step(val_mse)
        _log(f"[epoch {epoch + 1}/{cfg.epochs}] train_mse={tr_mse:.6f} val_mse={val_mse:.6f}")
        if val_mse < best_val - 1e-7:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if (epoch + 1) % 5 == 0:
            _plot_curve(hist, out_dir / f"learning_curve_ep{epoch + 1}.png")
        if epoch + 1 >= cfg.min_epochs and wait >= cfg.patience:
            _log(f"[early-stop] epoch {epoch + 1} (no val improvement for {cfg.patience})")
            break

    _plot_curve(hist, out_dir / "learning_curve_final.png")
    if best_state is not None:
        model.load_state_dict(best_state)
    (out_dir / "history.json").write_text(json.dumps(hist), encoding="utf-8")
    return {"model": model, "adj_t": adj_t, "history": hist, "epochs_ran": len(hist["train_mse"]),
            "best_val_mse": best_val, "device": device}
