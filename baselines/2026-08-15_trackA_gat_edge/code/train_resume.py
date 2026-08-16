"""Checkpointed graph training with resume. One snapshot = one cross-stock graph at a target date."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _forward_snap(model, snap, device, apply_graph=True):
    pred = model(snap["price"].unsqueeze(0).to(device), snap["news"].unsqueeze(0).to(device),
                 snap["news_mask"].unsqueeze(0).to(device),
                 snap["ticker_ids"].unsqueeze(0).to(device),
                 snap["adjacency"].unsqueeze(0).to(device), apply_graph=apply_graph)
    return pred.squeeze(0)


def _val_loss(model, snaps, device, apply_graph=True):
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for s in snaps:
            p = _forward_snap(model, s, device, apply_graph)
            t = s["target"].to(device)
            tot += torch.mean((p - t) ** 2).item() * len(t)
            cnt += len(t)
    return tot / max(cnt, 1)


def save_checkpoint(path, model, optimizer, epoch, best_val, best_state):
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "best_val": best_val, "best_state": best_state}, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def train_with_resume(model, train_snaps, val_snaps, ckpt_path: Path, epochs: int,
                      device, seed: int, resume: bool = False, apply_graph: bool = True,
                      patience: int = 0, min_epochs: int = 0):
    """Train up to `epochs`, keeping the best-val checkpoint. If patience>0, stop early once val
    loss has not improved for `patience` consecutive epochs (but never before `min_epochs`).
    Pooled models converge ~epoch 5-6 then overfit, so early stopping saves compute without cost.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    start_epoch, best_val = 0, float("inf")
    best_state = copy.deepcopy(model.state_dict())
    if resume and Path(ckpt_path).exists():
        ck = load_checkpoint(ckpt_path)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch, best_val, best_state = ck["epoch"], ck["best_val"], ck["best_state"]
    rng = np.random.default_rng(seed + start_epoch)
    since_improved, last_epoch = 0, start_epoch
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        for i in rng.permutation(len(train_snaps)):
            s = train_snaps[i]
            optimizer.zero_grad()
            loss = torch.mean((_forward_snap(model, s, device, apply_graph) - s["target"].to(device)) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        vl = _val_loss(model, val_snaps, device, apply_graph)
        last_epoch = epoch + 1
        if vl < best_val:
            best_val, best_state, since_improved = vl, copy.deepcopy(model.state_dict()), 0
        else:
            since_improved += 1
        if patience and last_epoch >= min_epochs and since_improved >= patience:
            break  # val loss plateaued -> stop (best_state already captured)
    save_checkpoint(ckpt_path, model, optimizer, last_epoch, best_val, best_state)
    return {"epoch": last_epoch, "best_val": best_val, "best_state": best_state}
