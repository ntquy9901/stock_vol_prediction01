"""Checkpointed graph training with resume. One snapshot = one cross-stock graph at a target date."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn


_BATCH_FIELDS = ("price", "news", "news_mask", "ticker_ids", "adjacency", "target", "presence_mask")


def _collate(snaps, device):
    """Stack a list of B snapshots into batched [B, N, ...] tensors on `device` (N is fixed = 33).

    ``presence_mask`` defaults to all-ones when a snapshot omits it (real basis snapshots always
    carry it; some synthetic fixtures do not).
    """
    b = {f: torch.stack([s[f] for s in snaps]).to(device, non_blocking=True)
         for f in _BATCH_FIELDS if f in snaps[0]}
    if "presence_mask" not in b:
        b["presence_mask"] = torch.ones_like(b["target"])
    return b


def _masked_mse(pred, target, presence):
    """Mean squared error over PRESENT nodes only (presence-mask-aware), matching eval semantics."""
    se = (pred - target) ** 2 * presence
    return se.sum() / presence.sum().clamp_min(1.0)


def _batched_forward(model, batch, apply_graph=True):
    return model(batch["price"], batch["news"], batch["news_mask"], batch["ticker_ids"],
                 batch["adjacency"], apply_graph=apply_graph)


def _val_loss(model, snaps, device, apply_graph=True, batch_size=64):
    """Presence-masked MSE over the val snapshots, batched for speed."""
    model.eval()
    tot, cnt = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(snaps), batch_size):
            b = _collate(snaps[i:i + batch_size], device)
            se = (_batched_forward(model, b, apply_graph) - b["target"]) ** 2 * b["presence_mask"]
            tot += se.sum().item()
            cnt += b["presence_mask"].sum().item()
    return tot / max(cnt, 1.0)


def save_checkpoint(path, model, optimizer, epoch, best_val, best_state):
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "best_val": best_val, "best_state": best_state}, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def train_with_resume(model, train_snaps, val_snaps, ckpt_path: Path, epochs: int,
                      device, seed: int, resume: bool = False, apply_graph: bool = True,
                      patience: int = 0, min_epochs: int = 0, batch_size: int = 64):
    """Train up to `epochs`, keeping the best-val checkpoint. If patience>0, stop early once val
    loss has not improved for `patience` consecutive epochs (but never before `min_epochs`).
    Pooled models converge ~epoch 5-6 then overfit, so early stopping saves compute without cost.

    Mini-batch: `batch_size` snapshots per gradient step (was 1). This is the speed lever (GPU is
    ~idle at batch=1) but changes the optimization vs pure SGD -> results differ from the batch=1
    runs and may need LR re-tuning. Loss is presence-mask-aware (absent nodes excluded), matching
    evaluation.
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
        perm = rng.permutation(len(train_snaps))
        for i in range(0, len(perm), batch_size):
            batch = _collate([train_snaps[j] for j in perm[i:i + batch_size]], device)
            optimizer.zero_grad()
            loss = _masked_mse(_batched_forward(model, batch, apply_graph),
                               batch["target"], batch["presence_mask"])
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        vl = _val_loss(model, val_snaps, device, apply_graph, batch_size)
        last_epoch = epoch + 1
        if vl < best_val:
            best_val, best_state, since_improved = vl, copy.deepcopy(model.state_dict()), 0
        else:
            since_improved += 1
        if patience and last_epoch >= min_epochs and since_improved >= patience:
            break  # val loss plateaued -> stop (best_state already captured)
    save_checkpoint(ckpt_path, model, optimizer, last_epoch, best_val, best_state)
    return {"epoch": last_epoch, "best_val": best_val, "best_state": best_state}
