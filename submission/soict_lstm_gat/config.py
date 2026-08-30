"""Central configuration for the SOICT HAR-LSTM-GAT experiment suite (self-contained submission).

Thin dataclass VIEW over the single canonical source of truth ``pipeline_config``: every field default
sources from ``pc.*`` so a value is edited in exactly one place (no drift). ``horizons`` stays the
submission-local vestigial default ``(1, 5)``; the canonical pipeline horizons are ``pc.HORIZONS``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pipeline_config as pc


@dataclass(frozen=True)
class Config:
    lookback: int = pc.LOOKBACK              # main lookback (22 = variation)
    horizons: tuple = (1, 5)                 # submission-local default; canonical = pc.HORIZONS
    seeds: tuple = pc.SEEDS                  # 5 seeds
    epochs: int = pc.EPOCHS                  # max epochs
    patience: int = pc.PATIENCE              # early-stop patience on val MSE
    min_epochs: int = pc.MIN_EPOCHS
    hidden: int = pc.HIDDEN
    heads: int = pc.HEADS
    dropout: float = pc.DROPOUT
    lr: float = pc.LR
    weight_decay: float = pc.WEIGHT_DECAY
    grad_clip: float = pc.GRAD_CLIP
    top_k: int = pc.EDGE_TOP_K               # graphical-lasso Top-K edges
    qlike_floor: float = pc.QLIKE_FLOOR      # identical across all compared models
    batch_size: int = pc.BATCH_SIZE
    train_frac: float = pc.TRAIN_FRAC
    val_frac: float = pc.VAL_FRAC            # test = 1 - train - val = 0.10
    data_root: str = "data"                  # relative to this folder


# Smoke config for fast tests / CI (2 epochs, 1 seed)
SMOKE = Config(epochs=2, seeds=(42,), min_epochs=1, patience=1, batch_size=64)
