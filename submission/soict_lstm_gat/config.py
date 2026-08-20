"""Central configuration for the SOICT HAR-LSTM-GAT experiment suite (self-contained submission)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    lookback: int = 10                       # main lookback (22 = variation)
    horizons: tuple = (1, 5)                 # h1 = next day, h5 = next trading week
    seeds: tuple = (42, 123, 2026, 7, 2024)  # 5 seeds
    epochs: int = 20                         # max epochs
    patience: int = 3                        # early-stop patience on val MSE
    min_epochs: int = 5
    hidden: int = 64
    heads: int = 4
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    top_k: int = 5                           # graphical-lasso Top-K edges
    qlike_floor: float = 1e-8                # identical across all compared models
    batch_size: int = 512
    train_frac: float = 0.80
    val_frac: float = 0.10                   # test = 1 - train - val = 0.10
    data_root: str = "data"                  # relative to this folder


# Smoke config for fast tests / CI (2 epochs, 1 seed)
SMOKE = Config(epochs=2, seeds=(42,), min_epochs=1, patience=1, batch_size=64)
