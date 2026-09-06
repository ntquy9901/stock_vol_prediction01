"""Single source of truth for the MASTER baseline's tunable constants.

MASTER-specific architecture/training hyperparameters live HERE (per CLAUDE.md config-single-source
rule); floors, windows, seeds, lookback and the QLIKE floor are re-used from the canonical
``pipeline_config`` so a value is edited in exactly one place. No tunable is hardcoded in ``master_net``
or ``run_master``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "submission" / "soict_lstm_gat",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pipeline_config as pc  # noqa: E402  (canonical floors/windows/seeds/lookback)

# ---- MASTER architecture (arXiv:2312.15235) ----
N_STOCK_FEAT: int = 5              # [pk, har_weekly, har_monthly, market_pk, volume_zscore] (gate_input_start)
N_MARKET_FEAT: int = 4             # causal market features appended for the gate (gate_input_end - start)
D_MODEL: int = 128                 # transformer width (modest for the 8 GB RTX 4060; divisible by both nheads)
T_NHEAD: int = 4                   # intra-stock temporal-attention heads (D_MODEL % T_NHEAD == 0)
S_NHEAD: int = 2                   # inter-stock (dense cross-sectional) attention heads (D_MODEL % S_NHEAD == 0)
DROPOUT: float = 0.2               # attention + FFN dropout (project default; reference used 0.5 on 158 feats)
BETA: float = 2.0                  # market-gate softmax temperature (reference: 5 for csi300, 2 for csi800)

# ---- training budget (matches the delivered walk-forward budget used for the cited LSTM/VolGA) ----
LR: float = 5e-4                   # Adam lr (lower than the LSTM 1e-3: attention is more lr-sensitive)
WEIGHT_DECAY: float = pc.WEIGHT_DECAY
GRAD_CLIP: float = pc.GRAD_CLIP
EPOCHS: int = 16                   # max epochs (early-stop on val); == edge_hmatched budget for fairness
PATIENCE: int = 5                  # early-stop patience on val MSE (matches the delivered walk-forward budget)
MIN_EPOCHS: int = pc.MIN_EPOCHS    # minimum epochs before early stop may fire (canonical = 5)
BATCH_SIZE: int = 64               # anchors per minibatch (SAttention [B,T,N,N] scores bound VRAM for N=102)
MAX_LEN: int = 100                 # positional-encoding table length (>= any lookback used)

# ---- re-used canonical constants (edit in pipeline_config only) ----
LOOKBACK: int = pc.LOOKBACK        # 10 (canonical); the cited LSTM/VolGA also use lb10
FOLDS_TARGET: int = 7              # expanding-window retrain points over the OOS region (canonical for edge_hmatched)
SEEDS: tuple = pc.SEEDS            # (42, 123, 2026, 7, 2024)
QLIKE_FLOOR: float = pc.QLIKE_FLOOR
POS_FLOOR_FRAC: float = pc.POS_FLOOR_FRAC
POS_FLOOR_EPS: float = pc.POS_FLOOR_EPS
SCALER_EPS: float = pc.SCALER_EPS
MARKET_Z_WINDOW: int = pc.HAR_MONTHLY_WINDOW   # 22-day causal window for the market vol-z feature
LIMIT_LOCK_MULT: float = 2.0       # non-lock conditional QLIKE drops test obs with target <= mult*floor


@dataclass(frozen=True)
class MasterConfig:
    """Resolved MASTER hyperparameters for one run (overridable for smoke / seed-count sweeps)."""
    d_model: int = D_MODEL
    t_nhead: int = T_NHEAD
    s_nhead: int = S_NHEAD
    dropout: float = DROPOUT
    beta: float = BETA
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    grad_clip: float = GRAD_CLIP
    epochs: int = EPOCHS
    patience: int = PATIENCE
    min_epochs: int = MIN_EPOCHS
    batch_size: int = BATCH_SIZE
    max_len: int = MAX_LEN
    n_stock_feat: int = N_STOCK_FEAT
    n_market_feat: int = N_MARKET_FEAT
    qlike_floor: float = QLIKE_FLOOR
    pos_floor_frac: float = POS_FLOOR_FRAC
    pos_floor_eps: float = POS_FLOOR_EPS
    scaler_eps: float = SCALER_EPS
    market_z_window: int = MARKET_Z_WINDOW
    limit_lock_mult: float = LIMIT_LOCK_MULT
    seeds: tuple = field(default=SEEDS)


SMOKE = MasterConfig(epochs=2, min_epochs=1, patience=1, batch_size=32, seeds=(42, 123))
