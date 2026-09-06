"""Single source of truth for every tunable constant of the XGBoost residual-ratio study.

Per CLAUDE.md "Single-source-of-truth app config (ENFORCED)": windows, floors, the XGBoost search
space, the alpha/clip guardrail grid, the OOF split parameters, the limit-lock multiplier, the seed and
the CPU thread count all live HERE, not as scattered literals in the pipeline modules. Shared pipeline
constants (floors, edge Top-K, HAR windows) are re-exported from the canonical ``pipeline_config`` rather
than re-declared, so this study never drifts from the project convention.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "submission" / "soict_lstm_gat"))
import pipeline_config as pc  # noqa: E402  (canonical shared constants, read-only)

# ---- shared with the rest of the project (re-exported, not re-declared) ----
QLIKE_FLOOR: float = pc.QLIKE_FLOOR                 # metric floor, identical across all compared models
POS_FLOOR_FRAC: float = pc.POS_FLOOR_FRAC           # shared per-node positivity floor fraction
POS_FLOOR_EPS: float = pc.POS_FLOOR_EPS
EDGE_TOP_K: int = pc.EDGE_TOP_K                     # graph neighbours per node (horizon-matched edge)
LOOKBACK: int = 10                                  # canonical split lookback (matches delivered edge_hmatched lb10 runs)
FOLDS_TARGET: int = 7                               # canonical number of walk-forward retrain points

# ---- causal feature windows (backward-looking) ----
PK_LAGS: tuple = (1, 2, 3, 5, 10, 22)              # Parkinson-variance lags
ROLL_MEANS: tuple = (5, 10, 22, 44)                # rolling-mean windows of pk
ROLL_STD_WINDOW: int = 22                          # rolling std of pk
ROLL_MED_WINDOW: int = 22                          # rolling median + MAD of pk
SLOPE_WINDOW: int = 22                             # rolling linear-trend window of pk
EWMA_SPANS: tuple = (5, 22)                        # short / long EWMA decay spans
RATIO_WINDOW: int = 22                             # rolling-mean window in the pk_t/(roll_mean+eps) ratio
RATIO_EPS: float = 1e-12                           # epsilon in pk_t/(roll_mean_22+eps)

# ---- chronological OOF HAR-X (residual-training target) ----
OOF_WARMUP_FRAC: float = 0.30                      # first fraction of TRAIN anchors used only to warm the OOF fits
OOF_SPLITS: int = 5                                # expanding-window blocks over the remaining TRAIN anchors
RESIDUAL_EPS: float = 1e-8                         # epsilon in z = log((y+eps)/(harx_oof+eps))

# ---- residual correction guardrail (chosen on validation) ----
ALPHA_GRID: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)   # shrinkage: yhat = HARX * exp(alpha*clip(zhat))
CLIP_GRID: tuple = (0.25, 0.5, 0.75, 1.0)         # symmetric clip bound c on zhat

# ---- XGBoost search space (small curated candidate list, not a full cross-product) ----
XGB_SEED: int = 42
XGB_N_JOBS: int = 4                                # bounded CPU threads (GPU agent runs concurrently; do not starve host)
XGB_EARLY_STOPPING_ROUNDS: int = 50
XGB_MAX_ESTIMATORS: int = 1000                     # upper bound; early stopping picks the effective count
XGB_GRID: tuple = (
    {"max_depth": 2, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.9,
     "colsample_bytree": 0.9, "reg_lambda": 5.0, "reg_alpha": 0.0},
    {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.9,
     "colsample_bytree": 0.9, "reg_lambda": 5.0, "reg_alpha": 0.0},
    {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 10, "subsample": 0.7,
     "colsample_bytree": 0.7, "reg_lambda": 10.0, "reg_alpha": 0.1},
    {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 10, "subsample": 0.7,
     "colsample_bytree": 0.7, "reg_lambda": 10.0, "reg_alpha": 1.0},
    {"max_depth": 6, "learning_rate": 0.01, "min_child_weight": 10, "subsample": 0.7,
     "colsample_bytree": 0.7, "reg_lambda": 10.0, "reg_alpha": 1.0},
)

# ---- limit-lock / robustness ----
LIMIT_LOCK_MULT: float = 2.0                       # lock cell if target <= LIMIT_LOCK_MULT*floor OR zero_range_flag
TOP_PCT_DATES: float = 0.01                        # top 1% of unique dates by aggregate QLIKE contribution
SHOCK_MONTH: str = "2025-04"                       # April-2025 shock-cluster month (YYYY-MM prefix of target date)
