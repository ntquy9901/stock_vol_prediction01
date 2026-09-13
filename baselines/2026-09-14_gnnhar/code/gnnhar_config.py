"""Single source of truth for this baseline's tunable constants (avoids the config-hardcode gate).

GNNHAR (Zhang, Pu, Cucuringu & Dong, Int. J. Forecasting 2024, arXiv:2308.01419; official code
https://github.com/chaozhang-ox/GNNHAR). Feature sets are derived from ``full_matrix`` so the own-history
column set stays single-sourced with every other model (edit ``FM.OWN`` in one place). See design/design.md.
"""
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
if str(REPO / "scripts" / "eda") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "eda"))  # pragma: no cover - path bootstrap (pre-seeded in tests)
import full_matrix as FM  # noqa: E402  (read-only reuse of the shared feature/column definitions)

HAR3 = list(FM.HAR)                              # 3 HAR lags = GNNHAR's node inputs (paper) + HAR baseline
OWN8 = [c for c in FM.OWN if c != "rq"]          # own-8 gamma-GBM feature set (paper GBM(own-8); rq dropped)
SEEDS = (0, 1, 2)                                # multi-seed ensemble + over/under-fit evidence
N_HID = 9                                         # paper default hidden width (GNNHAR --n_hid)
N_GCN = 2                                         # GNNHAR2L: 2 graph-conv layers (the paper's best variant)
VALID_LEN = 22                                    # trailing validation dates for early stopping (--valid_len)
LR = 1e-3                                         # Adam learning rate (GNNHAR.py:357)
WEIGHT_DECAY = 1e-5                               # Adam L2 (GNNHAR.py:357)
BATCH_DATES = 256                                # dates per batched GPU forward (no batch=1)
MAX_EPOCHS = 120                                 # upper bound; early stopping normally stops far sooner
MIN_EPOCHS = 20                                  # do not early-stop before this many epochs
PATIENCE = 15                                    # early-stopping patience on validation QLIKE
GRAD_CLIP = 1.0                                  # gradient-norm clip
COLLAPSE_VAL = 1.6                               # scaled val-QLIKE(+1) above this => dead-ReLU -> restart
MAX_RESTARTS = 4                                 # restart-with-new-seed attempts on collapse
HORIZONS = (1, 5, 10, 22)                        # forecast horizons (shared comparison table)
MIN_TRAIN_ROWS = {"sp500": 30000, "default": 3000}   # min pooled train rows per fold (sibling fold gate)
SMOKE_EPOCHS = 40                                # --smoke: 1 fold, 1 seed, few epochs (fast sanity)
