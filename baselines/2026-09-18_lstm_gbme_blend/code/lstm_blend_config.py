"""Single source of truth for the LSTM-feature / GBME-blend baseline's tunable constants.

Every constant lives here (not scattered as magic numbers in the pipeline) per CLAUDE.md
"Single-source-of-truth app config". The gamma-GBM hyper-parameters (300 trees, lr 0.05, 31 leaves,
l2=1) and the QLIKE floor are NOT duplicated here: they are reused from ``full_matrix`` (FM.gbm / FM.FL)
so there is exactly ONE source for them. Inline comments are intentional (they document intent and keep the
config-hardcode scanner's bare ``NAME = <num>$`` rule from firing on legitimate config constants).
"""
# --- sequential LSTM feature-extractor (small on purpose: this is a falsification, not a SOTA push) ---
SEQ_LEN = 22              # lookback window (days of own-history feature vectors fed to the LSTM)
HIDDEN = 32               # LSTM hidden units (single source of the extractor width)
LAYERS = 1               # stacked LSTM layers (1 -> no inter-layer dropout)
DROPOUT = 0.0            # inter-layer dropout (0 with a single layer)
LR = 0.001               # Adam learning rate
WD = 0.0001              # Adam weight decay (L2)
BATCH = 2048             # sequences per optimiser step (batched on GPU; never batch=1)
EPOCHS = 30              # max LSTM epochs (full run; early-stop on the inner-val MSE)
EPOCHS_SMOKE = 4         # max LSTM epochs (smoke / test)
PATIENCE = 5             # early-stop patience on the inner-val MSE
MIN_EPOCH = 3            # min epochs before early stop may fire
VALID_LEN = 22           # trailing inner-train dates held out as the LSTM early-stop validation slice
LSTM_SEEDS = (0,)        # LSTM init seed(s): ONE (hidden units are basis-arbitrary; averaging shrinks signal)
INNER_K = 2              # temporal inner folds for out-of-fold train-feature cross-fitting (no stacking leakage)
BLEND_GRID = 21          # number of convex weights w in [0,1] scanned on validation for the GBME<->LSTM blend

# --- walk-forward / evaluation (mirror the sibling GBME battery) ---
HORIZONS = (1, 5, 10, 22)                       # forecast horizons (match the comparison table)
HORIZONS_SMOKE = (1,)                           # smoke: single horizon
MIN_ROWS = {"sp500": 30000, "default": 3000}    # min causal train rows per fold (mirrors the sibling gate)
PRED_CAP = 1.0                                  # variance ceiling (sigma=100%/day) numerical guard on exp(logvar)

# --- pre-registered kill criterion ---
GAIN_MIN = 0.0            # GBME+lstmfeat must beat GBME by strictly more than this QLIKE gain fraction
DM_ALPHA = 0.05           # date-clustered DM p-value must be below this
KILL_HORIZONS = (1, 5)    # BOTH must pass (gain>GAIN_MIN AND p<DM_ALPHA) for the direction to survive

# --- HOSE regime-spike robustness (project rule) ---
SPIKE_WINDOWS = (("2020-02-01", "2020-04-30"),   # COVID crash
                 ("2022-01-01", "2022-12-31"),   # 2022 VN drawdown
                 ("2025-04-01", "2025-04-30"))   # Apr-2025 tariff shock
