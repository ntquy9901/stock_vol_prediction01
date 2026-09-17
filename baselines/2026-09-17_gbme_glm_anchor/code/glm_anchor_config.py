"""Single source of truth for the GLM-base-margin anchor baseline (Hướng 3).

Every tunable constant lives here (not scattered as magic numbers in the pipeline) per CLAUDE.md
"Single-source-of-truth app config". Inline comments are intentional (they document intent and keep the
config-hardcode scanner's bare ``NAME = <num>$`` rule from firing on legitimate config constants).
"""
# --- Stage-2 XGBoost gamma booster (capacity mirrors the champion HGBR: 300 trees, lr 0.05, ~31 leaves, l2=1) ---
XGB_N_ESTIMATORS = 300      # boosting rounds (match champion max_iter)
XGB_LR = 0.05               # learning rate (match champion learning_rate)
XGB_MAX_LEAVES = 31         # leaves per tree under lossguide (match champion max_leaf_nodes)
XGB_MAX_DEPTH = 0           # 0 = depth unbounded; leaf count is the capacity control under grow_policy=lossguide
XGB_L2 = 1.0                # L2 regularisation (match champion l2_regularization)
XGB_MIN_CHILD_WEIGHT = 20   # min hessian per leaf (analog of the champion HGBR min_samples_leaf=20; regularises)

# --- Stage-1 Gamma GLM (log-link, IRLS on the gamma deviance = QLIKE up to constants) ---
GLM_ALPHA = 1.0            # L2 penalty for the gamma GLM = sklearn GammaRegressor default (a lighter penalty
                          # under-regularises z-scored coefficients -> eta over-disperses -> negative MSE-R2)
GLM_MAX_ITER = 1000         # IRLS/solver iterations for the GLM
PRED_CAP = 1.0             # variance ceiling (sigma=100%/day) — numerical guard so a GLM/XGB exp-link on an
                           # extreme z-scored row cannot overflow to +inf; far above any real daily variance

# --- walk-forward / evaluation (mirror the sibling GBME battery) ---
HORIZONS = (1, 5, 10, 22)                       # forecast horizons (match the comparison table)
HORIZONS_SMOKE = (1,)                           # smoke: single horizon
MIN_ROWS = {"sp500": 30000, "default": 3000}    # min causal train rows per fold (mirrors the sibling gate)

# --- pre-registered kill criterion ---
GAIN_MIN = 0.0             # anchored must beat the baseline by strictly more than this QLIKE gain fraction
DM_ALPHA = 0.05            # date-clustered DM p-value must be below this
KILL_HORIZONS = (1, 5)     # BOTH must pass (gain>GAIN_MIN AND p<DM_ALPHA) for success

# --- HOSE regime-spike robustness (project rule) ---
SPIKE_WINDOWS = (("2020-02-01", "2020-04-30"),   # COVID crash
                 ("2022-01-01", "2022-12-31"),   # 2022 VN drawdown
                 ("2025-04-01", "2025-04-30"))   # Apr-2025 tariff shock
