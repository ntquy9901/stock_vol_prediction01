"""Single source of truth for the OOF metric-constrained stacking baseline (Hướng 4).

Every tunable constant lives here (not scattered as magic numbers in the pipeline) per CLAUDE.md
"Single-source-of-truth app config". Inline comments are intentional: they document intent and keep
the config-hardcode scanner's bare ``NAME = <num>`` rule from firing on legitimate config constants.
"""
# --- OOF cross-fitting (leakage-safe meta-weight estimation) ---
INNER_K = 3                 # inner temporal K-fold blocks over the outer train window (mirror embed.oof_train_z)
VALID_LEN = 22              # trailing train dates held out as a true val slice (mirrors the sibling battery)

# --- level-2 meta-optimiser (non-negative simplex, QLIKE objective) ---
SLSQP_MAXITER = 200         # SLSQP iteration cap for the simplex QLIKE minimisation
SLSQP_FTOL = 1e-12          # SLSQP function tolerance (tight: QLIKE differences between bases are small)

# --- numerical guards ---
PRED_CAP = 1.0              # variance ceiling (sigma=100%/day) — clip so an XGB/GLM exp-link on an extreme
                            # z-scored row cannot overflow to +inf; far above any real daily variance
HAR_FLOOR_FRAC = 0.01       # HAR OLS positivity floor as a fraction of the mean train target (matches FM._har_ols)

# --- base-model capacity (XGB mirrors the champion HGBR: 300 trees, lr 0.05, ~31 leaves, l2=1) ---
XGB_N_ESTIMATORS = 300      # boosting rounds (match champion max_iter)
XGB_LR = 0.05               # learning rate (match champion learning_rate)
XGB_MAX_LEAVES = 31         # leaves per tree under lossguide (match champion max_leaf_nodes)
XGB_MAX_DEPTH = 0           # 0 = depth unbounded; leaf count is the capacity control under grow_policy=lossguide
XGB_L2 = 1.0                # L2 regularisation (match champion l2_regularization)
XGB_MIN_CHILD_WEIGHT = 20   # min hessian per leaf (analog of the champion HGBR min_samples_leaf=20; regularises)

# --- Gamma GLM (log-link) member ---
GLM_ALPHA = 1.0             # L2 penalty = sklearn GammaRegressor default (a lighter penalty under-regularises
                            # the z-scored coefficients -> over-dispersion -> negative MSE-R2 that trips the gate)
GLM_MAX_ITER = 1000         # IRLS/solver iterations for the GLM

# --- walk-forward / evaluation (mirror the sibling GBME battery) ---
HORIZONS = (1, 5, 10, 22)                       # forecast horizons (match the comparison table)
HORIZONS_SMOKE = (1,)                           # smoke: single horizon
MIN_ROWS = {"sp500": 30000, "default": 3000}    # min causal train rows per fold (mirrors the sibling gate)

# --- pre-registered kill criterion ---
GAIN_MIN = 0.0             # stack must beat the best single base by strictly more than this QLIKE gain fraction
DM_ALPHA = 0.05            # date-clustered DM p-value must be below this
KILL_HORIZONS = (1, 5)     # BOTH must pass (gain>GAIN_MIN AND p<DM_ALPHA) for GO; else NO-GO

# --- HOSE regime-spike robustness (project rule) ---
SPIKE_WINDOWS = (("2020-02-01", "2020-04-30"),   # COVID crash
                 ("2022-01-01", "2022-12-31"),   # 2022 VN drawdown
                 ("2025-04-01", "2025-04-30"))   # Apr-2025 tariff shock
