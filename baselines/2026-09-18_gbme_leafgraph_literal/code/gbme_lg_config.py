"""Single source of truth for the LITERAL GBME + leaf-cooccurrence-graph baseline (Option A).

Every tunable constant lives here (not scattered as magic numbers in the pipeline) per CLAUDE.md
"Single-source-of-truth app config". Inline comments are intentional: they document intent and keep the
config-hardcode scanner's bare ``NAME = <num>`` rule from firing on legitimate config constants.

``GBME_PARAMS`` mirrors the champion gamma-GBME built inline in ``scripts/eda/full_matrix.py::gbm`` (300 trees,
lr 0.05, 31 leaves, l2=1, min_samples_leaf=20). A test (``test_fit_gbme_matches_full_matrix``) asserts our
fitted model reproduces ``full_matrix.gbm`` predictions exactly, so the two never silently drift.
"""
# --- champion gamma-GBME (HGBR) hyperparameters (mirror full_matrix.gbm) ---
GBME_PARAMS = {
    "loss": "gamma",            # gamma deviance (positive, multiplicative volatility target)
    "max_iter": 300,            # max boosting rounds (HGBR early_stopping='auto' may fit fewer)
    "learning_rate": 0.05,      # shrinkage
    "max_leaf_nodes": 31,       # leaves per tree (capacity control)
    "l2_regularization": 1.0,   # L2 penalty on leaf values
    "min_samples_leaf": 20,     # min samples per leaf (sklearn default; stated explicitly for the graph)
}

RECON_TOL = 1e-9                # max|baseline + sum(leaf value) - _raw_predict| tolerance (empirically ~1e-15);
                                # the fail-loud reconstruction guard raises above this (pins vs sklearn drift)

# --- leaf-cooccurrence graph + prediction smoothing ---
K_NEIGHBOURS = 10               # per-day kNN degree over leaf-Hamming similarity (matched to the S1 graph TOPK)
ALPHA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
              0.8, 0.9, 1.0)    # smoothing-weight grid; alpha fit on the val slice per fold (0 => graph inert)

# --- walk-forward / evaluation (mirror the sibling GBME battery) ---
VALID_LEN = 22                                  # trailing train dates held out as a true val slice (fit alpha)
HORIZONS = (1, 5, 10, 22)                       # forecast horizons (match the comparison table)
HORIZONS_SMOKE = (1,)                           # smoke: single horizon
MIN_ROWS = {"sp500": 30000, "default": 3000}    # min causal train rows per fold (mirrors the sibling gate)

# --- pre-registered kill criterion ---
GAIN_MIN = 0.0             # GBME+leafgraph must beat GBME by strictly more than this QLIKE gain fraction
DM_ALPHA = 0.05            # date-clustered DM p-value must be below this
KILL_HORIZONS = (1, 5)     # BOTH must pass (gain>GAIN_MIN AND p<DM_ALPHA) for pre-registered success

# --- HOSE regime-spike robustness (project rule) ---
SPIKE_WINDOWS = (("2020-02-01", "2020-04-30"),   # COVID crash
                 ("2022-01-01", "2022-12-31"),   # 2022 VN drawdown
                 ("2025-04-01", "2025-04-30"))   # Apr-2025 tariff shock
