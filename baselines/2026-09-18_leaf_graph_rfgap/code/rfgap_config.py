"""Single source of truth for the leaf-graph v2 (RF-GAP / KeRF) baseline.

Every tunable constant lives here (not scattered as magic numbers in the pipeline) per CLAUDE.md
"Single-source-of-truth app config". Inline comments are intentional: they document intent and keep the
config-hardcode scanner's bare ``NAME = <num>$`` rule from firing on legitimate config constants.

v2 upgrades the v1 leaf-Hamming kNN graph (baselines/2026-09-18_gbm_leaf_graph) with two literature-grounded
proximity weightings, selectable via ``WEIGHT_SCHEMES``:
  - ``knn``        : v1's hard top-k leaf-Hamming neighbour mean (the incumbent champion version).
  - ``rfgap``      : RF-GAP-style proper regression proximity (arXiv 2307.01077). Each tree contributes a
                     leave-one-out neighbour distribution normalised so every stock's neighbour weights sum to
                     1 and no stock smooths itself; the neighbour mean reconstructs the tree-ensemble's
                     leave-one-out same-day cross-section prediction.
  - ``rfgap_kerf`` : RF-GAP with KeRF large-leaf down-weighting (arXiv 2601.02735). Each tree's contribution is
                     scaled by ``KERF_FUNC(leaf_population)`` so a giant storm-day leaf (hundreds of trivially
                     "similar" stocks) injects less noise -> targets the mandated HOSE spike-robustness.

The XGBoost gamma capacity mirrors the champion HGBR gamma booster (300 trees, lr 0.05, ~31 leaves, l2=1,
min_child_weight ~ min_samples_leaf 20), matched to the GBME champion so the comparison is fair. XGBoost is used
(not HGBR) because it exposes per-tree leaf indices via ``bst.predict(dmatrix, pred_leaf=True)``.
"""
# --- XGBoost gamma booster (capacity mirrors the champion HGBR: 300 trees, lr 0.05, ~31 leaves, l2=1) ---
XGB_N_ESTIMATORS = 300      # boosting rounds = number of trees = leaf-vector length (match champion max_iter)
XGB_LR = 0.05               # learning rate (match champion learning_rate)
XGB_MAX_LEAVES = 31         # leaves per tree under lossguide (match champion max_leaf_nodes)
XGB_MAX_DEPTH = 0           # 0 = depth unbounded; leaf count is the capacity control under grow_policy=lossguide
XGB_L2 = 1.0                # L2 regularisation (match champion l2_regularization)
XGB_MIN_CHILD_WEIGHT = 20   # min hessian per leaf (analog of the champion HGBR min_samples_leaf=20; regularises)
PRED_CAP = 1.0              # variance ceiling (sigma=100%/day) — numerical guard so an XGB gamma exp-link on an
#                            extreme row cannot overflow to +inf; far above any real daily variance

# --- leaf-cooccurrence graph + prediction smoothing ---
K_NEIGHBOURS = 10           # per-day kNN degree for the 'knn' scheme (matched to the S1 graph TOPK=10)
WEIGHT_SCHEMES = ("knn", "rfgap", "rfgap_kerf")   # the three smoothing variants compared this run
KERF_FUNC = "inv"           # KeRF large-leaf down-weight g(pop): 'inv' => 1/pop, 'invsqrt' => 1/sqrt(pop)
ALPHA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
              0.8, 0.9, 1.0)   # smoothing weight grid; alpha fit per scheme on the val slice (0 => graph inert)

# --- walk-forward / evaluation (mirror the sibling GBME battery) ---
VALID_LEN = 22                                  # trailing train dates held out as a true val slice (fit alpha)
HORIZONS = (1, 5, 10, 22)                       # forecast horizons (match the comparison table)
HORIZONS_SMOKE = (1,)                           # smoke: single horizon
MIN_ROWS = {"sp500": 30000, "default": 3000}    # min causal train rows per fold (mirrors the sibling gate)

# --- pre-registered kill criterion ---
GAIN_MIN = 0.0             # a smoothed variant must beat its reference by strictly more than this QLIKE fraction
DM_ALPHA = 0.05            # date-clustered DM p-value must be below this
KILL_HORIZONS = (1, 5)     # BOTH must pass (gain>GAIN_MIN AND p<DM_ALPHA) for a variant to 'beat XGB'
V2_VARIANTS = ("XGB+rfgap", "XGB+rfgap+kerf")   # the v2 variants judged against the v1 'XGB+knn' champion

# --- HOSE regime-spike robustness (project rule) ---
SPIKE_WINDOWS = (("2020-02-01", "2020-04-30"),   # COVID crash
                 ("2022-01-01", "2022-12-31"),   # 2022 VN drawdown
                 ("2025-04-01", "2025-04-30"))   # Apr-2025 tariff shock
