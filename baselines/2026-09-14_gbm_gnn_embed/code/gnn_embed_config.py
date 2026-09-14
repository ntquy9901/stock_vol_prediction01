"""Single source of truth for the GBME+GNN-embed baseline's tunable constants.

Every constant lives here (not scattered as magic numbers in the pipeline) per CLAUDE.md
"Single-source-of-truth app config". Optimizer hyper-parameters (LR, weight decay, batch size, the
embedding dim N_HID, the seed ensemble, the trailing-validation length) are NOT duplicated here: they are
reused from the faithful GNNHAR module (`gnnhar_sp500`) so there is exactly ONE source for them. See
design/design.md. Inline comments are intentional (they also keep the config-hardcode scanner's bare
`NAME = <num>$` rule from firing on legitimate config constants).
"""
N_GCN = 2                 # GCN layers in the embedding GNN (design: 2-layer message passing)
INNER_K = 3               # temporal inner folds for out-of-fold z_train cross-fitting (no stacking leakage)
ARCH = "gcn"              # embedding GNN architecture: 'gcn' (implemented) | 'gat' (reserved, raises)
EPOCHS = 150              # max GNN epochs (full run; early-stop on inner-val QLIKE)
EPOCHS_SMOKE = 40         # max GNN epochs (smoke / test)
PATIENCE = 20             # early-stop patience on inner-val QLIKE
MIN_EPOCH = 20            # min epochs before early stop may fire (matches GNNHAR's guard)
VALID_LEN = 22            # trailing inner-train dates held out as the GNN early-stop validation slice
HORIZONS = (1, 5, 10, 22)                       # forecast horizons (match the sibling comparison table)
MIN_ROWS = {"sp500": 30000, "default": 3000}    # min causal train rows per fold (mirrors the sibling gate)

# --- pre-registered kill criterion (design section 6) ---
GAIN_MIN = 0.0            # GBME+z must beat GBME by strictly more than this QLIKE gain fraction
DM_ALPHA = 0.05           # date-clustered DM p-value must be below this
KILL_HORIZONS = (1, 5)    # BOTH of these horizons must pass (gain>GAIN_MIN AND p<DM_ALPHA) for success

# --- HOSE regime-spike robustness (design section 6 / project rule) ---
SPIKE_WINDOWS = (("2020-02-01", "2020-04-30"),   # COVID crash
                 ("2022-01-01", "2022-12-31"),   # 2022 VN drawdown
                 ("2025-04-01", "2025-04-30"))   # Apr-2025 tariff shock
