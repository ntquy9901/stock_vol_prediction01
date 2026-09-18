"""Single source of truth for the DTW self-similarity feature baseline.

Every tunable constant lives here (not scattered as magic numbers in the pipeline) per CLAUDE.md
"Single-source-of-truth app config". Inline comments are intentional: they document intent and keep the
config-hardcode scanner's bare ``NAME = <num>$`` rule from firing on legitimate config constants.
"""
# --- DTW feature construction ---
DTW_WINDOW = 22             # trailing-window length W (days) of log-variance matched by DTW (monthly path)
DTW_BAND = 4                # Sakoe-Chiba band radius (cells) constraining the DTW warping path
DTW_BANDS = ((0.0, 1.0 / 3.0),   # low vol-level archetype  (quantile band of train-window mean level)
             (1.0 / 3.0, 2.0 / 3.0),  # mid vol-level archetype
             (2.0 / 3.0, 1.0))        # high vol-level archetype  -> one template (=> one DTW feature) per band
DTW_PLACEBO_SHIFT = 126     # placebo query window ends this many business days earlier (~6mo, still causal)
DTW_MIN_TRAIN_WINDOWS = 30  # min train windows required to form templates for a ticker (else NaN feature)

# --- walk-forward / evaluation (mirror the sibling GBME battery) ---
VALID_LEN = 22              # trailing train dates held out as a true val slice (mirrors the sibling battery)
HORIZONS = (1, 5, 10, 22)   # forecast horizons (match the comparison table)
HORIZONS_SMOKE = (1,)       # smoke: single horizon
MIN_ROWS = {"sp500": 30000, "default": 3000}   # min causal train rows per fold (mirrors the sibling gate)
PRED_CAP = 1.0             # variance ceiling (sigma=100%/day) numerical guard on the floored prediction

# --- pre-registered kill criterion ---
GAIN_MIN = 0.0             # GBME+dtw must beat GBME by strictly more than this QLIKE gain fraction
DM_ALPHA = 0.05            # date-clustered DM p-value must be below this
KILL_MIN_HORIZONS = 2      # GBME+dtw must beat GBME (spike-robust, placebo-clean) at >= this many horizons

# --- HOSE regime-spike robustness (project rule) ---
SPIKE_WINDOWS = (("2020-02-01", "2020-04-30"),   # COVID crash
                 ("2022-01-01", "2022-12-31"),   # 2022 VN drawdown
                 ("2025-04-01", "2025-04-30"))   # Apr-2025 tariff shock
