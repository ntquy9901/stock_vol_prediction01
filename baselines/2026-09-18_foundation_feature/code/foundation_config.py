"""Single source of truth for the foundation-model feature baseline (Hướng B falsification).

Every tunable constant lives here (not scattered as magic numbers in the pipeline) per CLAUDE.md
"Single-source-of-truth app config". Inline comments are intentional (they document intent and keep the
config-hardcode scanner's bare ``NAME = <num>$`` rule from firing on legitimate config constants).

Hướng B prior = NO-GO (Brini 2026: zero-shot TimesFM/Chronos lose to Log-HAR under QLIKE). This baseline tests
the WEAKER claim that the zero-shot forecast, added as ONE causal feature to the champion gamma-GBM (NOT as the
multiplicative anchor, which detonates QLIKE), can still beat the champion out-of-sample.
"""
# --- Zero-shot foundation forecaster (Amazon Chronos-Bolt: small, CPU-friendly, avoids the TiRex Windows blocker) ---
MODEL_NAME = "amazon/chronos-bolt-tiny"   # frozen pretrained model; learns nothing at inference (true zero-shot)
DEVICE = "cuda"                           # falls back to CPU in the forecaster if CUDA is unavailable
CTX_LEN = 256                             # trailing own-history context truncation (causal, <= t); ~1 trading year
PRED_LEN = 22                             # forecast steps per call (one call covers every horizon in HORIZONS)
QUANTILE_LEVELS = (0.1, 0.5, 0.9)         # low / median / high; median is the point forecast, hi-lo is the spread
FORECAST_BATCH = 512                      # sliding-window contexts per model call (GPU batch; not batch=1)
MIN_CONTEXT = 8                           # minimum trailing observations before a forecast row is emitted (causal)
USE_SPREAD = True                         # add the (q0.9 - q0.1) forecast spread as a second causal feature

# --- walk-forward / evaluation (mirror the sibling GBME battery) ---
HORIZONS = (1, 5, 10, 22)                        # forecast horizons (match the comparison table)
HORIZONS_SMOKE = (1,)                            # smoke: single horizon
MIN_ROWS = {"sp500": 30000, "default": 3000}     # min causal train rows per fold (mirrors the sibling gate)
PRED_CAP = 1.0                                   # variance ceiling (sigma=100%/day) numerical guard on any forecast

# --- placebo: wrong-ticker foundation forecast (destroys firm identity, keeps the marginal + date structure) ---
PLACEBO_MODE = "wrong_ticker"    # each ticker takes another ticker's foundation forecast aligned by trading date

# --- pre-registered kill criterion ---
GAIN_MIN = 0.0                   # GBME+FND must beat GBME by strictly more than this QLIKE gain fraction
DM_ALPHA = 0.05                  # date-clustered DM p-value must be below this
KILL_MIN_HORIZONS = 2            # GBME+FND must beat GBME (gain>GAIN_MIN AND p<DM_ALPHA) at >= this many horizons

# --- HOSE regime-spike robustness (project rule) ---
SPIKE_WINDOWS = (("2020-02-01", "2020-04-30"),   # COVID crash
                 ("2022-01-01", "2022-12-31"),   # 2022 VN drawdown
                 ("2025-04-01", "2025-04-30"))   # Apr-2025 tariff shock
