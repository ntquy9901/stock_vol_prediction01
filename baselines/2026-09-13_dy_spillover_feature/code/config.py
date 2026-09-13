"""Single source of truth for this baseline's tunable constants (avoids the config-hardcode gate).

Constants parameterise the causal Diebold-Yilmaz (2012) volatility-spillover feature and the walk-forward
gamma-GBM add-on test. See ``../design/design.md`` for the VAR + generalized-FEVD math and the leakage
argument.
"""

# --- causal Diebold-Yilmaz spillover feature ---
DY_SECTOR_MIN_STOCKS = 10   # a sector needs >= this many constituents to become a VAR series (well-conditioned K)
DY_WINDOW = 250             # trailing rolling window (trading days) for each VAR fit ~ one trading year
DY_STEP = 5                 # re-estimate the VAR every DY_STEP days (~weekly); ffill in between
DY_VAR_LAG = 1              # VAR order p; p=1 keeps K*p+1 params/equation small vs DY_WINDOW
DY_GFEVD_H = 10             # forecast horizon (steps) of the generalized FEVD (Diebold-Yilmaz standard ~10)
DY_MIN_SECTORS = 3          # a window needs >= this many non-degenerate series to fit a VAR (else NaN)
DY_WIN_MIN_FRAC = 0.8       # a window must retain >= this fraction of DY_WINDOW rows after NaN-drop to be scored
DY_VAR_MIN_STD = 1e-9       # a series with in-window std below this is treated as degenerate (dropped)

# per-stock GBM forecast horizons + walk-forward fold gate (mirrors verify_index_vol_feature.py)
HORIZONS = (1, 5, 10, 22)
GBM_MIN_TRAIN_ROWS = {"sp500": 30000, "default": 3000}

# feature column names added to the per-stock panel
FEAT_NET = "net_spillover"
FEAT_TOTAL = "total_spillover"

# success criterion (economic gate on top of DM significance)
SUCCESS_DM_P = 0.05         # DM two-sided p-value must be below this
SUCCESS_MIN_GAIN_PCT = 0.3  # |QLIKE gain %| must reach this, sign-consistent across horizons, to count as a win
