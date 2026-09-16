"""Single-source tunables for the equity conditional-covariance forecasting baseline.

Every adjustable constant (basket size, estimation window, rebalance horizons, EWMA decay, ridge, winsor
band, spike windows) lives here and is imported — no magic numbers scattered in pipeline modules (per the
project single-source-of-truth config rule).
"""
from __future__ import annotations

# ---- data / panel ----
N_BASKET = 40          # number of most-liquid, full-history stocks in the basket (N < W for invertibility)
TRAIN_START = "2015-01-01"    # panel start (mirrors vn_gbm_graph_stage1.TRAIN_START)
EVAL_START = "2022-07-01"     # first rebalance date (mirrors FOLDS[0]); OOS evaluation region
WINDOW = 252           # rolling estimation window (trading days) used to estimate each Sigma_hat_t
MIN_COVERAGE = 0.95    # a ticker must have >= this fraction of trading days over the panel to be eligible

# per-market return winsor band: cap |daily_return| to kill rare unadjusted-split jumps before covariance.
# HOSE has a hard +/-7% daily price limit, so any |r| beyond it is a corporate-action artifact; SP500 uses a
# looser robust cap (adjusted data, real crashes can exceed 7%).
WINSOR_BAND = {"hose": 0.07, "sp500": 0.25, "default": 0.25}

# ---- estimators ----
EWMA_LAMBDA = 0.94     # RiskMetrics decay for the EWMA covariance
RIDGE_EPS = 1e-10      # relative ridge added to the diagonal (eps * mean(diag)) for invertibility
FACTOR_K = 3           # number of principal factors for the rank-k (POET/PCA) factor covariance

# ---- evaluation ----
REBALANCE_HORIZONS = (5, 10, 22)   # hold periods h (trading days) between GMV rebalances
ANNUALIZE = 252        # trading days per year for annualizing portfolio volatility

# ---- robustness (per HOSE spike rule) ----
SPIKE_WINDOWS = (("2020-02-01", "2020-04-30"),   # COVID crash
                 ("2022-01-01", "2022-12-31"),   # 2022 VN drawdown
                 ("2025-04-01", "2025-04-30"))   # Apr-2025 tariff shock

# ---- go/no-go ----
DM_ALPHA = 0.05        # DM significance threshold for "beats"
