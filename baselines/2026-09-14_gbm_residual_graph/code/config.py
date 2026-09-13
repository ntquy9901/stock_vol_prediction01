"""Single source of truth for this baseline's tunable constants (avoids the config-hardcode gate).

Residual-graph refiner: a linear (ridge) refiner trained on the GBM(own-8) OOS residual, using causal
neighbour-spillover features reused read-only from ``vn_gbm_graph_stage1`` (S1). See ``../design/design.md``
for the architecture and the causality/leakage argument. Only the tunables THIS baseline introduces live here.
"""
# walk-forward gamma-GBM base (mirrors full_compare / run_gbm)
HORIZONS = (1, 5, 10, 22)                       # forecast horizons
MIN_ROWS = {"sp500": 30000, "default": 3000}    # min causal train rows per fold

# ridge residual refiner
RIDGE_ALPHA = 1.0            # L2 strength (linear-first: keep the overfit surface small vs the prior tree blow-up)
MIN_STACK_ROWS = 500        # min pooled earlier-fold OOS residual rows before the refiner activates (else identity)
RESID_CLIP = 0.5            # bounded (robust) variant: winsorize the log-variance residual target + clip the
#                            prediction to +-RESID_CLIP, so exp(.) adjusts the GBM forecast within
#                            [exp(-0.5), exp(0.5)] = [0.61x, 1.65x]. Guards the QLIKE-catastrophic exp() blow-up
#                            that HOSE floor/limit-lock residuals cause in the raw (unclipped) refiner.

# fit-diagnostics verdict (overfit gate): test residual-MSE worse than train by more than this ratio -> overfit
OVERFIT_RATIO = 1.25

# pre-registered kill criterion (a big-n p-value alone is not a real win)
SUCCESS_DM_P = 0.05          # DM two-sided p-value must be below this
SUCCESS_HORIZONS = (1, 5)    # gain must be positive AND DM-significant at BOTH of these horizons to be a GO
