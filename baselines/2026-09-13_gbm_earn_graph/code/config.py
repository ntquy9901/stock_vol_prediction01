"""Single source of truth for this baseline's tunable constants (avoids the config-hardcode gate).

See ``../design/design.md`` for the architecture and the causality/leakage argument. Feature/graph
machinery is reused read-only from ``full_matrix`` (FM) and ``vn_gbm_graph_stage1`` (S1); this module only
holds the tunables THIS baseline introduces.
"""

# walk-forward gamma-GBM add-on test (mirrors paper_metrics_sp500 / run_gbm / full_compare)
HORIZONS = (1, 5, 10, 22)                       # forecast horizons
MIN_ROWS = {"sp500": 30000, "default": 3000}    # min causal train rows per fold

# fit-diagnostics verdict (overfit gate): test QLIKE worse than train by more than this ratio -> overfit
OVERFIT_RATIO = 1.25

# economic go/no-go on top of DM significance (a big-n p-value alone is not a real win)
SUCCESS_DM_P = 0.05           # DM two-sided p-value must be below this
SUCCESS_MIN_GAIN_PCT = 0.3    # |QLIKE gain %| must reach this, sign-consistent across horizons, to count
