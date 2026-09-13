"""Constants for the principled-HAR feature baseline (option A: fixed Corsi windows).

See spec docs/superpowers/specs/2026-09-13-principled-har-features-design.md. No data-driven lag selection:
the HAR windows are the standard Corsi (2009) 1/5/22 (already precomputed as har_daily/weekly/monthly).
"""
HAR_WINDOWS = (1, 5, 22)     # Corsi (2009) daily/weekly/monthly (informational; features are precomputed cols)
SEMI_WINDOW = 5              # trailing window (days) for realized semivariance (weekly scale, matches har_weekly)
SEMI_MIN_PERIODS = 3         # min observations before a trailing semivariance value is defined
HORIZONS = (1, 5, 10, 22)   # forecast horizons
MIN_ROWS = {"sp500": 30000, "default": 3000}   # min causal train rows per fold (mirrors run_gbm's gate)
