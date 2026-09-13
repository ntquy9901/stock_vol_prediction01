# Design — Principled HAR-family feature set (option A)

Canonical design: `docs/superpowers/specs/2026-09-13-principled-har-features-design.md`. This file records the
concrete implementation.

## Data flow
`FM.load(market)` → per-ticker enriched frames (already carry HAR trio + GK/RS/YZ variances + daily_return) →
`build_panel.feature_frames` adds `semi_neg`/`semi_pos` (causal, per-ticker) → `FM.panel(_, {}, h)` builds the
pooled panel with `y = parkinson_variance.shift(-h)` → `run_har.run` does the walk-forward GBM comparison.

## Modules (code/)
| File | Responsibility |
|---|---|
| `config.py` | Constants: `HAR_WINDOWS=(1,5,22)` (informational), `SEMI_WINDOW=5`, `SEMI_MIN_PERIODS=3`, `HORIZONS`, `MIN_ROWS`. |
| `estimators.py` | `semivariance(ret, window, min_periods)` — daily-frequency realized semivariance (the only new feature). GK/RS/YZ/Parkinson reused from enriched columns (already formula-tested in `2026-08-31_enriched_processed`). |
| `build_panel.py` | `FEATURES` (the 8 principled features); `add_features` (adds semivariance); `feature_frames`. |
| `run_har.py` | `run(market)`: GBM(principled) vs GBM(FM.OWN) walk-forward, pooled QLIKE, date-clustered DM, leave-one-out, fit diagnostics. Fixed windows; reuses `FM.gbm/panel/SEEDS/OWN`, `S1.FOLDS/TRAIN_START`, `M.per_obs_qlike`, `ST.date_clustered_dm`. |

## Key decisions
- **Fixed Corsi (1,5,22)** — no GPH/AIC lag selection (option A). Rationale + the documented-but-not-adopted
  data-driven alternative: `docs/paper/2026-09-13_har_lag_selection_methodology.md`.
- **Reuse precomputed estimator columns** — GK/RS/YZ are already in the enriched frames (both markets, after
  the 2026-09-13 OHLC reprocess); recomputing would duplicate already-tested code.
- **Leverage via realized semivariance** replaces the ad-hoc `mr_*` momentum features with a cited signal.
- **Non-inferiority** success criterion + **leave-one-out** pruning; DM (not univariate MI) is the arbiter.

## Gates (Simplicity / Anti-Abstraction / Performance)
- Simplicity: 3 small modules, no new abstraction; reuses the sibling GBM/DM harness verbatim.
- Anti-Abstraction: uses `FM.gbm`/`FM.panel` directly; no wrapper.
- Performance: seed-averaged HistGBM (OpenMP multicore) per fold; batched panel; no per-row Python loop in the
  hot path. Leave-one-out is 8×horizons extra fits (documented cost); SP500 runs on Colab.
