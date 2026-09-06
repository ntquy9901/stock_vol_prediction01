# Requirements — Limit-lock hurdle on HAR-X

## Problem / motivation
On a limit-lock (price-limit) day the intraday high-low range collapses, so the Parkinson
**variance** target is ~0 (floored). Models that see recent high volatility keep forecasting a
moderate variance `f`. QLIKE = `y/f - log(y/f) - 1` with `y -> 0` and moderate `f` produces a large
`-log(y/f)` term, so a handful of lock days dominate the pooled QLIKE. Concrete example: VN30
**2025-04-10**, where the market limited up after a limit-down streak (2025-04-03/04-08 both near
the -7% HOSE limit) and Parkinson variance collapsed to 0 for the affected stocks (e.g. FPT
`parkinson_variance=0.0`, `zero_range_flag=True`).

## Hypothesis (two-part / hurdle)
Predict `P(target day is a limit-lock / zero-range day)` from **causal** limit-lock features, and on
high-probability-lock days pull the HAR-X variance forecast DOWN to the positivity floor so its QLIKE
contribution cannot explode. Probe on the QLIKE champion HAR-X (OLS, CPU) first — before any deep
model — to isolate whether the hurdle idea works at all.

## Scope
- CPU-only, no torch/GPU. Variance forecaster = HAR-X OLS (reuse delivered panel/folds read-only).
  Hurdle classifier = `sklearn.linear_model.LogisticRegression` (CPU, deterministic lbfgs).
- Markets: VN30 (primary — limit-lock happens here), VN100. Horizons h in {1,5,10,22}.
- Compare plain HAR-X vs HAR-X + limit-lock hurdle at several override thresholds.

## Inputs
- `data/processed_enriched/<market>/<TIC>.csv` columns used: the 5 enriched node features (via the
  delivered `wf_enriched_panel`), plus `daily_return` and `zero_range_flag` (read here, aligned to the
  panel dates/tickers) for the causal limit-lock features and the classifier target.

## Outputs
- Per market/horizon result JSON `results/limitlock_hurdle/hurdle_<market>_h<h>.json` with:
  metrics (MSE/RMSE/MAE/QLIKE/R2 + `qlike_robust` excluding lock target days + `qlike_lockdays`) for
  HAR-X and each HAR-X+hurdle@threshold; date-clustered DM (hurdle vs HAR-X) per loss family;
  classifier confusion (TP/FP/FN/TN) on test lock days; the 2025-04-10 concrete case (if in test).

## Acceptance criteria (verifiable)
1. All feature functions are **causal** (prefix-invariance test: features on days 0..k unchanged when
   future days appended).
2. `pytest baselines/2026-09-06_limitlock_hurdle/test/ -v` passes; C0=100% on changed non-`pragma`
   lines.
3. Same `qlike_floor` and per-node positivity floor (`nfloor`) applied identically to both compared
   models (H2 lesson).
4. Result JSON written for VN30 (all h) and VN100 (all h).

## Go / No-Go
- **GO** if HAR-X+hurdle LOWERS pooled QLIKE vs plain HAR-X on VN30 (especially short h where locks
  cluster), the classifier catches real lock days, and false positives do not erase the gain (i.e.
  `qlike_robust`, the non-lock-day loss, is not materially worse).
- **NO-GO** (a valid, useful result) if the classifier cannot predict locks, or false positives on
  normal days cost more than the lock-day fix saves. Report honestly either way.
