# Hướng 4 — OOF metric-constrained stacking (HOSE)

## Objective
Test whether a 2-level **out-of-fold (OOF) metric-constrained stack** of diverse own-history base
models beats the best single base model on out-of-sample QLIKE for HOSE volatility forecasting.

The level-2 meta-learner picks **non-negative simplex weights** (`w_m >= 0`, `sum w_m = 1`) that
minimise **pooled QLIKE** on leakage-safe OOF base predictions, then applies the frozen weights to
test-window base predictions.

## Pre-registered hypothesis (expected NO-GO)
Base models trained on own history are highly correlated, so the constrained stack is expected to
**collapse onto the single best base model** (one weight ~1, the rest ~0) and provide no OOS gain.
The deliverable is a clean, gate-clean **falsification** — not a positive result. Report honestly.

## Inputs
- HOSE enriched panel: `data/processed_enriched/hose/*.csv` (via `full_matrix.load`).
- Real crawled VN earnings dates: `results/gamma_gbm/hose_earnings_combined.parquet` (402 tickers).
- Features: OWN-8 (single-sourced from `baselines/2026-09-13_paper_models/code/config.py::own_set`,
  = `FM.OWN` minus `rq`) + `FM.EARN` (4 earnings-proximity features).
- Target: `parkinson_variance.shift(-h)`, floored at `FM.FL = 1e-8` for the gamma members.

## Base models (diversity by construction)
| name | learner | features | loss |
|------|---------|----------|------|
| GBME | `HistGradientBoostingRegressor(loss="gamma", …)` (champion, `FM.gbm`) | OWN-8+EARN | gamma |
| XGB  | `xgboost reg:gamma` (champion-matched capacity) | OWN-8+EARN | gamma |
| GLM  | `sklearn GammaRegressor(alpha=1.0)` log-link | OWN-8+EARN | gamma |
| HAR  | OLS on `har_daily/weekly/monthly` | HAR-3 | MSE (deliberately different) |

## Meta-learner
Minimise pooled QLIKE (champion floor `FM.FL`) over the non-negative simplex with
`scipy.optimize.minimize` (SLSQP, `sum w = 1`, bounds `[0,1]`, uniform start). Report learned weights
per horizon (whether they collapse onto one base).

## OOF generation (leakage-safe)
Within each outer walk-forward train window, an inner **temporal K-fold (K=3)**: every train row is
predicted by a base model that did NOT train on it (mirrors
`baselines/2026-09-14_gbm_gnn_embed/code/embed.py::oof_train_z`). Test predictions come from base
models trained on the full outer train window (minus the trailing validation slice).

## Evaluation
- Walk-forward over `vn_gbm_graph_stage1.FOLDS` from `TRAIN_START`, embargo `int(1.6h)+5` days.
- 5 metrics (mse/rmse/mae/r2/qlike) train/val/test for every model + `fit_diagnostics`.
- Date-clustered Diebold-Mariano: **stack vs best-single base** (primary) and **stack vs GBME**.
- HOSE regime-spike robustness (COVID-2020 / 2022 / Apr-2025): re-evaluate excluding the spike windows.
- Per-fold QLIKE.

## Success / kill criterion (pre-registered)
Stack **beats the best single base** at BOTH h1 and h5: DM `p < 0.05` AND QLIKE gain `> 0` AND the
sign survives spike-exclusion. Otherwise → **NO-GO** (the expected outcome).

## Go / No-Go
- GO: kill criterion met (stack materially and significantly beats the best single base at h1 & h5).
- NO-GO: otherwise. Document the collapse (weights on one base) as the falsification finding.

## Out of scope
- SP500 (driver supports it for parity but the HOSE run is the deliverable).
- Any non-own-history / graph member (exhausted per project memory).
