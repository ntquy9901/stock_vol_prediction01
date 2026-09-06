# Requirements — HAR-X + XGBoost residual-ratio volatility study

Spec for the study defined by `docs/experement_guide/AI_MASTER_INSTRUCTION_XGBOOST_EXPERIMENTS.md`
(authoritative). This file is the Specify artifact (CLAUDE.md §5 SDD).

## Objective

Determine whether a nonlinear tabular model — primarily **HAR-X + XGBoost residual-ratio** — and a
**direct XGBoost** improve out-of-sample **Parkinson VARIANCE** QLIKE over HAR-X for VN30/VN100 at
horizons `h ∈ {1,5,10,22}`. A negative result is valid and must not be narrated away.

## Target (audited)

- Forecast target = `parkinson_variance` column at `t+h` (a VARIANCE σ², confirmed in the enriched CSV
  schema audit — see design.md; matches project memory "Parkinson target is variance").
- Shared QLIKE positivity floor = `training_config().qlike_floor` (= `pc.QLIKE_FLOOR` = 1e-8), plus the
  shared per-node positivity floor `nfloor = pc.POS_FLOOR_FRAC*t_mean + pc.POS_FLOOR_EPS` applied to every
  model's forecast identically (fairness gate).

## Inputs

- Enriched panels `data/processed_enriched/{vn30,vn100}/<ticker>.csv` (columns include
  `parkinson_variance, har_weekly, har_monthly, market_pk, volume_zscore_22, daily_return, log_range,
  volume, zero_range_flag`).
- Canonical walk-forward split: `lb10, folds_target=7` (matches the delivered `edge_hmatched` runs, so
  HAR-X / VolGA / LSTM pooled numbers are directly comparable). Expanding-window, calendar-date isolated,
  purge = horizon, validation tail selects all hyperparameters, test scored once.

## Models (ladder)

1. HAR (reference, OLS)                    — reused `_har_ols_preds`
2. HAR-X (reference, OLS 5-feat)           — reused `_har_ols_preds`
3. Direct XGBoost on log target
4. HAR-X + XGBoost residual-ratio (base = stock features)  — **PRIMARY**
5. + market-wide features
6. + graph-derived features (horizon-matched `directed_vol2pk_hmatched`, train-only adjacency)

CatBoost/LightGBM: optional, only if the ladder works (not installed → out of scope, noted).
LSTM / VolGA: reference numbers cited from delivered `results/edge_hmatched/*.json` (GPU models NOT
retrained — CPU-only mandate; the GPU is in use by a concurrent agent).

## Success criteria (acceptance) — GO for HAR-X+XGB residual

GO only if MOST hold (guide Stage 12):
- lower standard QLIKE than HAR-X on >1 panel/horizon;
- improvement persists on non-lock ticker-date observations;
- not driven solely by April-2025 / top-1% dates;
- ≥1 predeclared primary comparison statistically significant (date-clustered DM p<0.05) OR a stable,
  practically meaningful cross-fold effect;
- calm-regime performance not materially degraded;
- reproducible (fixed seed, deterministic tree model).

NO-GO if: validation selects no correction (alpha=0); gains vanish outside lock/shock; gains depend on
post-hoc choices; residual model overfits and harms OOS QLIKE; effect inconsistent/negligible; leakage
or unfair comparison cannot be ruled out.

## Non-goals / constraints

- CPU-ONLY (`tree_method='hist'`, bounded `n_jobs`). No torch/GPU.
- Hard isolation (§3.F): all new code under this baseline dir; delivered panel/fold/HAR-X/metric/DM
  machinery imported READ-ONLY; no other baseline or src file modified; no existing result overwritten.
- Every feature at forecast origin `t` uses information ≤ `t` (causal). Preprocessing / HAR-X coefficients
  / OOF construction / hyperparameters / alpha-clip guardrail fit on train+val only; test never used for
  selection.
- Never call ticker-date observations "days"; report ticker-date count AND unique-date count.
- Do not commit — leave staged-clean for the coordinator.

## Go/No-Go gate for the task itself

Smoke on one panel/horizon must pass before the full VN30×VN100 × h{1,5,10,22} matrix. Tests
(causality/OOF-cutoff/graph-cutoff/split-isolation/count/reproducibility) must pass. Overfit evidence
(train/val/test fit metrics + `fit_diagnostics`) written for the XGBoost learner so the pre-push gate
accepts the result JSONs.
