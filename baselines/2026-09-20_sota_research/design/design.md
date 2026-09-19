# Design — overnight SOTA experiments (XAI + conformal)

## Shared harness (leak-safe, reuses VolTree)
Both experiments import the VolTree runner (`baselines/2026-09-18_leaf_graph_paper/code/
run_leaf_graph_paper.py`) for the exact pipeline: `FM.load` / `FM.panel` (OWN-8 + EARN-4 features,
target `y` = Parkinson variance), `S1.FOLDS` (8 semi-annual walk-forward folds from 2015-01-01),
horizon-scaled embargo, `VALID_LEN=22` validation tail, and `LG.fit_booster` (reg:gamma, 300 rounds,
31 leaves, mcw 20). No numbers/config are re-derived — single-sourced from the sibling config.

Leak-safety (identical to VolTree's α-on-validation): every fold fits on TRAIN minus the last 22 dates;
any calibration/selection uses only that held-out VALIDATION slice; the frozen artifact is applied to the
test window. Test never informs any choice.

## Exp 1 — TreeSHAP explainability (`run_xai.py`)
- **Method:** exact TreeSHAP via XGBoost native `predict(pred_contribs=True)` (avoids the shap-lib
  base_score bug on XGBoost 3.x). Attribution is in the gamma **log-margin** space (log link) — stated,
  per the standard rigor caveat for gamma/Tweedie SHAP. Global importance = mean|SHAP| over test rows,
  aggregated across folds (weighted by n), per (market, horizon). 6k test rows sampled per fold (global
  mean|SHAP| is stable; exact SHAP is O(n)).
- **Outputs:** per-feature share, signed mean, ranking, and HAR/momentum/earnings group shares.
- **Why:** a rigorous, model-faithful alternative to permutation importance (which hit a documented
  placebo pitfall in this project). Validates the paper's "HAR lags dominate" + earnings/leaf-graph roles.

## Exp 2 — Conformal prediction intervals (`run_conformal.py`)
- **Split conformal** (baseline): Q = (1-α) quantile of |y-μ| on the validation slice; band μ±Q, floored.
- **CQR** (Romano et al. 2019): two XGBoost quantile heads (reg:quantileerror @ 0.05/0.95); conformity
  E = max(lo-y, y-hi) on validation; band [lo-Q, hi+Q]. Adaptive width for heteroscedastic volatility.
- **Metrics:** marginal coverage (target 0.90), mean/median interval width, and coverage inside vs
  outside the pre-specified spike windows (COVID/2022/Apr-2025) — the HOSE robustness mandate applied to
  intervals. Demo horizons {1,5}, quantile heads 200 rounds (endpoints), both markets.
- **Why:** point-QLIKE is saturated; calibrated intervals are a NEW, publishable axis with a
  finite-sample coverage guarantee.

## Go / no-go + honesty
- A direction is "GO" only with the mandated evidence: leak-safe protocol, per-regime robustness, and an
  honest record of nulls (no p-hacking). NO-GO directions (deep/graph/foundation standalone; univariate
  SSM/KAN; rough-vol own-path expected null) are recorded, not hidden.
- Deferred (build/GPU/API cost, not run tonight): iTransformer+SOFTS on the residual panel; LLM
  feature-discovery agent; NGBoost/XGBoostLSS distributional baseline; adaptive conformal (ACI/PID).

## Quality gates
- Pure helpers (shap_global, cqr_band, split_band, _spike_mask, _feature_group) unit-tested
  (`test/test_experiments.py`, 6 tests). Data-driven drivers are `# pragma: no cover`.
- post-gen ruff-F clean; pre-push diff-cover on changed pure lines. Result JSONs are experiment outputs
  (not training-result JSONs subject to overfit-evidence gate — these are explainability/interval
  diagnostics, not model-selection metrics).

## Deliverables
- `results/gamma_gbm/xai_shap_<market>.json`, `conformal_<market>.json`
- `docs/reports/2026-09-20_overnight_sota/report.html` (build_report.py) + `research_synthesis.md`
- roadmap (`requirements/requirements.md`), this design, memory update.
