# Design — Gamma-GLM base-margin anchor + GBDT residual (Hướng 3)

Date: 2026-09-17.

## 1. Two-stage model
**Stage 1 — Gamma GLM (log-link, IRLS on gamma deviance).** `sklearn.linear_model.GammaRegressor` on
z-scored (train-fit scaler) OWN-8 (+EARN) features, target floored at `FL`. Its linear predictor in log space is
`η = log(μ)`, recovered as `η = log(max(glm.predict(X), FL))` (log-link ⇒ μ = exp(η)). η is loss-consistent
(gamma deviance = QLIKE up to constants) so it does NOT bias calm-day QLIKE, and being linear it **extrapolates**
beyond the train range where trees saturate.

**Stage 2 — XGBoost gamma residual with base margin.** `xgboost.train(objective="reg:gamma")` with
`DMatrix.set_base_margin(η)` on BOTH train and test. XGBoost boosts the residual over the fixed margin, so
`ŷ = exp(η_GLM + Σ f_tree(X))`. Capacity mirrors the champion (300 rounds, lr 0.05, ≤31 leaves, λ=1).

## 2. Fair comparison (isolate the anchor)
Both the residual model and its control are **XGBoost gamma** — `GLM+XGB` (with base margin) vs `XGB` (no base
margin) isolates the base-margin contribution free of any library difference. `GBME` (the canonical HGBR gamma
champion) is also run as the real target-to-beat and a validity check that `XGB ≈ GBME`.

## 3. Walk-forward, causality, evidence (mirror the sibling GBME battery)
Per horizon h and outer fold k (`S1.FOLDS`, `S1.TRAIN_START`, embargo `int(1.6h)+5` days):
- `trf` = causal train window (`< fold_start − embargo`), `tef` = test window; skip if `len(trf) < MIN_ROWS`.
- Hold out the last `VALID_LEN`-style val slice of `trf` → fit on `trf_e = trf∖val`, predict
  `combo = [tef, trf_e, vaf]` so test/train/val metrics are genuine (val is a true hold-out for fit_diagnostics).
- **Every fit (GLM scaler + coefficients, XGB trees) uses `trf_e` rows only** → strictly causal; a fold's fit
  never sees any test row (verified by a prefix-mutation test).
- Seed-ensemble XGB over `FM.SEEDS`; GLM is deterministic.
- Pool test predictions over folds → 5-metric train/val/test per model + `fit_diagnostics` (via
  `overfit_check.classify_fit`), **date-clustered DM** (`stats.date_clustered_dm`) GLM+XGB vs GBME and vs XGB,
  QLIKE gain, verdict. HOSE also: per-fold QLIKE + **regime-spike robustness** (rerun excluding COVID/2022/
  Apr-2025; sign must hold).

## 4. Gates (§5 SDD)
- **Simplicity:** reuse `full_matrix` (panel/load/gbm/floor/seeds), `vn_gbm_graph_stage1` (folds), `metrics`,
  `stats`, `overfit_check`; no new abstractions. OWN-8 single-sourced from the paper_models config by path
  (no second bare `config` module).
- **Anti-abstraction:** use `xgboost`/`sklearn` directly.
- **Performance/batching:** XGBoost `tree_method="hist"` (multithreaded, vectorised); GLM IRLS is a closed-form
  solver. No per-row Python loops in the hot path; per-fold matrices are dense NumPy. No batch=1 anti-pattern.

## 5. Config (single source: `code/config.py`)
XGB (n_estimators/lr/max_leaves/max_depth/λ), GLM (alpha/max_iter), floor via `FM.FL`, HORIZONS, MIN_ROWS,
GAIN_MIN, DM_ALPHA, KILL_HORIZONS, SPIKE_WINDOWS. No magic numbers in the pipeline modules.

## 6. Files
`code/{config.py, glm_anchor.py (fit_glm_eta, xgb_gamma), run_glm_anchor.py (run/verdict/_pool_doc/_spike_mask/
_checkpoint/main)}`, `test/test_glm_anchor.py`.
