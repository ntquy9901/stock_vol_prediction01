# Code review 2026-09-17 — Gamma-GLM base-margin anchor + GBDT residual (Hướng 3)

Adversarial review of `glm_anchor.py` + `run_glm_anchor.py` (leakage, loss consistency, numerical stability,
comparison fairness, config hygiene, positivity floor). Findings found and fixed during the build.

| # | Severity | Finding | Disposition |
|---|---|---|---|
| 1 | Major | `import config` resolved to `submission/soict_lstm_gat/config.py` (bare `config` collides across sys.path) → `AttributeError: HORIZONS_SMOKE`. | **Fixed:** renamed `config.py` → `glm_anchor_config.py` (unique name, repo convention `gnn_embed_config`/`cov_config`). Root lesson: baselines must not use a bare `config` module. |
| 2 | Major | On real HOSE data the Gamma GLM / XGBoost gamma exp-link overflowed to `+inf` on extreme z-scored rows → `_metrics5` raised `ValueError: Input contains infinity`. | **Fixed:** compute `eta` as the GLM linear predictor and clip to `[log(FL), log(PRED_CAP)]`; clip XGB predictions to `[FL, PRED_CAP]`. `PRED_CAP=1.0` (σ=100%/day) is a numerical guard far above any real variance, not a modelling fallback (documented). |
| 3 | Major | Unfair "beat-champion" comparison: plain XGBoost gamma was ~4% worse than the HGBR champion (XGB 1.6509 vs GBME 1.5825 at h1) due to weaker default leaf regularisation, confounding "anchor effect" with "XGB<HGBR". | **Fixed:** added `min_child_weight=20` (analog of the champion's `min_samples_leaf=20`) → XGB≈GBME (1.5835 vs 1.5825). The comparison `GLM+XGB vs XGB` now isolates the anchor within one library; `vs GBME` is the deployed-champion test. |
| 4 | Minor | `keep` variable assigned but unused (post-gen ruff-F). | **Fixed:** removed. |
| 5 | Info | Leakage audit: the StandardScaler, GLM coefficients and XGB trees are all fit on `trf_e` (train minus the val slice) only; the base margin for the `combo` rows is produced by that train-only GLM; test rows never influence any fit. Walk-forward embargo `int(1.6h)+5` days. Causal by construction. | No change. |
| 6 | Info | Positivity floor `FL` is identical across GBME / XGB / GLM+XGB and in every QLIKE evaluation (per the H2 leakage lesson: compared models must share the floor). | No change. |
| 7 | Info | Loss consistency: Stage-1 GammaRegressor optimises gamma deviance (= QLIKE up to constants) and Stage-2 uses `objective="reg:gamma"`, so the anchor is loss-consistent — the exact fix for the E8 HAR-anchor loss-mismatch this baseline targets. | No change. |

## Verification
- 12 tests pass; diff-coverage C0 line = 100% and C1 branch = 100% on all four code modules.
- Real HOSE smoke (h1, 1 fold) runs clean after the numerical guard: GBME 1.5825 / XGB 1.5835 / GLM+XGB 1.5866
  → anchor neutral-to-slightly-negative, DM ns (the expected NO-GO signal; full run pending).
- No Critical/Major findings remain open.
