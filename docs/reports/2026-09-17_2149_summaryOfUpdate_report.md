# Summary of update — Gamma-GLM base-margin anchor + GBDT residual (Hướng 3), full HOSE result

Date: 2026-09-17. New baseline `baselines/2026-09-17_gbme_glm_anchor` (code pushed at `df363e73`; this adds the
full HOSE result JSONs + verdict). Tests the methodologically-correct anchor that the failed HAR-anchor (E8)
lacked: a Stage-1 Gamma GLM (log-link, IRLS on the gamma deviance = QLIKE) used as the XGBoost gamma **base
margin**, Stage-2 boosting the residual — `ŷ = exp(η_GLM + Σ trees)`. Loss-consistent by construction, so it
cannot bias calm-day QLIKE the way an OLS/MSE HAR anchor did.

## Result (HOSE, 8 folds × 3 seeds, date-clustered DM, spike-robust; GLM_ALPHA=1.0)
| h | GBME | XGB | GLM+XGB | GLM+XGB vs GBME | DM p | vs XGB (anchor) DM p | beats | ex-spike gain / p |
|---|---|---|---|---|---|---|---|---|
| h1 | 1.5719 | 1.5689 | 1.5674 | +0.28% | 0.007 | 0.125 | True | +0.41% / 0.000 |
| h5 | 1.6479 | 1.6512 | 1.6499 | −0.12% | 0.139 | 0.011 | False | −0.05% / 0.52 |
| h10 | 1.6856 | 1.6838 | 1.6835 | +0.12% | 0.283 | 0.524 | False | +0.12% / 0.27 |
| h22 | 1.7270 | 1.7270 | 1.7272 | −0.01% | 0.920 | 0.703 | False | +0.02% / 0.90 |

**Verdict: NO-GO.** Pre-registered success (beat GBME at BOTH h1 and h5) fails at h5. More decisively, the
**anchor mechanism itself is null**: `GLM+XGB vs plain XGB` (same library, isolates the base margin) is
non-significant at h1/h10/h22 (p=0.125/0.524/0.703) and only borderline at h5 (p=0.011) with a trivial +0.08%
(1.6512→1.6499) that does not survive spike exclusion. The small +0.28% h1 win vs GBME is a library artifact
(XGB≈GBME after matching leaf regularisation; the anchor adds nothing on top of XGB). All four horizons pass the
over/under-fit gate (fit=ok, test R² +0.13…+0.21).

**Caveat (train R²).** GLM+XGB shows a persistently negative *train* R² (−0.84…−1.56) while *test* R² is
positive — a numerical artifact of the `PRED_CAP=1.0` guard clipping a handful of extreme GLM-extrapolated
predictions on the long 2015–2022 training history (the 2022+ test period has no such extreme feature rows). It
does not affect the QLIKE verdict, the DM tests, or the gate (underfit requires BOTH train and test R² < 0). It
is itself consistent with the finding: the GLM's linear extrapolation over-disperses on tail rows.

**Interpretation.** This closes the open question "did E8 (HAR anchor) fail only because of loss mismatch?"
The answer is **no**: even the loss-consistent Gamma-GLM anchor is null. The champion own-history gamma-GBM is
already loss-aligned (gamma deviance ≡ QLIKE, proven) and near estimator-optimal for its 8 features; a linear
GLM base margin adds the one thing trees lack — linear extrapolation — but own-history extrapolation carries no
QLIKE signal beyond the tree (storm onset is exogenous; under-dispersion is Bayes-optimal). Consistent with the
information-ceiling finding across every prior lever.

## GLM regularisation correction (why the numbers were re-run)
A first run used `GLM_ALPHA=1e-4` — far below sklearn's `GammaRegressor` default of `1.0`. That
under-regularised the z-scored coefficients, so the GLM linear predictor over-dispersed and GLM+XGB acquired a
negative MSE-R² (train −0.15, test −0.52) even while staying QLIKE-competitive — which correctly tripped the
pre-push over/under-fit gate. Rather than bypass the gate, the regularisation was set to the **library default
`GLM_ALPHA=1.0`** (the principled choice) and the run repeated. The well-regularised anchor shrinks toward plain
XGB (R² ≈ +0.21, matching XGB), passes the gate cleanly, and gives the same NO-GO verdict — confirming the
apparent effect of the under-regularised anchor was an artifact, and that a properly-specified loss-consistent
anchor collapses to XGB (adds nothing). The table above is the `alpha=1.0` run.

## Files
- `baselines/2026-09-17_gbme_glm_anchor/` — requirements, design, code (`glm_anchor_config`, `glm_anchor`,
  `run_glm_anchor`), code_review, tests (all pushed at `df363e73`).
- `results/gamma_gbm/glm_anchor_hose_h{1,5,10,22}.json` — this push (each carries all-5 metrics train/val/test +
  fit_diagnostics for GBME/XGB/GLM+XGB + DM + per-fold QLIKE + spike robustness).

## Tests / gate
- 12 tests pass; diff-coverage C0 line = 100% and C1 branch = 100% on all four code modules.
- Build-time findings (see `code_review/code_review_2026-09-17.md`): bare-`config` module collision (renamed to
  `glm_anchor_config`), exp-link overflow on real data (`PRED_CAP` clip guard), and unfair XGB-vs-HGBR base gap
  (`min_child_weight=20` matches the champion's `min_samples_leaf`, giving XGB≈GBME). All fixed.

## Paper GARCH (companion task, pushed at `df363e73`)
GARCH was already complete + valid for SP500 and HOSE at all 4 horizons; only 3 HOSE cells in
`docs/paper/soict_2026-09-12_multimarket.tex` were stale (written before the d195a917 near-IGARCH guard
regenerated `garch_hose.json`). Synced to the committed JSON (h1 1.654→1.656, h10 1.780→1.782, h22 1.838→1.837),
no prose change, PDF regenerated (11 pages, compiled clean).

## Code review
`/code-review`-equivalent adversarial pass recorded in the baseline `code_review/`; the pre-push 3-tier gate
(ruff-F, config-hardcode 0 BLOCK, diff-cover C0=100%/C1=100%, overfit-evidence, checklist) passed on the code
push. No production-path logic altered outside the new baseline.
