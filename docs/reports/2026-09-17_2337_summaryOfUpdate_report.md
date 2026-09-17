# Summary of update — Hướng 4: OOF metric-constrained stacking (HOSE)

Date: 2026-09-17. Baseline: `baselines/2026-09-17_constrained_stacking/`. Verdict: **NO-GO** (as
pre-registered). The constrained stack never beats the best single base model on out-of-sample QLIKE.

## What changed
New self-contained baseline implementing a 2-level stack: level-1 diverse own-history base models
produce leakage-safe out-of-fold (OOF) predictions; a level-2 meta-learner picks non-negative simplex
weights minimising pooled QLIKE, fit on OOF and applied to test. Ran on HOSE, all 4 horizons, 8
walk-forward folds, 3 seeds.

## Files (path → purpose)
| path | purpose |
|------|---------|
| `baselines/2026-09-17_constrained_stacking/requirements/requirements.md` | spec, base set, kill criterion |
| `baselines/2026-09-17_constrained_stacking/design/design.md` | data flow, OOF/meta design, gates |
| `baselines/2026-09-17_constrained_stacking/code/stacking_config.py` | single-source tunable constants |
| `baselines/2026-09-17_constrained_stacking/code/base_models.py` | 4 base predictors, OOF cross-fitting, simplex-QLIKE meta-optimiser |
| `baselines/2026-09-17_constrained_stacking/code/run_constrained_stacking.py` | walk-forward driver, pooling, DM, spike, checkpoint |
| `baselines/2026-09-17_constrained_stacking/code_review/code_review_2026-09-17.md` | adversarial review |
| `baselines/2026-09-17_constrained_stacking/test/{conftest,__init__,test_constrained_stacking}.py` | tests |
| `results/gamma_gbm/stacking_hose_h{1,5,10,22}.json` | pooled results (metrics/train/val + fit_diagnostics + DM + weights + spike) |

## Design (brief)
- **Base models:** GBME (champion HGBR gamma, `FM.gbm`), XGB (`xgboost reg:gamma`, champion-matched
  capacity), GLM (`GammaRegressor(alpha=1.0)` log-link) on OWN-8+EARN; HAR (OLS on har_daily/weekly/
  monthly, MSE-trained — deliberately different member).
- **OOF:** within each outer train window, inner temporal K-fold (K=3) — every train row predicted by
  a base not trained on it (mirrors `embed.oof_train_z`). Test predictions from bases fit on the full
  outer train window (minus a trailing 22-date val slice). OOF feeds ONLY the meta-weights.
- **Meta-optimiser:** `scipy.optimize.minimize` SLSQP over the simplex (`w≥0`, `Σw=1`), objective =
  pooled QLIKE with champion floor `FM.FL`; degenerate all-zero return falls back to uniform.
- **Numerical guards:** gamma predictions floored at `FL`; XGB/GLM/HAR/stack clipped to `[FL, 1.0]`.
- Walk-forward over `S1.FOLDS` from `TRAIN_START`, embargo `int(1.6h)+5` days; real crawled VN
  earnings (`hose_earnings_combined.parquet`, 402 tickers).

## Final HOSE results (pooled 8 folds, 3 seeds)
QLIKE per model, learned weights (mean over folds), and DM of **stack vs the ex-post best single base**
(primary, conservative test — the stack must beat an oracle-selected baseline):

| h  | GBME | XGB | GLM | HAR | **stack** | best single | stack vs best (gain%) | DM p | weights GBME/XGB/GLM/HAR | ex-spike gain% (p) | beats |
|----|------|-----|-----|-----|-----------|-------------|-----------------------|------|--------------------------|--------------------|-------|
| 1  | 1.5719 | 1.5689 | 1.7889 | 1.813 | 1.5690 | XGB  | **−0.007** | 0.939 | 0.41 / 0.55 / 0.03 / 0.01 | −0.148 (0.136) | False |
| 5  | 1.6479 | 1.6512 | 1.8129 | 1.823 | 1.6515 | GBME | **−0.220** | 0.000 | 0.50 / 0.44 / 0.06 / 0.01 | −0.279 (0.000) | False |
| 10 | 1.6856 | 1.6838 | 1.822  | 1.8289| 1.6855 | XGB  | **−0.102** | 0.406 | 0.51 / 0.45 / 0.03 / 0.02 | −0.145 (0.265) | False |
| 22 | 1.7270 | 1.7270 | 1.8328 | 1.8366| 1.7275 | XGB  | **−0.030** | 0.804 | 0.69 / 0.28 / 0.01 / 0.03 | −0.060 (0.658) | False |

Pre-registered kill criterion (stack beats best single at BOTH h1 and h5, DM p<0.05, gain>0,
spike-robust): **h1 False, h5 False → NO-GO**. Driver's `PRE-REGISTERED SUCCESS = False`.

## Interpretation (the falsification)
- The stack's QLIKE gain vs the best single base is **negative at every horizon** (−0.007% to −0.22%),
  and at h5 it is **significantly worse** (p<1e-6, robust to spike-exclusion). The stack never wins.
- The weights **collapse onto the two tree-gamma bases** (GBME+XGB ≈ 0.96–0.97 of the mass at every
  horizon); GLM and HAR get near-zero weight. GBME and XGB are near-identical gamma boosters (QLIKE
  within 0.002), so the stack effectively averages two almost-collinear models and cannot beat the
  ex-post best of the pair — the noise in the OOF-estimated weights makes it slightly worse.
- Confirms the pre-registered hypothesis and the project finding that own-history base models are too
  correlated for a metric-constrained stack to add OOS value on VN volatility. The stack-vs-GBME
  column (e.g. h1 +0.18%, p≈0) is trivial — it only reflects XGB>GBME with the stack tracking XGB, not
  any stacking gain; the correct (best-single) comparison is uniformly NO-GO.

## Tests + coverage
- `python -m pytest baselines/2026-09-17_constrained_stacking/test/` → **19 passed** (`.venv_gpu_encode`).
- Diff-coverage on the three code modules (pytest-cov `--cov-branch`): **C0 line = 100%, C1 branch =
  100%** (0 missed statements, 0 missed branches). Covered: every base predictor (positive/finite/
  capped), OOF coverage+causality+coverage-gap raise, simplex constraint + collapse-onto-dominant +
  degenerate fallback, `stack_qlike`, verdict/success, `_safe_dm` degenerate + ValueError, spike mask,
  `_apply_weights`, `_load_earn` (3 branches), `_metrics5`, and the walk-forward run smoke variants
  (hose per-fold+spike, sp500 no-spike, multi-fold skip, all-in-spike omit) with gate-required
  evidence keys asserted.
- Real-model end-to-end smoke booted the driver before the full run.

## Code review
`code_review/code_review_2026-09-17.md` (adversarial: leakage, loss consistency, numerical guards,
simplex-constraint correctness, config hygiene, performance). No CRITICAL/MAJOR blockers. Key points:
OOF-future-block training is used only for meta-weights (no forecast leakage); best-single picked
ex-post is a conservative test against the stack; stack train metric is optimistic but the stack is
deterministic-exempt from the fit gate. Documented the single-fold-vs-pooled XGB gap (below).

## Gate evidence / risks
- **Over/under-fit gate:** every model in all 4 result JSONs classifies **`ok`**; `check_result_evidence`
  returns `(True, [])` for h1/h5/h10/h22. The only auto-detected learned model (`XGB`) is `ok`
  (val→test QLIKE gap ~3–4% pooled). Note: on a SINGLE fold checkpoint XGB's gap can read ~35% (a
  val/test regime shift), collapsing to ~3–4% only on the full 8-fold pool — same behaviour as the
  delivered sibling `glm_anchor_hose` (XGB gap 0.0353, `ok`). This is why partial checkpoints must not
  be read as final.
- **Lint:** `ruff check --select F` clean on code + test. **Config-hardcode:** postgen gate exit 0 on
  both pipeline files (all tunables in `stacking_config.py`).
- **Data-quality gate (Pandera/Evidently):** N/A — no data/feature/manifest change (reads existing
  enriched HOSE panel + existing earnings parquet).
- **Performance:** vectorised tabular fits (sklearn/xgboost hist/numpy), no batch=1 neural loop, no GPU
  path needed; OOF bounded at K=3.

## Follow-ups
- None required for the falsification. If ever revisited, a genuinely-decorrelated member (not another
  own-history gamma booster) would be needed before a stack could plausibly help — consistent with the
  exhausted-lever findings in project memory.
