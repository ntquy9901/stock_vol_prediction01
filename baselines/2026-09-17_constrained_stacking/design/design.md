# Design — OOF metric-constrained stacking (Hướng 4)

## Data flow
```
FM.load(market) ── frames, edates ──► _load_earn (real VN dates for HOSE)
                                          │
                     for h in HORIZONS:   ▼
                     FM.panel(frames, edates, h)  = a  (OWN-8 + EARN + y=pk.shift(-h))
                                          │
        for each outer fold k (S1.FOLDS, embargo int(1.6h)+5 days):
            trf = train window (causal, < ts - embargo)   tef = test window [ts, tend)
            trailing VALID_LEN dates of trf ─► vaf (val slice); trf_e = trf minus vaf
                                          │
            ┌── base models fit on trf_e ──► predict combo=[tef, trf_e, vaf]
            │      GBME/XGB/GLM (OWN-8+EARN), HAR (HAR-3)     (test/train/val preds)
            │
            └── OOF: inner temporal K-fold over trf_e (base_models.oof_predict)
                   every trf_e row predicted by a base NOT trained on its inner block
                                          │
                   P_oof [n_trf_e × 4] ─► fit_simplex_qlike ─► w (simplex, QLIKE-optimal)
                                          │
            stack_{te,tr,va} = clip(P_{te,tr,va} @ w, FL, PRED_CAP)
                                          │
        pool over folds ─► 5 metrics + fit_diagnostics + date-clustered DM
                           (stack vs best-single, stack vs GBME) + per-fold + spike
        atomic checkpoint ─► results/gamma_gbm/stacking_<market>_h<h>.json
```

## Key design decisions
1. **Structural template = `baselines/2026-09-17_gbme_glm_anchor`.** Reuse its scaffolding verbatim
   where possible: `_own8()`, `_load_earn()`, walk-forward loop, val slice, `_metrics5`, `_safe_dm`,
   `_spike_mask`, `_checkpoint`, `_pool_doc`, atomic per-horizon checkpoint. Only the model set and
   the meta-learner are new. This keeps the leakage/embargo/DM logic identical to delivered baselines.
2. **OOF only feeds the meta-weights.** Base test/train/val predictions come from a single fit on
   `trf_e` (in-sample for train/val, out-of-sample for test — same as the sibling battery). The OOF
   inner K-fold exists solely so the level-2 weights are fit on predictions no base saw in training —
   the standard stacking anti-leakage guarantee. Fitting the meta on in-sample base predictions would
   let the stack overfit base quirks that do not generalise.
3. **Meta-optimiser = SLSQP on the simplex.** `minimize(stack_qlike, w0=uniform, method="SLSQP",
   bounds=[0,1], constraints=sum w = 1)`. QLIKE with the champion floor `FM.FL`. A degenerate solver
   return (all-zero) falls back to uniform (fail-safe, not silent-zero). The objective clips
   `P @ w` to `[FL, PRED_CAP]` so a gamma member overflow cannot poison QLIKE.
4. **Diversity by loss, not by feature.** Three gamma members (tree / boosted-tree / linear) plus one
   MSE-trained HAR OLS. If even the MSE member cannot pull weight off GBME, the correlation argument
   is confirmed.
5. **Numerical guards.** Champion GBME floors at `FL` only. XGB and GLM can exp-overflow on extreme
   z-scored rows, so their predictions are clipped to `[FL, PRED_CAP]` (PRED_CAP a config constant).
   GLM uses `GammaRegressor(alpha=1.0)` — the sklearn default; a lighter alpha under-regularises and
   produces negative R² that trips the over/under-fit gate.

## Simplicity / Anti-Abstraction / Performance gates
- **Simplicity:** one config module, two code modules (base models + driver), no new abstractions
  beyond the sibling template.
- **Anti-Abstraction:** uses sklearn / xgboost / scipy directly; reuses `FM.gbm`, `M.per_obs_qlike`,
  `ST.date_clustered_dm`, `OF.classify_fit` as-is.
- **Performance:** base predictions are seed-ensembled numpy/tree fits (vectorised, no batch=1 neural
  loop). The OOF inner K-fold is `K=3` (not leave-one-out) to bound cost. Gamma trees run on CPU
  hist; XGB uses `tree_method="hist"`. No GPU path needed (tabular gamma boosters).

## File list
| file | purpose |
|------|---------|
| `code/stacking_config.py` | single source of truth for all tunable constants |
| `code/base_models.py` | 4 base predictors, `inner_blocks`, `oof_predict`, `fit_simplex_qlike` |
| `code/run_constrained_stacking.py` | walk-forward driver, `_pool_doc`, DM, spike, checkpoint |
| `test/test_constrained_stacking.py` | unit + smoke tests (synthetic panels + fake base models) |

## Config constants (all in `stacking_config.py`)
INNER_K, VALID_LEN, HORIZONS(_SMOKE), MIN_ROWS, SLSQP_MAXITER, SLSQP_FTOL, GAIN_MIN, DM_ALPHA,
KILL_HORIZONS, SPIKE_WINDOWS, PRED_CAP, GLM_ALPHA, GLM_MAX_ITER, XGB_* (capacity), HAR_FLOOR_FRAC.
