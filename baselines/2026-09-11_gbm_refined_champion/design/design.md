# Design — Refined SP500 champion

## Architecture / data flow
Reuse the canonical walk-forward infrastructure verbatim (same folds/HAR-X/DM/metrics as the committed
`2026-09-09_gamma_gbm_earnings`), adding three feature panels:

```
enriched CSVs ──> build_enriched_panel ──> panel(feats[T,N,5]=HAR, pk, anchors, target_dates)
                                              ├─ G.extra_feature_panels(feats)      -> extras[T,N] x6 (rq, mr_*)
                                              ├─ estimator_panels(panel, files)     -> est[T,N,3] (GK/RS/YZ, day t)
                                              ├─ spike_panels(panel)                -> spk[T,N] x4 (vov/spike/accel)
                                              └─ asym_earnings_panels(panel, earn)  -> asym[A,N] x4 (prox/soon/pre/post)
make_folds ──> per fold: pack_fold ──> _design_refined(...) [n*N, 22] ──> HistGBM(gamma) ──> pooled test preds
                                        _earn_design(...)    [n*N, 13] ──> GBM+earn (committed champion)
                                        G._harx_ols(D)                  ──> HAR-X
RMR._metrics / RMR._dm_all ──> results JSON (+ train QLIKE overfit evidence)
```

## Key design decisions
- **Index spaces (the subtle part):** HAR/extras/estimators/spike are `[T,N]` (or `[T,N,·]`) on the panel date
  axis and are indexed by `panel.anchors[fold.*]`; the earnings panels are `[A,N]` on the anchor axis and are
  indexed by the positional `fold.*`. `_design_refined` respects both (mirrors the committed `GE._design`), so
  estimators/spike line up row-for-row with HAR at origin t while earnings line up by anchor.
- **NaN handling:** estimators may be NaN (missing column / off-date) and spike features NaN in warm-up; these
  are passed through to `HistGradientBoostingRegressor`, which supports NaN natively — no imputation, so **no
  leakage** from a global fill. HAR-X (OLS) uses only the 5 HAR features (never NaN at valid anchors).
- **Uniform design signature:** both `_earn_design` and `_design_refined` take
  `(har5, extras, est, spk, asym, anchors, pos)` so `run()` iterates the two models in one loop.
- **Overfit evidence:** each fold records TRAIN QLIKE alongside TEST; the JSON reports mean train→test gap%. GBM
  is a deterministic gamma-boosting baseline (not in `overfit_check.looks_learned`), so this is evidence-by-choice,
  not gate-forced — but it directly answers "does the richer 22-feature model overfit?".

## Gates (SDD §5)
- **Simplicity Gate:** no new project/abstraction; extends one existing module, adds 3 small panel builders. Pass.
- **Anti-Abstraction Gate:** uses `HistGradientBoostingRegressor` + the existing canonical panel/fold/metric code
  directly; no wrappers. Pass.
- **Performance/Batching Gate:** GBM is fit once per fold on the full pooled cross-section (all tickers × train
  days) — inherently batched (no per-item loop); estimator/spike panels are vectorised `[T,N]` rolling ops;
  earnings distances are `searchsorted` per ticker. Data-processing, not GPU. Pass.

## Files
- `code/gbm_refined_walkforward.py` — panels + `_design_refined` / `_earn_design` + `run()` + CLI.
- `test/test_gbm_refined.py` — unit (signed-dist, panel shapes/ranges, estimator alignment, NaN pass-through) +
  smoke (design column counts). Dummy data, no heavy SP500 load.

## Risks
- Earnings coverage from yfinance is imperfect (some tickers lack dates) — those contribute all-zero earnings
  features (neutral), same as the committed baseline.
- The 3 layers were each small (~0.4–1.5%); their SUM may be sub-additive (shared variance). The walk-forward run
  measures the true combined effect and its DM significance — the go/no-go criterion is on the combined result,
  not the sum of probe gains.
