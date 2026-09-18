# Design — GBME + leaf-cooccurrence graph (falsification)

## Data flow
```
processed_enriched/hose/*.csv  ─┐
hose_earnings_combined.parquet ─┤→ FM.load + _load_earn → frames, edates
                                └→ FM.panel(frames, edates, h) → panel a (per-day cross-section, target y=shift(-h))
per horizon h, per outer fold k (S1.FOLDS, embargo int(1.6h)+5):
  trf = train window [TRAIN_START, ts-embargo)   (min MIN_ROWS rows)
  tef = test window  [ts, tend)
  val = last VALID_LEN train dates;  trf_e = trf\val,  vaf = val
  combo = concat([tef, trf_e, vaf])                (single feature matrix; dates disjoint across splits)
  ── GBME       = mean_s FM.gbm(trf_e, combo, cols, s)           (HGBR gamma, seed ensemble)
  ── XGB        = mean_s predict_booster(fit_booster(trf_e,s))   (XGBoost gamma, seed ensemble) — SHARED base
  ── leaves     = leaf_matrix(fit_booster(trf_e, seeds[0]))      (n_combo × n_trees int, single booster)
  ── α          = fit_alpha on the VAL slice (grid ALPHA_GRID, minimise val QLIKE)
  ── XGB+leafgraph = clip( smooth_all(XGB base, leaves, dates, K_NEIGHBOURS, α), FL, PRED_CAP )
pool test/train/val over folds → _pool_doc → results/gamma_gbm/leaf_graph_hose_h<h>.json (atomic per fold)
```

## Key design decisions
- **Shared XGB base isolates the graph effect.** XGB and XGB+leafgraph use identical seed-ensembled base
  predictions; the ONLY difference is the α-smoothing. So `XGB+leafgraph vs XGB` measures the graph alone.
- **Single deterministic booster for the leaf-graph** (seed `FM.SEEDS[0]`), while the base prediction is the
  seed ensemble. The graph is a fixed structure; the forecast quality comes from the ensemble base.
- **Leaf-Hamming similarity via sparse one-hot inner product** (`day_similarity`): globally-unique `(tree,leaf)`
  ids → `S`, `sim = (S @ Sᵀ)/T`. Exact fraction-of-matching-leaves, far cheaper than an `m×m×T` broadcast.
- **Per-day smoothing = causal.** Each date's cross-section is smoothed independently (`smooth_all` groups by
  date). Train/val/test occupy disjoint date ranges (temporal split), so no split ever mixes with another and
  the graph never sees the future. The booster is fit on `trf_e` only.
- **α fit on val, frozen for test** (mirrors `baselines/2026-09-07_regime_blend`): 1-D grid search minimising
  validation QLIKE. `α=0` is an identity fast path in `smooth_all` (skips building any graph) — exact, and the
  expected NO-GO outcome, which keeps train-metric computation cheap.
- **Numerical guards:** XGB gamma predictions clipped to `[FL, PRED_CAP]`; the smoothed convex combination is
  re-clipped. Guards against the gamma exp-link overflow lesson from the sibling glm_anchor baseline.
- **Capacity matched to GBME:** XGB params single-sourced in `leaf_graph_config.py` mirror the champion HGBR
  (300 trees / lr 0.05 / 31 leaves / l2 1 / min_child_weight 20), so XGB ≈ GBME and the comparison is fair.

## Gates
- **Simplicity:** no neural net, no new dependency beyond XGBoost (already used by the sibling baseline) and
  scipy.sparse (already installed). Minimal convex-combination smoothing.
- **Anti-abstraction:** reuses `full_matrix` (FM.gbm/panel/load/OWN/EARN/SEEDS/FL), `metrics.per_obs_qlike`,
  `stats.date_clustered_dm`, `overfit_check.classify_fit`, `vn_gbm_graph_stage1.FOLDS/TRAIN_START/_feat`.
- **Performance/batching:** predictions are computed on batched feature matrices (`combo`); the α=0 identity
  fast path avoids all graph construction in the expected case. Per-day similarity is a vectorised sparse
  product, not a Python per-pair loop.

## Files
| Path | Purpose |
|------|---------|
| `code/leaf_graph_config.py` | all tunable constants (XGB params, K, α grid, horizons, gates, spike windows) |
| `code/leaf_graph.py` | booster fit/predict, leaf matrix, Hamming similarity, kNN neighbour-mean, smoothing, α fit |
| `code/run_leaf_graph.py` | walk-forward driver: per-fold predictions, pooling, DM, verdict, spike, checkpoints |
| `test/test_leaf_graph.py` | similarity/kNN/smoothing/causality/α-fit/driver-logic + run smoke (24 tests) |
| `test/conftest.py` | sys.path bootstrap for dash-named baseline folder |

## Result artifacts
`results/gamma_gbm/leaf_graph_hose_h{1,5,10,22}.json` — each carries `metrics`/`train_metrics`/`val_metrics`
(5 metrics × 3 models), `fit_diagnostics`, `alpha` (per_fold + mean), `dm`, `gain_pct`, `verdict`,
`per_fold_qlike`, `spike_robustness`.
