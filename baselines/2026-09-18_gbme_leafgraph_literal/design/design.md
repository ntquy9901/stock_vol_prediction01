# Design — Literal GBME + leaf-cooccurrence graph (Option A)

## Architecture / data flow
```
full_matrix.load(market) -> frames, sectors, edates
  edates <- _load_earn (real crawled VN dates for HOSE if present)
for h in HORIZONS:
  panel = FM.panel(frames, edates, h)                      # OWN-8 (+EARN-4) features, y = pk_var.shift(-h)
  for k in folds (S1.FOLDS / S1.TRAIN_START, embargo=int(1.6h)+5):
     trf  = causal train window;  tef = test window
     val_dates = last VALID_LEN train dates ;  trf_e = trf \ val ; vaf = val slice
     models = [fit_gbme(trf_e, cols, seed) for seed in seeds]     # HGBR gamma, seed-ensemble
     X = combo[cols]  where combo = concat([tef, trf_e, vaf])
     p_gbme = mean_seed( predict_gbme(model, X) )                 # floored base
     leaves = leaf_matrix(models[0], X)                           # seed[0] tree structure + RECON GUARD
     alpha  = fit_alpha(y_val, p_gbme|val, leaves|val, dates|val) # grid, min val QLIKE
     GBME+leafgraph = maximum( smooth_all(p_gbme, leaves, dates, k, alpha), FL )   # same floor as base
  pool over folds -> 5-metric train/val/test, fit_diagnostics, date-clustered DM, verdict, alpha, spike
```

## The NEW piece — `hgbr_leaf.py`
HGBR has no public `.apply()`. Per-tree leaf indices come from `model._predictors` (list per boosting iter x
list-per-output=1). Each `TreePredictor.nodes` is a structured array with fields
`(value, count, feature_idx, num_threshold, missing_go_to_left, left, right, gain, depth, is_leaf, ...)`.
Traversal on the RAW path avoids `_bin_mapper`: at an internal node go `left` if `x[feature_idx] <=
num_threshold` else `right`, until `is_leaf`.

- `fit_gbme(trf, cols, seed, floor)` -> fitted HGBR (params single-sourced `GBME_PARAMS`, matched to
  `full_matrix.gbm` — verified equal by test on a fixture; NB HGBR default `early_stopping='auto'` fits fewer
  than `max_iter` trees; leaf-vector length = number of fitted trees, identical across our uses because
  `random_state` and params are fixed).
- `predict_gbme(model, X, floor)` -> `np.maximum(model.predict(X), floor)` (matches champion `FM.gbm`, no upper
  cap on the base so parity with the deployed GBME is exact).
- `_traverse_tree(nodes, X)` -> `(leaf_node_idx[n], leaf_value[n])`, **vectorised over samples** (all samples
  descend in lockstep; a boolean `active = ~is_leaf[node]` mask iterates until every sample reaches a leaf).
  Bounded by tree depth. This is the perf-critical path (HGBR node traversal in pure Python per-sample is slow).
- `leaf_matrix(model, X, check=True, tol=RECON_TOL)` -> int32 `(n x n_trees)`. Loops trees (vectorised over
  samples), accumulates `sum(leaf value)`; when `check`, asserts
  `max| baseline + sum(value) - _raw_predict(X) | <= tol` (link space) -> **raise** on failure (fail-loud vs
  sklearn drift). Runs each fold at runtime (seed[0]).

## Graph helpers — `leaf_graph.py`
Copied (self-contained, pure numpy/scipy — no XGBoost) from the sibling's verified graph functions:
`day_similarity` (sparse one-hot inner product / T), `knn_neighbour_mean` (top-k, self excluded, singleton
returns base), `smooth_day`, `smooth_all` (per-day cross-section, alpha==0 identity fast path), `fit_alpha`
(grid min val QLIKE, ties keep smaller alpha).

## Driver — `run_gbme_leafgraph.py`
Structural template = the sibling `run_leaf_graph.py`: walk-forward over `S1.FOLDS`/`S1.TRAIN_START`, embargo
`int(1.6h)+5`, val slice `VALID_LEN=22` days, `_pool_doc` (5-metric train/val/test + `classify_fit` +
date-clustered DM via `_safe_dm` + per-fold + spike), atomic checkpoint. Models = `GBME`, `GBME+leafgraph`
(ORDER of length 2). One horizon per fresh process; per-fold train arrays are transient (freed by GC; `del fp`).

## Gates (SDD)
- **Simplicity Gate:** reuse `full_matrix` / `metrics` / `stats` / `overfit_check` / graph helpers; the only new
  logic is the HGBR leaf traversal + guard. No new abstraction beyond that. PASS.
- **Anti-Abstraction Gate:** uses sklearn HGBR directly; no wrapper framework. PASS.
- **Performance/Batching Gate:** leaf traversal is **vectorised over samples** (not per-item); seed[0]-only leaf
  extraction; per-day smoothing vectorised via sparse matmul; alpha==0 identity fast path; one horizon/process,
  per-fold arrays freed. No batch=1 hot loop. PASS.

## Config — `gbme_lg_config.py` (single source of truth)
`GBME_PARAMS` (loss/max_iter/lr/max_leaf_nodes/l2/min_samples_leaf), floor via `FM.FL`, `RECON_TOL`,
`K_NEIGHBOURS=10`, `ALPHA_GRID` 0..1, `VALID_LEN=22`, `HORIZONS`, `MIN_ROWS`, `GAIN_MIN`, `DM_ALPHA`,
`KILL_HORIZONS=(1,5)`, `SPIKE_WINDOWS`. No magic numbers in pipeline modules. Both prediction arms are floored
by `np.maximum(., FL)` (no asymmetric upper cap), so `alpha==0` makes the smoothed arm identical to GBME.

## Leakage controls
- Leaf matrix + GBME fit on `trf_e` only (val excluded from fit); graph on each test day uses only that day's
  cross-section (contemporaneous, no future/cross-day mixing); alpha fit on the val hold-out, frozen for test.
- Embargo `int(1.6h)+5` between train end and test start.

## Fragility note
`_predictors`, `TreePredictor.nodes` dtype are sklearn-private. Accepted because (a) sklearn version pinned in
the venv (1.7.2), (b) the reconstruction guard fails loud each fold if a future upgrade changes the layout.
Fallback (Option B, binned via `_bin_mapper`) documented in `gbme_leafgraph_integration.html`, not implemented.
