# Design — leaf-graph FINAL paper-grade (3-seed, XGB-only, earnings ablation)

## Architecture / data flow
```
FM.load(market) -> frames, sectors, edates
_load_earn(market, edates)  # HOSE: swap in real crawled VN announcement dates parquet; SP500: keep FM edates
cols = OWN-8  +  (EARN-4 if feature_set == "full" and has_earn else [])     # <-- the feature-set / ablation switch
for h in horizons:
  a = FM.panel(frames, edates, h)                 # earn cols present but ignored when cols == OWN (noearn)
  for fold k in 8 walk-forward folds:             # S1.FOLDS gives 8 folds, embargo = h*1.6+5 days
     trf  = train window [TRAIN_START, fold_start - embargo)
     tef  = test window  [fold_start, fold_end)
     val  = trailing VALID_LEN dates of trf ; trf_e = trf - val
     XGB   = mean_s XGBoost-gamma(trf_e -> combo[cols], seed s)      # 3 seeds ; SHARED base
     leaves = leaf_matrix(booster(trf_e, seed0))                     # per-tree leaf idx for the same-day graph
     alpha  = fit_alpha(val)                                         # causal, frozen for test
     XGB+leafgraph = clip(smooth_all(XGB, leaves, dates, k, alpha), FL, PRED_CAP)
     stream[m].update(y_tr, pred_tr[m])            # TRAIN metrics streamed (suff. stats) -> free train arrays
     pool val/test arrays (needed for DM)
  doc = pooled 5-metric train(streamed)/val/test + fit_diagnostics + DM(XGB+leafgraph vs XGB) + per-fold + spike + alpha
  checkpoint atomically per horizon
```

## Key design decisions
- **Model set = {XGB, XGB+leafgraph} ONLY.** `GBME`/`FM.gbm` is deliberately not fit or reported (paper scope +
  RAM). `ORDER = [XGB, XGB+leafgraph]`; every pooled/DM/fit block iterates that pair only.
- **Reuse, don't reinvent.** The leaf-graph math (`day_similarity`, `knn_neighbour_mean`, `smooth_day`,
  `smooth_all`, `fit_alpha`, `fit_booster`/`predict_booster`/`predict_xgb`/`leaf_matrix`) is COPIED verbatim from
  the committed `leaf_graph.py` into `code/leaf_graph_lib.py` (module renamed, import points to THIS baseline's
  config) so the baseline is self-contained and every tunable constant is single-sourced in
  `leaf_graph_paper_config.py`. The committed baseline is NOT edited.
- **Feature-set switch = leave-one-out of EARN.** `resolve_cols`: `cols = OWN` (noearn) drops exactly the 4 EARN
  features; XGB fits on `cols`. Rows are identical to `full` (panel dropna on OWN+y), so full-vs-noearn on the
  SAME base isolates the earnings marginal effect, and the leaf-graph GO/NO-GO can be checked in both arms.
- **3-seed ensembling.** `SEEDS=(0,1,2)`; the XGB base is averaged over seeds; the leaf graph is built from
  `seeds[0]`'s booster (a single deterministic tree geometry — the graph is a re-encoding, not a seed-averaged
  object).
- **Streaming train metrics (RAM).** `_Stream` accumulates sufficient statistics (`n, sse, sae, sy, syy, sq`) and
  reproduces mse/rmse/mae/r2/qlike exactly on pooled data (pooled `SS_tot = syy - sy^2/n`); each fold's train
  arrays are folded in then dropped. Only val/test arrays are pooled (needed for date-clustered DM). One horizon
  per process (`--horizon`).
- **Causality.** alpha fit on a trailing val slice carved from train (never test); each test day's graph uses ONLY
  that day's cross-section (`smooth_all` groups by date, no cross-day mixing); booster fit on the past train
  window only (embargo `h*1.6+5` days). Same guarantees as the committed baseline (unchanged math).
- **Numerical guards.** target floored at `FL`; predictions clipped to `[FL, PRED_CAP]`; `_safe_dm` returns p=1.0
  for degenerate DM (identical loss series / too-few dates); zero-variance target -> r2 = 0.0 (both `_Stream` and
  `_metrics5`).

## Gate design decisions (SDD gates)
- **Simplicity gate:** one driver + one copied math lib + one config; no new abstractions beyond the feature-set
  switch and the streaming accumulator (both required by the task).
- **Anti-abstraction gate:** reuse `full_matrix`/`vn_gbm_graph_stage1`/`metrics`/`stats`/`overfit_check` read-only.
- **Performance/batching gate:** XGBoost `hist` trains batched over the whole train matrix (no per-item loop); the
  seed loop is inherent to the ensemble; the graph is a vectorised sparse one-hot inner product per day; the
  streaming accumulator bounds memory. GPU is not applicable (tree models on tabular data). Batch=1 is not used.

## Config (single source — `leaf_graph_paper_config.py`)
`N_SEEDS=3`, `SEEDS=(0,1,2)`, XGB gamma params (`XGB_N_ESTIMATORS/LR/MAX_LEAVES/MAX_DEPTH/L2/MIN_CHILD_WEIGHT`),
`PRED_CAP`, `K_NEIGHBOURS`, `ALPHA_GRID`, `VALID_LEN`, `HORIZONS`, `HORIZONS_SMOKE`, `MIN_ROWS`, `GAIN_MIN`,
`DM_ALPHA`, `KILL_HORIZONS`, `SPIKE_WINDOWS`. XGB capacity mirrors the champion HGBR gamma (300 trees, lr 0.05,
~31 leaves, l2=1, min_child_weight~20).

## Files
- `code/leaf_graph_paper_config.py` — all tunable constants.
- `code/leaf_graph_lib.py` — copied leaf-graph math (imports this baseline's config).
- `code/run_leaf_graph_paper.py` — walk-forward driver: {XGB, XGB+leafgraph} only, feature-set switch, 3-seed,
  streaming, per-horizon, DM, fit diagnostics, per-fold + spike, atomic checkpoint, distinct filenames.
- `test/test_leaf_graph_paper.py` — math + driver logic + streaming + feature-set switch + causality + run smoke.
