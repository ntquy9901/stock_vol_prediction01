# Requirements — Literal GBME + leaf-cooccurrence graph (Option A)

Baseline: `2026-09-18_gbme_leafgraph_literal` · markets HOSE + SP500 · target = Parkinson variance sigma^2.

## Objective
Falsify (or confirm) whether smoothing the champion **gamma-GBME's own predictions** over a graph built from
**GBME's own trees** (leaf-index cooccurrence) beats the un-smoothed GBME on out-of-sample QLIKE. This is the
*literal* Option A: the leaf graph is extracted from the fitted `HistGradientBoostingRegressor` (HGBR) itself —
**no XGBoost proxy**. The committed sibling `2026-09-18_gbm_leaf_graph` used an XGBoost booster as a
capacity-matched proxy (XGB exposes public leaf indices); this baseline replaces that proxy with the real GBME
tree structure and smooths the real GBME predictions.

## Mechanism (per walk-forward fold, per horizon h)
1. Fit GBME = `HistGradientBoostingRegressor(loss="gamma", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
   l2_regularization=1.0, min_samples_leaf=20, random_state=seed)` on the causal train slice `trf_e` (target
   floored at `FL`). Hyperparameters mirror the champion `full_matrix.gbm` (single-sourced in `gbme_lg_config`;
   verified equal by test).
2. Extract the (n_samples x n_trees) **leaf-index matrix** from the fitted HGBR by traversing each tree's
   `.nodes` structured array on the RAW feature path (`num_threshold`), from `model._predictors`. A **fail-loud
   reconstruction guard** asserts `baseline_prediction + sum(leaf value) ~= model._raw_predict(X)` (link space)
   each fold — proving the private-API extraction is correct and pinning against sklearn drift.
3. Per TEST DAY cross-section: leaf-Hamming similarity `sim = (#trees same leaf)/T` -> kNN k=10 (self excluded)
   -> smooth `y_hat = (1 - alpha) * y_gbme + alpha * mean_kNN(y_gbme)`. `alpha` is fit on the val slice per fold
   (grid 0..1, min QLIKE), frozen for test.
4. GBME predictions are **seed-ensembled** over seeds (0,1,2). The leaf graph is extracted from **seed[0] only**
   (the graph is a structural re-encoding; a single seed's tree structure suffices) while the smoothed
   predictions operate on the seed-ensembled GBME base.
5. Features = OWN-8 + EARN-4 (full). SP500 carries real earnings; HOSE uses crawled VN announcement dates if
   present.

## Input / Output
- Input: `data/processed_enriched/{hose,sp500_clean}/*.csv` via `full_matrix.load`; earnings parquet as in the
  sibling drivers.
- Output: `results/gamma_gbm/gbme_lg_<market>_h<h>.json`, one per horizon, atomic-checkpointed per fold. Each
  carries pooled 5-metric train/val/test for `GBME` and `GBME+leafgraph`, `fit_diagnostics`, date-clustered DM
  (`GBME+leafgraph` vs `GBME`), gain %, verdict, per-fold fitted alpha, HOSE per-fold + spike robustness.

## Success / kill criterion (pre-registered)
- A horizon "beats" iff QLIKE gain over GBME `> GAIN_MIN` **AND** date-clustered DM p `< DM_ALPHA`.
- Pre-registered GO: `GBME+leafgraph` beats `GBME` at **BOTH** h1 and h5 (gain>0, DM p<0.05, spike-robust on
  HOSE) -> else NO-GO.
- Prior (from the XGB proxy): small positive on HOSE (h1/h5/h10 significant, h22 not), ~0 on SP500 (alpha->0).
  This baseline reports whether the literal HGBR version reproduces that pattern.

## Acceptance (Definition of Done)
- [ ] 5 subfolders present.
- [ ] Tests pass under `.venv_gpu_encode` pytest; diff-cov C0=100% / C1>=95% on changed lines.
- [ ] Reconstruction guard verified in a test (tiny fit) and enforced each fold at runtime.
- [ ] Adversarial code review done; HIGH/MAJOR fixed.
- [ ] HOSE + SP500 run for all 4 horizons; every JSON `n_folds == 8`.
- [ ] JSONs pass `check_overfit_evidence` (carry train/val/fit evidence).
- [ ] Summary report with the GBME vs GBME+leafgraph table per market x horizon.

## Non-goals
- No modification of any sibling baseline or shared `src/` (read-only imports only).
- No git operations (coordinator commits).
- `archive/` out of scope.
