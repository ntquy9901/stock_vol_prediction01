# Design — leaf-graph v2 (RF-GAP / KeRF)

## Data flow
```
processed_enriched/hose/*.csv  ─┐
hose_earnings_combined.parquet ─┤→ FM.load + _load_earn → frames, edates
                                └→ FM.panel(frames, edates, h) → panel a (per-day cross-section, y=shift(-h))
per horizon h (one fresh process via --horizon), per outer fold k (S1.FOLDS, embargo int(1.6h)+5):
  trf = train window [TRAIN_START, ts-embargo)   (min MIN_ROWS rows)
  tef = test window  [ts, tend)
  val = last VALID_LEN train dates;  trf_e = trf\val,  vaf = val
  combo = concat([tef, trf_e, vaf])                (single feature matrix; dates disjoint across splits)
  ── GBME   = mean_s FM.gbm(trf_e, combo, cols, s)          (HGBR gamma, seed ensemble)
  ── XGB    = mean_s predict_booster(fit_booster(trf_e,s))  (XGBoost gamma, seed ensemble) — SHARED base
  ── leaves = leaf_matrix(fit_booster(trf_e, seeds[0]))     (n_combo × n_trees int, single booster)
  for scheme in {knn, rfgap, rfgap_kerf}:
     α = fit_alpha on the VAL slice (grid ALPHA_GRID, minimise val QLIKE)
     XGB+<scheme> = clip( smooth_all(XGB base, leaves, dates, scheme, K, α, KERF_FUNC), FL, PRED_CAP )
  train arrays → streamed into _Stream (freed per fold); val/test arrays pooled for DM
pool → _pool_doc → results/gamma_gbm/rfgap_hose_h<h>.json (atomic per fold)
```

## Proximity weightings (the v2 contribution)
For one day's cross-section, `leaf_rows` is (m stocks × T trees) of `pred_leaf` indices.

- **knn (v1):** `day_similarity` = fraction of trees where two stocks share a leaf (sparse one-hot inner
  product `(S Sᵀ)/T`); hard top-`k` neighbour mean. A disjoint stock still gets `k` neighbours (v1 behaviour).
- **rfgap:** each tree `t` puts stock `i` in a leaf; its same-day co-members `C_i(t)` each get `1/|C_i(t)|`
  from that tree (leave-one-out, self excluded). Averaging over trees with ≥1 co-member and renormalising gives
  a **row-stochastic** proximity `W` (`Σ_j W[i,j]=1`, `W[i,i]=0`). `neighbour_mean_i = Σ_j W[i,j] ŷ_j` is the
  tree-ensemble's leave-one-out same-day prediction — a stock never dominates its own smoothing. A fully
  isolated stock (no co-member in any tree) keeps its base prediction (all-zero `W` row).
- **rfgap_kerf:** as rfgap, but tree contribution scaled by `KERF_FUNC(leaf_population)` (`1/pop`/`1/√pop`).
  A large leaf → small weight → down-weighted; stays row-stochastic.

Two implementations, cross-checked in tests (`test_vectorised_neighbour_mean_matches_day_weights`):
- `day_weights(...)` — explicit (m×m) reference matrix (used by tests; exposes the normalisation + KeRF
  down-weight properties).
- `_rfgap_neighbour_mean(...)` — vectorised O(m·T) hot path via a single `bincount` over globally-unique
  `(tree,leaf)` ids computing each stock's leaf group-sum/size and the LOO mean — no (m×m) matrix, the path used
  on the large train/val/test panels.

## Key design decisions
- **Shared XGB base isolates the graph.** All smoothed variants use the identical seed-ensembled base; the only
  difference is the α-weighted proximity, so `XGB+<scheme> vs XGB` measures that scheme's smoothing alone, and
  `XGB+rfgap(+kerf) vs XGB+knn` isolates the v2-over-v1 delta.
- **Streaming train metrics (RAM fix).** v1 accumulated every fold's smoothed TRAIN arrays and OOM'd at h22.
  Here `_Stream` folds each split's train arrays into sufficient statistics (Σsse, Σsae, Σy, Σy², Σqlike) then
  drops them; `finalize()` reproduces `mean_squared_error`/`r2_score`/pooled QLIKE exactly
  (`test_stream_matches_direct_pooled_metrics`). Only the smaller val/test arrays are pooled (needed for DM).
- **One horizon per fresh process** (`--horizon`) for extra isolation from any cross-horizon leak; each JSON is
  a complete 8-fold pool (`n_folds==8` verified before trusting).
- **α fit on val, frozen for test**, `α=0` identity fast path (skips graph build) — the expected NO-GO outcome.
- **Numerical guards:** XGB gamma clipped `[FL, PRED_CAP]`; the smoothed convex combination re-clipped.
- **Capacity matched to GBME:** XGB params single-sourced in `rfgap_config.py` mirror the champion HGBR.

## Gates
- **Simplicity:** no neural net; only XGBoost (already used) + scipy.sparse (installed). Convex-combination
  smoothing + linear-algebra proximity.
- **Anti-abstraction:** reuses `full_matrix` (FM.gbm/panel/load/OWN/EARN/SEEDS/FL), `metrics.per_obs_qlike`,
  `stats.date_clustered_dm`, `overfit_check.classify_fit`, `vn_gbm_graph_stage1.FOLDS/TRAIN_START/_feat`.
- **Performance/batching:** predictions on batched feature matrices; RF-GAP neighbour mean fully vectorised
  (single bincount per day, no Python per-pair loop); α=0 identity fast path.

## Files
| Path | Purpose |
|------|---------|
| `code/rfgap_config.py` | all tunable constants (XGB params, K, schemes, KeRF func, α grid, horizons, gates, spikes) |
| `code/rfgap.py` | booster fit/predict, leaf matrix, Hamming (knn), RF-GAP/KeRF weights + vectorised neighbour mean, smoothing, α fit |
| `code/run_rfgap.py` | walk-forward driver: per-fold predictions, streamed train metrics, pooling, DM, verdict, spike, v2-vs-v1, checkpoints |
| `test/test_rfgap.py` | similarity/kNN/RF-GAP/KeRF/consistency/causality/streaming/driver-logic + run smoke (36 tests) |
| `test/conftest.py` | sys.path bootstrap for dash-named baseline folder |

## Result artifacts
`results/gamma_gbm/rfgap_hose_h{1,5,10,22}.json` — each carries `metrics`/`train_metrics`/`val_metrics`
(5 metrics × 5 models), `fit_diagnostics`, per-scheme `alpha` (per_fold + mean), `dm` (vs_XGB/vs_GBME/vs_knn),
`gain_pct`, `verdict`, `spike_robustness` (per-variant ex-spike), `per_fold_qlike`, and the `v2_vs_v1` decision.
