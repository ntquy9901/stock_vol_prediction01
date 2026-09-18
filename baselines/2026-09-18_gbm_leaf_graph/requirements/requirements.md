# Requirements — GBME + leaf-cooccurrence graph (falsification)

## Objective
Test whether smoothing the champion gamma-GBM's per-stock volatility predictions over a graph built from the
GBM's OWN gradient-boosting trees (leaf-index cooccurrence) improves out-of-sample QLIKE on HOSE. The graph is
a **target-aware** structure derived from the same 8 own-history features the GBM already uses, so it is
expected to carry no new information (NO-GO). The deliverable value is a clean, gate-clean falsification:
"even a graph derived from the champion GBM's own trees does not help per-stock QLIKE."

## Inputs
- HOSE processed-enriched panel: `data/processed_enriched/hose/*.csv` (per-ticker OHLCV + engineered features).
- Real crawled VN earnings dates: `results/gamma_gbm/hose_earnings_combined.parquet` (adds `earn_*` features).
- OWN-8 own-history feature list, single-sourced from `baselines/2026-09-13_paper_models/code/config.py`
  (`own_set(FM.OWN)`), plus `FM.EARN` when earnings present.

## Models (per horizon, per fold, seed-ensembled over `FM.SEEDS`)
- **GBME** — canonical HGBR gamma champion (`FM.gbm`), the deployed target.
- **XGB** — plain XGBoost `reg:gamma`, capacity matched to GBME (300 trees, lr 0.05, 31 leaves, l2=1,
  min_child_weight 20, `tree_method=hist`). Base predictions shared with the graph model.
- **XGB+leafgraph** — the XGB base predictions smoothed over a per-day kNN leaf-cooccurrence graph:
  `ŷ_smooth,i = (1−α)·ŷ_i + α·mean_{j∈kNN(i)} ŷ_j`, with α fit on a per-fold validation slice, frozen for test.

## Leaf-cooccurrence graph (contemporaneous, causal)
For each day, each stock's leaf-vector = its row of `pred_leaf` from a single train-fitted booster. Two stocks
are neighbours if their leaf-vectors overlap (Hamming similarity = fraction of trees landing in the same leaf).
A per-day kNN graph (k = `K_NEIGHBOURS`) over that similarity feeds the smoothing. Uses only same-day rows +
the train-fitted booster → strictly causal (no future, no cross-day leakage).

## Evaluation
- Horizons {1, 5, 10, 22}; expanding walk-forward over `S1.FOLDS`/`TRAIN_START`, embargo `int(1.6h)+5` days,
  `MIN_ROWS` gate, val slice = last `VALID_LEN` train dates.
- Metrics: MSE, RMSE, MAE, R², QLIKE (floor `FM.FL`) on train/val/test for every model.
- Date-clustered Diebold-Mariano: XGB+leafgraph vs XGB (isolate the graph, primary) and vs GBME (vs champion).
- HOSE regime-spike robustness: per-fold QLIKE + rerun excluding COVID-2020 / 2022 / Apr-2025 windows.
- Over/under-fit evidence (`overfit_check.classify_fit`) stamped per model for the pre-push gate.

## Success criteria (pre-registered kill criterion)
XGB+leafgraph beats XGB at **BOTH** h1 and h5: QLIKE gain > `GAIN_MIN` AND date-clustered DM p < `DM_ALPHA`,
surviving spike-exclusion. Otherwise **NO-GO**. Report the fitted α per fold/horizon (α → 0 ⇒ graph inert).

## Go / No-Go
- GO: both kill horizons beat XGB (unexpected).
- NO-GO (expected): α → 0, or the DM comparison does not clear the bar at h1 and h5.

## Out of scope
No neural training, no basis-drift risk (minimal convex-combination smoothing only). `archive/` excluded.
