# Requirements — leaf-graph v2: RF-GAP / KeRF proximity (principled upgrade of the v1 leaf-Hamming kNN)

## Objective
v1 (`baselines/2026-09-18_gbm_leaf_graph`) showed a small but spike-robust, DM-significant HOSE QLIKE gain from
smoothing the champion gamma-GBM's per-stock predictions over a **hard top-k leaf-Hamming kNN graph** built from
the XGBoost booster's own `pred_leaf` vectors (h1 +0.25% p=4e-4, h5 +0.26% p=3e-5, h10 +0.12% p=0.005; h22 ns).
v2 tests whether two **literature-grounded proximity weightings** improve on that simple graph:

- **RF-GAP** (arXiv 2307.01077): a *proper regression proximity* over the boosted-tree leaves — a leakage-safe
  leave-one-out neighbour distribution, renormalised so each stock's neighbour weights sum to 1 and no stock
  smooths itself. Soft weighted neighbour mean, not a hard kNN.
- **RF-GAP + KeRF large-leaf down-weighting** (arXiv 2601.02735): each tree's contribution scaled by
  `KERF_FUNC(leaf_population)` (`1/pop` or `1/sqrt(pop)`), so a giant storm-day leaf (hundreds of trivially
  "similar" stocks) injects less noise — directly targeting the mandated HOSE spike-robustness.

## Inputs
- HOSE processed-enriched panel: `data/processed_enriched/hose/*.csv`.
- Real crawled VN earnings dates: `results/gamma_gbm/hose_earnings_combined.parquet` (adds `earn_*`).
- OWN-8 own-history features, single-sourced from `baselines/2026-09-13_paper_models/code/config.py`
  (`own_set(FM.OWN)`), plus `FM.EARN` when earnings present.

## Models (per horizon, per fold, seed-ensembled over `FM.SEEDS`)
- **GBME** — canonical HGBR gamma champion (`FM.gbm`).
- **XGB** — plain XGBoost `reg:gamma`, capacity-matched to GBME; **shared base** for every smoothed variant.
- **XGB+knn** — v1 hard top-k leaf-Hamming smoothing (the incumbent champion version, re-run identically here).
- **XGB+rfgap** — RF-GAP proper-proximity soft smoothing.
- **XGB+rfgap+kerf** — RF-GAP + KeRF large-leaf down-weighting.

All smoothed variants: `ŷ_smooth = (1−α)·ŷ_xgb + α·(W ŷ_xgb)`, with `α` fit per scheme on a per-fold val slice,
frozen for test. `W` is the per-day proximity (hard kNN / RF-GAP / RF-GAP+KeRF). Strictly causal: same-day
cross-section + a booster fit on the past train window only.

## Evaluation
- Horizons {1, 5, 10, 22}; expanding walk-forward over `S1.FOLDS`/`TRAIN_START`, embargo `int(1.6h)+5`,
  `MIN_ROWS` gate, val slice = last `VALID_LEN` train dates.
- Metrics MSE/RMSE/MAE/R²/QLIKE (floor `FM.FL`) on train/val/test for every model. **Train metrics streamed**
  (sufficient statistics) so the large train arrays never all live in memory (v1 OOM'd at h22).
- Date-clustered Diebold-Mariano: each smoothed variant vs XGB (isolate the graph), vs GBME (champion), and
  each v2 variant vs XGB+knn (**v2-vs-v1**).
- HOSE regime-spike robustness: per-fold QLIKE + rerun excluding COVID-2020 / 2022 / Apr-2025 windows.
- Over/under-fit evidence (`overfit_check.classify_fit`) stamped per model for the pre-push gate.

## Success criteria (pre-registered)
An RF-GAP(+KeRF) variant **BEATS** v1's XGB+knn on QLIKE at ≥1 horizon (gain>`GAIN_MIN` AND date-clustered
DM p<`DM_ALPHA` vs knn), **OR** is at least as good (gain ≥ 0 vs knn) while **strictly more spike-robust**
(lower ex-spike QLIKE than knn). Otherwise **NO-GO**: v1 stays the champion version, v2 reported as "no
improvement over the simple graph." Honest either way.

## Out of scope
No neural training, no basis-drift risk (minimal convex-combination smoothing only). `archive/` excluded.
No git operations (coordinator pushes).
