# Design — gamma-loss GBM walk-forward

## Data flow
1. `build_enriched_panel` + `frozen_universe` → panel `feats [T,N,5]` (channel 0 = daily Parkinson variance).
2. `extra_feature_panels(feats)` → causal RQ + log-vol decline panels `[T,N]` from channel 0 (all backward-
   looking; `rolling(...,min_periods=1)` / `shift` / `diff`).
3. `make_folds` (canonical folds) + `assert_no_leakage`.
4. Per fold: `pack_fold` → D. Build the `[n_anchor·N, 11]` design (`_design`: 5 HAR + 6 extras at the anchor
   rows); fit `HistGradientBoostingRegressor(loss='gamma')` on masked TRAIN rows (target `pk[t+h]` floored
   positive); predict TEST; floor per node; pool via `RMR._pred_dict`. HAR-X OLS predicted the same way.
5. Pool across folds; `RMR._metrics` + `RMR._dm_all` (GBM vs HAR-X).

## Key decisions
- **Gamma loss = QLIKE.** The gamma unit deviance equals QLIKE up to a constant, so the GBM optimises the
  evaluation metric directly (log-target + squared loss badly under-forecasts the tail → QLIKE blow-up; the
  gamma loss predicts the positive level and is tail-appropriate).
- **Reuse, single-feature-family delta.** HAR-X is the canonical OLS (reproduces exactly); the GBM adds
  nonlinearity + RQ + decline features on the same panel/folds/floor, so the comparison is clean.
- **Causal features only.** RQ and decline features use days ≤ t; extracted at the same anchors as `har5`,
  `nan_to_num`'d.
- **Honest scope.** GBM needs data: it wins on SP500 (large) and overfits VN (small). The baseline reports
  both markets; the SP500 win is the contribution, VN is a documented negative (parsimony wins on small data).

## Gates (SDD)
- Simplicity: one off-the-shelf regressor + engineered features; no bespoke architecture.
- Anti-abstraction: reuses delivered panel/folds/OLS/metrics/DM + sklearn.
- Performance: vectorised feature panels; per-fold GBM fit (CPU, seconds–minutes on SP500); no GPU, no
  per-item loop.

## Caveats (from code review)
- **Objective vs nonlinearity (disclosed).** HAR-X is OLS (squared-error) while the GBM optimises gamma
  deviance ≡ QLIKE, then is judged on QLIKE — so part of the edge could be "trains on the eval metric".
  A control isolates this: a LINEAR gamma model (GammaRegressor) on the same features is **much WORSE** than
  HAR-X OLS (SP500 h5: 0.6466 vs 0.4870, −33%), and adding the extras to it does not help (−34%). Only the
  NONLINEAR GBM wins (+4.63%). So the gain is from the trees' nonlinearity + feature interactions, NOT from
  the gamma objective alone — the objective in a linear model actually hurts. This strengthens the result.
- **Union-calendar rolling.** `extra_feature_panels` computes rolling/shift on `panel.feats[:,:,0]`, which is
  reindexed to the union date grid; for a ticker with listing gaps a "5-day" RQ can span fewer own-trading
  days (NaN-skipped). Still strictly causal; negligible on SP500 (single NYSE calendar) and irrelevant to the
  SP500 win (VN, the market with more gaps, is the one that loses anyway).

## Files
- `code/gamma_gbm_walkforward.py` — extra_feature_panels / _design / _harx_ols / run / main.
- `test/test_gbm.py` — pytest: feature causality, design shape/order, HAR-X reproduction + GBM-vs-HARX sign on
  a real-data smoke (SP500 GBM < HAR-X; VN GBM not required to win).
