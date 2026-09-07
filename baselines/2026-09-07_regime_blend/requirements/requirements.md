# Regime-conditional blend of deep forecast + HAR-X — requirements

## Goal
Fix the h5/h10 QLIKE failure of the deep volatility models (over-forecast of the mean-reverted post-storm
tail — see `docs/reports/2026-09-07_1927_anchor_and_h5h10_failure_analysis.md`) with a causal,
regime-conditional convex blend of the deep forecast and HAR-X, WITHOUT the global HAR-X anchor that
removes the calm-cell edge.

## Inputs
- `results/qlike_anchor/cells/cells_{vn100,vn30}_qlike_none_h{1,5,10,22}.parquet`
  (columns: model ∈ {HAR, HAR-X, LSTM, VolGA}, split ∈ {train,val,test}, fold, ticker, date, y_true, y_pred).
- Deep base: VolGA (paper flagship) and LSTM (secondary). Reference: HAR-X.

## Method (a priori, no test peeking)
- Blend `f = w·deep + (1−w)·HAR-X`, `w = sigmoid(θ0 + θ1·z_level + θ2·z_disagree)`.
- Causal regime features known at forecast origin t−h: `z_level` = standardized `log(HAR-X forecast)`
  (recent-vol regime proxy); `z_disagree` = standardized `log(deep) − log(HAR-X)` (echo signal).
- θ fit PER walk-forward fold on that fold's VALIDATION split (minimize val QLIKE); applied to that fold's
  TEST split with the fold-val standardization. No pooling across folds (avoids look-ahead), no target-day
  info in the gate.

## Output / metrics
- All five metrics (MSE, RMSE, MAE, QLIKE, R²) on test for deep / HAR-X / blend, per market × horizon.
- Date-clustered Diebold–Mariano (`stats.date_clustered_dm`) on per-obs QLIKE: blend vs HAR-X, blend vs deep.
- Val vs test QLIKE (overfit evidence) and per-fold mean blend weight.
- Shared QLIKE positivity floor `pipeline_config.QLIKE_FLOOR` across all compared models.

## Success criteria (go / no-go)
- MUST: blend never worse than the deep model on test QLIKE at any (market, horizon) — the fix cannot harm.
- TARGET (go for full adoption): blend's test QLIKE ≤ HAR-X at h1/h5/h10/h22 on VN100 and VN30, with the
  daily-horizon advantage over HAR-X statistically significant (DM p < 0.05), and no significant loss at
  the longer horizons.
- Honest fallback: if the blend only ties HAR-X at long horizons (expected, since HAR-X is strong there and
  deep ≈ HAR-X), report it as a no-harm calibration improvement, not a claimed long-horizon win.

## Non-goals
- Not re-training any neural model; this is a post-hoc combination layer over frozen forecasts.
- Not the global training-time `anchor=harx` (already shown worse — the negative control).
