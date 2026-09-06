# Design — Limit-lock hurdle on HAR-X

## Components
### `limitlock_config.py` (baseline-local single source of truth for this baseline's tunables)
No magic numbers scattered in the pipeline modules (§config-hardcode). Reuses `pipeline_config` (`pc`)
for shared constants; defines only the limit-lock-specific ones:
- `LIMIT_FRAC = 0.065` — just under the HOSE +/-7% daily price limit (VN30/VN100 are HOSE).
- `NEAR_LIMIT_MULT = 0.9` — "near-limit" day = `|daily_return| >= LIMIT_FRAC * NEAR_LIMIT_MULT`.
- `LOCK_WINDOW = pc.HAR_MONTHLY_WINDOW` (22) — trailing window for the rolling lock/near-limit freqs.
- `LOCK_THRESHOLDS = (0.5, 0.7, 0.9)` — override-probability sweep.

### `limitlock_features.py` (causal, per-ticker; unit-tested for no look-ahead)
Given a 1-D per-ticker `daily_return` series and its `zero_range_flag` series, compute 3 causal
features per day t (using data up to t only):
- `limit_down_streak`: consecutive days with `daily_return <= -LIMIT_FRAC` ending at t (reset on a
  non-limit-down day).
- `recent_lock_freq`: trailing fraction of `zero_range_flag` over `LOCK_WINDOW` (min_periods=1).
- `near_limit_freq`: trailing fraction of `|daily_return| >= LIMIT_FRAC*NEAR_LIMIT_MULT` over the same
  window.
Prefix-invariance is unit-tested. NaN returns treated as 0 (no move) for the streak / near-limit test;
NaN lock flags treated as 0.

### `run_limitlock_hurdle.py` (walk-forward HAR-X vs HAR-X+hurdle)
Reuse read-only: `pipeline_config`, `run_masked_rich` (RMR: `_pred_dict`, `_metrics`, `_dm_all`,
`_split_metrics`), `wf_folds.make_folds`, `wf_enriched_panel` (`build_enriched_panel`,
`frozen_universe`, `pack_fold`), `run_volga_walkforward` (`VolgaWFConfig`, `enriched_glob`),
`run_walkforward.training_config`.

Per walk-forward fold (same fold machinery as `2026-09-06_regime_features`):
- **Part B (variance):** HAR-X OLS on the 5 HAR-X features (`ols_fit_predict`, floored at
  `qlike_floor`), then clamped at the per-node positivity floor `nfloor = POS_FLOOR_FRAC*t_mean +
  POS_FLOOR_EPS` — identical basis to the hurdle model (H2).
- **Part A (hurdle classifier):** `LogisticRegression` on the 3 causal limit-lock features
  (train-fold masked rows) predicting `zero_range_flag` at the target day t+h. Features standardized
  by train mean/std. Degenerate single-class train target -> constant `P = mean(train target)` (no
  crash, graceful no-op when P<threshold).
- **Combine (`apply_hurdle`):** on test cells with `P(lock) >= threshold`, override the HAR-X
  variance forecast with `nfloor` (the shared positivity floor); else keep HAR-X. Threshold sweep
  `LOCK_THRESHOLDS`.

Scoring (pooled across folds, same as regime baseline):
- `_metrics` (MSE/RMSE/MAE/QLIKE/R2) for HAR-X and each HAR-X+hurdle@thr.
- `qlike_robust`: QLIKE excluding target cells that are lock days (the same lock exclusion idea) —
  isolates the effect on normal days.
- `qlike_lockdays`: QLIKE on lock target cells only — shows the fix magnitude.
- Classifier confusion (TP/FP/FN/TN) over test cells at each threshold.
- Date-clustered DM (`RMR._dm_all`) HAR-X+hurdle vs plain HAR-X per loss family.
- The 2025-04-10 concrete case: whether flagged + before/after per-obs QLIKE contribution.

### Testable-helper extraction (diff-cover)
`run()`/`main()` are `# pragma: no cover` driver glue. All logic in covered helpers:
`ols_fit_predict`, `_har_design`, `logistic_fit_predict_proba`, `apply_hurdle`, `_pooled`,
`_filter_keys`, `_metrics_on`, `_confusion`, `_case_qlike`.

## Gates
- **Simplicity:** OLS + logistic + 3 rolling features; no new heavy dependency (sklearn already used).
- **Anti-abstraction:** reuse delivered panel/fold/metrics directly; no wrappers.
- **Performance/batching:** vectorised lstsq + a single logistic fit per fold on flattened `[A*N,k]`
  arrays; no per-row Python loop over observations. CPU-only by design (GPU is occupied).

## Isolation
Imports delivered baselines read-only; modifies no other baseline or `src/`. Results to the shared
`results/` tree per §3.D.
