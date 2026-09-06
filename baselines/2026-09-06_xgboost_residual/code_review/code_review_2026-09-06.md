# Adversarial code review — XGBoost residual-ratio study (2026-09-06)

Self-review across three lenses (Blind Hunter / Edge-Case Hunter / Acceptance Auditor) per CLAUDE.md.
The coordinator is expected to run the full 3-layer `/code-review` before commit; this documents the
issues examined and resolved during development.

## Leakage lens (highest priority for a forecasting study)

| # | Check | Status |
|---|---|---|
| L1 | Rolling/lag features use only ≤ t | PASS — `test_pk_lag_and_rollmean_are_backward_looking`, `test_no_lookahead_future_change_leaves_feature_unchanged`; per-ticker on own series (no calendar-gap bleed). |
| L2 | Market features cross-sectional at t (no future) | PASS — `test_market_features_are_cross_sectional_at_t`. |
| L3 | Graph adjacency built train-only | PASS — `last_tr_row = last_train_anchor + horizon`; delivered `directed_vol2pk_hmatched` reused unchanged. Neighbour features use same-day pk/shock/return (known at t) through the fixed train-only adjacency. |
| L4 | OOF HAR-X residual target uses strictly-earlier fits (not in-sample) | PASS — `test_each_block_uses_only_strictly_earlier_anchors`, `test_future_target_change_does_not_move_earlier_oof`. |
| L5 | Preprocessing / HAR-X coef / hyperparams / guardrail fit on train+val only | PASS — HAR-X via delivered `_har_ols_preds` (train rows); grid + alpha/clip selected on val QLIKE; test never enters selection. |
| L6 | Calendar-date split isolation, purge = horizon | PASS — delivered `make_folds` + `assert_no_leakage` called in `run()`. |
| L7 | Shared positivity floor identical across models | PASS — `nfloor = POS_FLOOR_FRAC*t_mean + POS_FLOOR_EPS` + `qlike_floor` applied to HAR/HAR-X and every XGB forecast; `test_harx_pooled_parity_with_edge_hmatched` confirms the HAR-X baseline equals the delivered edge_hmatched HAR-X (common ground). |

## Edge-case lens

- OOF with too few train anchors → all-NaN (no crash); block skipped when <6 earlier fit rows or no valid
  target cells — `test_block_skipped_*`, `test_too_few_anchors_returns_all_nan`.
- Degenerate logistic/OLS: OLS via `lstsq` (rank-deficient safe). XGBoost handles NaN features natively
  (`hist`); no `nan_to_num` on the design that would fabricate a 0 signal.
- Empty pooled subsets (non-lock / lock-only / top-1% removing all dates) → `None` not a divide-by-zero —
  `test_empty_dict_guards`, `test_exclude_all_dates_returns_none`.
- Identical forecasts (e.g. two residual models both selecting alpha=0) → `_dm_all` catches the
  non-positive-variance DM error and records `{"error": ...}` rather than crashing; summary prints `p=None`.
  Documented limitation, not a fault.
- Counts never conflate ticker-date with unique-date (`count_summary`, reported both) —
  `test_count_summary_distinguishes_ticker_date_from_dates`.

## Acceptance lens (guide requirements)

- Target = Parkinson VARIANCE, unchanged; horizon target = point `pk[t+h]` (delivered panel). Confirmed in
  design.md target audit.
- Model ladder (direct, residual base/+market/+graph), val-only tuning, alpha/clip guardrail (alpha=0
  reported as "no correction") — `test_residual_alpha_zero_when_residual_is_pure_noise`.
- Metrics: standard QLIKE (primary, all obs) + MSE/RMSE/MAE/R2 + non-lock conditional / lock-only /
  lock share / top-1% exclusion + ticker & unique-date win rates; DM date-clustered for the predeclared
  comparisons; April-2025 shock trace. VolGA vs HAR-X cited from delivered edge_hmatched DM.
- Reproducibility: fixed `XGB_SEED`, deterministic tree fit — `test_direct_reproducible_same_seed`,
  `test_residual_guardrail_in_grid_and_reproducible`.
- Overfit evidence: train/val/test fit metrics + `fit_diagnostics` for each XGB learner; the pre-push gate
  detects `xgb` (pattern added to `overfit_check._LEARNED_PATTERNS`).

## Performance lens

- CPU-only, `tree_method='hist'`, bounded `n_jobs=4` (does not starve the host while the GPU agent runs).
  One deterministic seed (tree model) — no multi-seed loop. XGBoost fits are batched matrix ops; the grid
  is a curated 5-candidate list, not a full cross-product sweep. No per-item Python training loop.

## Findings requiring the reader's awareness (accepted limitations, not defects)

1. **VolGA / LSTM not retrained** (CPU-only mandate; GPU in use). Their pooled QLIKE and the
   VolGA-vs-HAR-X DM are cited from the delivered `results/edge_hmatched/*.json` (same lb10/folds_target=7
   split). A cell-level DM of VolGA vs XGB+graph is NOT run (needs VolGA per-cell predictions from GPU);
   reported as pooled point estimates only.
2. **Multiple comparisons** across panels × horizons × models are UNADJUSTED and labelled *exploratory*
   in the report; GO/NO-GO leans on the predeclared primary comparison + cross-fold consistency.
3. **Gate file touched**: `scripts/quality_gate/overfit_check.py` gained `xgb`/`catboost`/`lightgbm` to
   its learned-pattern list (+ regression test). This is the single deviation from strict "modify no src
   file" isolation; it makes the gate genuinely enforce over/under-fit evidence for tree learners (which
   the task presupposes). Flagged for coordinator sign-off.

No critical/major defects found. Minor items above are documented behaviours.
