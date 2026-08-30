# Code review — walk-forward HAR-X vs no-graph LSTM (2026-08-30)

Adversarial 3-lens review (Blind Hunter / Edge-Case Hunter / Acceptance Auditor) plus a dedicated
**Leakage lens** (the highest-risk axis for a walk-forward). Scope: `code/wf_folds.py`,
`code/wf_panel.py`, `code/run_walkforward.py` + `test/`. Live-training-path files are imported
read-only and out of scope.

## Leakage lens (primary)
- **Train-only scalers.** `wf_panel._fit_scalers` fits per-node target + 5-feat scalers on
  `panel.node_ok[fold.train]` rows only; forecast/val anchors never enter the fit. Verified by
  `test_pack_fold_real_slice_train_only_scaler` and the scratch check (perturbing a forecast-region
  feature leaves `t_mean`/`f_mean` unchanged). PASS.
- **Purge between val (early-stop) and forecast (scored).** `assert_no_leakage` enforces
  `val.stop == forecast.start - horizon` and, in DATE space, that every train/val target date strictly
  precedes every forecast target date. So no training/val TARGET reaches the scored OOS region.
  Verified by `test_assert_no_leakage_*`. PASS.
- **Causal features.** `market_pk` = cross-sectional median of `sqrt(pk)` at day t (no future);
  `volume_zscore` = trailing rolling z-score; HAR = trailing rolling means; all reused unchanged from
  the delivered `masked_rich`. PASS.
- **Frozen forecast.** Each fold trains on `[0, r-h-val)`, freezes, then predicts every forecast anchor
  from its own window ending at t (1-step-ahead, features <= t). PASS.
- **MINOR (documented, accepted).** Per the approved design, train and the val tail are contiguous
  (no `horizon` purge BETWEEN train and val). The val tail is used only for LSTM early-stopping, not
  scored; the scored forecast region is fully purged from both train and val. Impact is limited to a
  marginal early-stop-selection effect on the boundary val day and does not touch the OOS metrics or
  the DM. Left as-is to match the approved design spec (`train=[0,r-h-val)`, `val=[r-h-val,r-h)`).

## Blind Hunter (hidden bugs)
- **Identical floors across models (prior H2 bug).** `cfg.qlike_floor` (1e-8) is the single floor for
  every `_metrics`/`_dm_all` call; the per-node positivity floor `1e-2*t_mean+1e-12` is applied to
  HAR/HAR-X here and is the SAME floor `train_masked_rich` applies to the LSTM (zscore_floor path).
  HAR-X and LSTM are floored identically within each fold. PASS.
- **Pooling correctness.** LSTM seed dicts are pooled per seed across folds then `_ens`-averaged; all
  models share `(node, date)` keys (same `tmask_te`), so DM aligns on identical points with identical
  true `y`. PASS.
- **Per-fold floor drift.** `t_mean` is refit per fold, so `nfloor` differs across folds — but within a
  fold BOTH compared models use the same fold `t_mean`, and pooled DM compares per-obs losses that each
  used their own fold's identical-across-models floor. Fair. PASS.

## Edge-Case Hunter
- Empty split (`len(aa)==0`) → zero-length arrays (`test_pack_fold_empty_split...`). PASS.
- Node with no valid train rows → neutral scaler `(0,1)` (`test_fit_scalers_node_without_train_rows...`).
  In the real VN100 run every node has ample history (fold 0 already trains on ~90% of anchors). PASS.
- Short final fold (`min(r+K, n)`); degenerate empty-forecast fold skips the date check without crash.
  PASS.
- Bad params (`test_start` out of range, non-positive K/val/horizon, empty train window) raise
  `ValueError`. PASS.

## Acceptance Auditor (vs requirements.md)
- OOS == the delivered fixed-split test region: independent rebuild reproduces 102 nodes and the exact
  454-date set (`match delivered = True`). PASS.
- K=66, expanding train, val=66, purge=h, h=1, epochs=16, 5 seeds, lookback=10, batch=32, HAR-X OLS,
  no-graph LSTM: all implemented and asserted. PASS.
- Over/under-fit evidence per fold (train/val/test metrics + `classify_fit` verdict + per-seed learning
  curves) + a pooled `fit_summary` roll-up. PASS.
- Pooled OOS forecast dates == the tiled union of fold forecast blocks (`n_oos_dates` check). PASS.

## Performance lens
- LSTM trains on batched `[B,N,seq,5]` snapshots (batch 32) on GPU via the reused `train_masked_rich`
  (no batch=1, no per-item main-thread loop). HAR/HAR-X are vectorised OLS. Folds are necessarily
  sequential (expanding window + single shared GPU). GPU politeness: `wait_for_gpu` holds until the GPU
  is free (util<15, VRAM<1200MiB, 3 consecutive samples). PASS.

## Tests / coverage
- 27 tests pass; 100% line + 100% branch coverage on the three changed modules under the GPU venv
  (`--cov-branch`). Ruff `--select F` clean (and full ruff clean). Real-data-sample smoke present
  (`test_build_wf_panel_real_slice_shapes`, `test_pack_fold_real_slice_train_only_scaler`).

## Verdict
No HIGH/MAJOR findings. One documented MINOR (train↔val contiguity, early-stop-only, per approved
design) accepted. Ready for the full run + report.
