# Adversarial code review — VolGA multi-horizon walk-forward (2026-08-31)

Three-lens adversarial pass (Blind Hunter / Edge-Case Hunter / Acceptance Auditor) over the new code
(`code/wf_enriched_panel.py`, `code/run_volga_walkforward.py`) + tests. Reused, already-reviewed
modules (`run_masked_rich`, `masked_rich`, `wf_folds`, `run_walkforward`) are imported read-only and
out of scope. Performance lens included (train/eval code).

## Blind Hunter (hidden bugs)
- **B1 — enriched vol→PK edge uses imputed feature 4.** `pack_fold` builds `adj_vol2pk` from
  `panel.feats[:,:,4]` (volume_zscore, 0-imputed on own dates, NaN off-dates) + `sqrt(pk)`.
  `MR._directed_vol2pk` masks with `np.isfinite`, so off-date NaN never enter; the leading 0-imputed
  rows match the delivered `_volume_zscore_wide` fillna(0) behaviour. VERIFIED semantically identical
  to the delivered edge construction (same `EDGE_TOP_K`, same `EDGE_MIN_PAIRS_DIRECTED` default). No bug.
- **B2 — leakage in the per-fold graph/scalers.** `last_tr_row = tr_anchor[-1] + horizon` bounds the
  edge to train rows; `_fit_scalers` reads only `panel.anchors[fold.train]`. `test_no_lookahead`
  perturbs every row after `last_tr_row` and asserts `adj_vol2pk`, `t_mean`, `t_std`, `X_tr` are
  bit-identical. PASS — no future leakage.
- **B3 — pooled train/val double-counts overlapping expanding windows.** The top-level `train_metrics`
  concatenate each fold's masked train predictions; expanding windows share early rows, so early data
  is weighted more. This block is used ONLY for the over/under-fit *verdict* (evidence schema), never
  for the headline test metric (`metrics`) or the DM test. Accepted as an aggregate-fit signal;
  documented. Per-fold verdicts are also kept in `per_fold` + `fit_summary`.
- **B4 — ensemble method consistency.** Per-fold evidence uses `_ens_split` (mean of raw arrays);
  headline test metric uses `_ens` (mean of per-(node,date) dict values). Both are seed-means of the
  same predictions and coincide numerically. No divergence.

## Edge-Case Hunter
- **E1 — too-few valid nodes.** `MIN_VALID_NODES=8`; a panel where no anchor reaches 8 valid nodes now
  raises `build_enriched_panel: no anchor has >= 8 valid nodes` (added guard) instead of an opaque
  empty-`np.stack`/`make_folds(n=0)` failure. Tested (`test_build_raises_when_no_anchor...`).
- **E2 — missing enriched column** → `ValueError: ... missing columns`. Tested.
- **E3 — whole feature missing for a valid ticker** (all-NaN har_weekly/monthly/market_pk/volume on own
  dates) → fail-loud in `_check_feature_coverage` (no silent all-zero degradation, per CLAUDE.md). Tested
  directly (crafted arrays). NOTE: one structurally-broken ticker aborts the whole `frozen_universe`
  build — this is the intended fail-loud contract (P1–P6 pipeline must fix the offending ticker), not a
  silent skip.
- **E4 — keep-ticker with no file / <2 tickers / rejection files.** Filtered / raised; all covered.
- **E5 — horizon plumbing.** `--horizon 5` shifts the target to `pk[t+5]` and drops 4 more trailing
  anchors; `default_out_path(5)` → `..._h5.json`. Tested (fixes the delivered hardcoded-`_h1` OUT bug).
- **E6 — test_start / K.** `K=max(1, ceil(OOS/folds_target))` guards div; `make_folds` guards
  `0<test_start<n`. Both `test_start` branches (None vs explicit) and both `out_path` branches
  (write vs None) are exercised.

## Acceptance Auditor (vs requirements.md)
- 5 features read DIRECTLY (no recompute); target `parkinson_variance` at t+h formed at train time. ✓
- Causal NaN handling (leading excluded by anchor start + win_ok; interior volume_zscore imputed to
  neutral 0 on own dates only). ✓
- 3 models HAR-X / LSTM / VolGA; VolGA = `use_graph=True`, 2-hop GAT (MaskedRichNet default
  `gat_layers=2`), leave-one-out `VolGA − LSTM` + both vs HAR-X. ✓
- Per-fold TRAIN-only vol→PK graph + scalers, frozen for val/test. ✓
- CLI `--horizon --lookback --epochs --batch --folds-target --smoke --out --no-gpu-wait`; OUT encodes
  horizon. ✓
- 22 folds via `K=ceil(OOS/22)` (measured OOS=454 → K=21 → 22 folds; universe=102, matches delivered). ✓
- Over/under-fit evidence schema (`train_metrics`/`val_metrics`/`metrics` + `fit_diagnostics` +
  `learning_curves` for `LSTM` + `LSTM_wGAT_vol2pk`) recognised by `check_overfit_evidence`. ✓
- Tests (a)–(f) present + real-data-sample smoke. ✓

## Performance lens
- Reuses the delivered **already-batched** `train_masked_rich`: `[B, N, seq, 5]` tensors on GPU,
  mask-aware masked-MSE loss, per-node train-only scalers, `ReduceLROnPlateau`, early stop, grad-clip;
  tensors stay on-device (host↔device copy only at eval `infer`). No batch=1 loop is introduced.
  Measured: one full-size fold (102 nodes, 297,240 masked train obs, 16 epochs, 1 seed, LSTM+VolGA) =
  61.4 s on RTX 4060 → full run ≈ 22 folds × 5 seeds × ~60 s ≈ **~1.9 h per horizon**. Feasible for B2.

## Verdict
No CRITICAL/MAJOR findings. B3 (pooled-train double-count for the verdict only) documented as an
accepted simplification. Coverage on the two changed source modules: C0 line = 100%, C1 branch = 100%.
Ready for commit + the B2 full sweep.
