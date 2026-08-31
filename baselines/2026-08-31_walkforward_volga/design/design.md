# VolGA multi-horizon walk-forward — design

## Data flow
```
data/processed_enriched/vn100/<ticker>.csv (104 files; pre-computed causal columns)
  -> frozen_universe(files, lookback, horizon):
        build_enriched_panel over ALL tickers, screen nodes with >= MIN_TRAIN_ROWS valid
        TRAIN anchors in the first TRAIN_FRAC region -> 102-node frozen universe.
  -> build_enriched_panel(files, lookback=22, horizon, keep=102):
        READ (no recompute) the 5 node features from the enriched columns:
          [parkinson_variance, har_weekly, har_monthly, market_pk, volume_zscore_{VOLUME_ZSCORE_WINDOW}]
        pk[T,N] = parkinson_variance; feats[T,N,5]; union-of-dates panel (NaN off a ticker's dates).
        Causal NaN handling: leading har_monthly / volume_zscore NaN (first 21) are excluded by the
        anchor start (FIRST_VALID+lookback-1) + win_ok window-validity mask; interior volume_zscore
        NaN on flat-volume windows are imputed to 0.0 (neutral shock) ON A TICKER'S OWN DATES ONLY
        (matches the delivered _volume_zscore_wide fillna(0.0)); off-date NaN stay NaN (masked out).
        FAIL LOUD if har_weekly/har_monthly/market_pk is all-NaN on a valid ticker's own dates, or if
        volume_zscore is all-NaN on own dates for ANY ticker (no silent all-zero feature).
        anchors[A] kept where >= MIN_VALID_NODES valid nodes; target_dates[A] = dates[anchor+horizon].
  -> make_folds(A, test_start=int(A*WF_TEST_FRAC), K=ceil(OOS/22), val=WF_VAL_TAIL, horizon)  [REUSED]
        -> 22 Fold(train/val/forecast/purge); assert_no_leakage(...)  [REUSED] raises on violation.
  for each fold f:
    -> pack_fold(panel, f, lookback, horizon) -> MaskedRichData
         per-node TRAIN-ONLY feature+target scalers; scaled X windows; masks; y=pk[t+h]; har(3)/har5(5)
         at t; forecast dates d_te; adj_vol2pk = _directed_vol2pk on TRAIN rows only (frozen).
    -> HAR-X: _har_ols_preds (5-feat OLS on train, floored)                      [REUSED]
    -> LSTM:  train_masked_rich(D, cfg, seed, use_graph=False, eye, return_splits)  [REUSED] x5 seeds
    -> VolGA: train_masked_rich(D, cfg, seed, use_graph=True,  adj_vol2pk, ...)     [REUSED] x5 seeds
    -> collect per-(node,date) (y,pred) into pooled dicts; per-fold train/val/test fit evidence.
  -> pool over folds -> metrics (test) + per-seed stats + date-clustered DM
       (VolGA vs LSTM = graph marginal; VolGA vs HAR-X; LSTM vs HAR-X; HAR-X vs HAR)
  -> pooled train/val/test metrics + fit_diagnostics + learning_curves (overfit-evidence schema)
  -> write results/walkforward_volga/walkforward_volga_vn100_h{H}.json
```

## Reuse (READ-ONLY imports; never edit the live-training-path files)
- `wf_folds` (2026-08-30): `make_folds`, `assert_no_leakage`.
- `run_walkforward` (2026-08-30): `_har_ols_preds`, `training_config`, `wait_for_gpu`.
- `run_masked_rich` (RMR, 2026-08-21): `MaskedRichNet`, `train_masked_rich`, `_pred_dict`, `_ens`,
  `_ens_split`, `_metrics`, `_split_metrics`, `seed_metric_stats`, `_dm_all`, `OF`.
- `masked_rich` (MR): `MaskedRichData`, `_directed_vol2pk`, `N_FEAT`, `FIRST_VALID`, `EDGE_TOP_K`.
- `pipeline_config` (pc): single source of truth for every tunable (lookback, windows, floors, edge
  params, val tail, test frac, seeds, MIN_* thresholds). NO tunable is hardcoded in this baseline.
- `metrics` (M): mse/rmse/mae/qlike/r2 for the pooled train/val aggregate metrics.

## New code (this baseline)
- `code/wf_enriched_panel.py` — `EnrichedPanel`, `build_enriched_panel`, `frozen_universe`,
  `_fit_scalers`, `pack_fold` (enriched reader + train-only vol→PK adjacency).
- `code/run_volga_walkforward.py` — `VolgaWFConfig`, `run_fold` (3 models), `run_walkforward`
  (pool + evidence + DM + JSON), `_agg_metrics`, `_fit_summary`, `default_out_path`, `main`.

## Design gates (SDD §5)
- Simplicity: no new abstraction; the runner is a thin composition of reused helpers + the enriched
  reader. Only the reader (enriched columns instead of raw recompute) and the 3-model fold assembly
  are new.
- Anti-abstraction: uses the delivered `MaskedRichData` / `MaskedRichNet` / OLS directly.
- Performance/Batching: reuses the already-batched `train_masked_rich` ([B,N,seq,5] on GPU, mask-aware
  loss, per-node train scalers). No batch=1 loop is introduced. Recorded in the report.

## Key decisions
- market_pk is read directly from the enriched column. Verified it is the shared cross-sectional
  factor (identical across tickers on the same date), so reading per-ticker equals the delivered
  broadcast market factor on that ticker's own dates.
- The deep-model keys are named exactly `LSTM` and `LSTM_wGAT_vol2pk` so the pre-push overfit-evidence
  gate (`OF.LEARNED`) recognises them; `LSTM_wGAT_vol2pk` == VolGA.
- Node-universe freeze uses a fixed 0.80-split screen (mirrors delivered `build_masked_rich`'s
  `MIN_TRAIN_ROWS` node drop) so every fold shares one leakage-free node set + OOS region.
