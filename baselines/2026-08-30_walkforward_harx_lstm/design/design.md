# Walk-forward HAR-X vs no-graph LSTM (VN100 h1) — design

## Data flow
```
vn100 processed CSVs (104)
   -> MR.build_masked_rich (fixed 0.8/0.1 split, ONCE) -> D.tickers = 102 frozen node universe
   -> build_wf_panel(files, price_dir, lookback=10, horizon=1, keep=102)
        loads wide pk[T,N], market_pk[T], volume_zscore[T,N] (reuse MR helpers, read-only),
        feats[T,N,5] = [har_daily, har_weekly, har_monthly, market_pk, volume_z],
        anchors[A] (FIRST_VALID+lookback-1 .. T-h, kept where >=8 valid nodes),
        win_ok/tgt_ok/node_ok[A,N], dates.
   -> make_folds(A, test_start=int(A*0.9), K=66, val=66, h=1) -> ~7 Fold(train/val/forecast/purge)
        assert_no_leakage(folds, ...) (expanding, purge, disjoint) — raises on violation.
   for each fold f:
     -> pack_fold(panel, fold, lookback, horizon) -> MaskedRichData
          per-node TRAIN-ONLY feature+target scalers; X scaled windows; masks; y; har(3)/har5(5) at t;
          forecast dates d_te; dummy adj=eye(N) (no graph).
     -> HAR-X: OLS on train (5-feat + intercept) -> forecast preds (floored).
     -> LSTM: RMR.train_masked_rich(D_fold, cfg, seed, use_graph=False, adj=eye, return_splits=True)
              for each of 5 seeds -> ensemble forecast preds.
     -> collect per-(node,date) (y,pred) into pooled dicts (HAR, HAR-X, LSTM); fold train/val evidence.
   -> pool over all folds -> _metrics + seed_metric_stats + date-clustered DM (LSTM vs HAR-X, and HAR)
   -> write results/walkforward_harx_lstm/walkforward_vn100_h1.json
```

## Reuse (READ-ONLY imports; never edit live-training-path files)
- `masked_rich` (MR): `_load_wide`, `_volume_zscore_wide`, `N_FEAT`, `FIRST_VALID`, `MaskedRichData`.
- `data_utils` (du): `har_features`.
- `run_masked_rich` (RMR): `MaskedRichNet`, `train_masked_rich`, `_pred_dict`, `_ens`, `_metrics`,
  `_split_metrics`, `_ens_split`, `seed_metric_stats`, `_dm_all`, `OF`.
- `baselines` (B): not needed — HAR-X is a 5-feature OLS built inline (matches the delivered
  `run_masked_rich.run` HAR-X block exactly). HAR (3-feat) via `B.har_fit`/`B.har_predict`.
- `config` (Config): base hyper-params; overridden to epochs=16, patience=5, seeds, batch=32.

`pack_fold` reimplements only the panel-packing math (train-only scalers + windowing) with
configurable fold boundaries — `build_masked_rich` bakes a single fixed fraction split and cannot
express per-fold expanding windows, and it must not be edited. All feature computation and the
model/metric/DM machinery are reused unchanged.

## Fold construction (wf_folds.py)
`make_folds(n, test_start, K, val, horizon)` -> list of `Fold(train, val, forecast, purge)` slices in
anchor-position space. For r in `range(test_start, n, K)`:
- forecast = slice(r, min(r+K, n))
- train    = slice(0, r-horizon-val)
- val      = slice(r-horizon-val, r-horizon)
- purge    = slice(r-horizon, r)

`assert_no_leakage(folds, anchors, dates, horizon)` raises unless, for every fold:
(a) forecast positions disjoint from train and val positions;
(b) train.stop == val.start, val.stop == forecast.start - horizon (purge = horizon);
(c) max val TARGET date < min forecast TARGET date (date-space purge);
and across folds train.stop strictly increases (expanding). This is the executable form of the three
requirements acceptance criteria.

## Over/under-fit evidence
Each fold stamps seed-ensembled train/val/test split metrics + `OF.classify_fit` verdict + per-seed
learning curves (reusing `RMR._split_metrics` / `_ens_split` / `OF.classify_fit`). The JSON carries
per-fold and a pooled-fold-mean verdict. (Path is not `*result.json` under `masked_rich`, so the
push-time overfit gate does not force-check it, but the mandate evidence is captured regardless.)

## Statistics
Pooled date-clustered DM (`RMR._dm_all` -> `stats.date_clustered_dm`) aggregates each loss to one
value per unique OOS date (cross-sectional mean) then runs HLN DM at h=1. This handles the
cross-sectional dependence and the block/temporal structure of the tiled folds; it is NOT a fully
HAC-corrected DM — stated as a caveat in the report.

## GPU politeness
`wait_for_gpu(query_fn, util_max=15, mem_max_mib=1200, hold=3, poll=15)` polls `nvidia-smi` and
returns only after `hold` consecutive samples show util < util_max and mem < mem_max_mib. Called once
at the start of a real run (main). Single process, batch 32, GPU venv `.venv_gpu_encode` (torch 2.6).
Never runs concurrent GPU training.

## SDD gates
- Simplicity Gate: one new baseline folder; no new abstraction beyond the fold/panel/runner split;
  all model + metric code reused. PASS.
- Anti-Abstraction Gate: direct reuse of MR/RMR/du; HAR-X inline OLS (no wrapper). PASS.
- Performance/Batching Gate: LSTM trains on batched `[B,N,seq,5]` snapshots (batch 32) on GPU via the
  reused `train_masked_rich` (no batch=1, no per-item main-thread loop). HAR-X is vectorised OLS.
  Folds are sequential by necessity (expanding window depends on prior data + single shared GPU). PASS.

## Test plan (unique names, no duplicate basenames)
- `test_walkforward_folds.py`: make_folds tiling/expanding; assert_no_leakage passes on valid folds
  and raises on injected overlap / bad purge / non-expanding / date overlap.
- `test_walkforward_panel.py`: pack_fold produces train-only scalers (forecast/val rows excluded from
  the fit — perturb a forecast row, scaler unchanged), correct shapes/masks, date alignment.
- `test_walkforward_runner.py`: run_walkforward on a tiny real VN100 slice with `train_masked_rich`
  stubbed (CPU, no epochs) — asserts pooled metrics, DM, per-fold + evidence blocks, JSON written;
  plus `wait_for_gpu` with a stub query, and `main` dry/stubbed branches.
