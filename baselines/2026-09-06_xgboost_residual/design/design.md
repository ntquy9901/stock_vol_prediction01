# Design — HAR-X + XGBoost residual-ratio

Plan artifact (CLAUDE.md §5 SDD). Passes the three SDD gates:

- **Simplicity gate:** no new project/abstraction; reuses the delivered panel/fold/HAR-X/metric/DM
  machinery read-only. Only new code = causal feature builders, chronological OOF HAR-X, and the
  XGBoost fold loop + ladder.
- **Anti-abstraction gate:** uses `xgboost.XGBRegressor` directly (no wrapper), numpy/pandas directly.
- **Performance/batching gate:** XGBoost fits are batched matrix ops on `tree_method='hist'` with bounded
  `n_jobs` (CPU); one deterministic seed (tree model — no multi-seed loop needed). No per-item Python
  training loop. Grid is a small curated candidate list (not a full cross-product sweep).

## Target audit

`data/processed_enriched/{vn30,vn100}/<ticker>.csv` column 7 = `parkinson_variance`. The delivered
`wf_enriched_panel` docstring states it is a VARIANCE (σ²); `pipeline_config`/`masked_rich` treat the
Parkinson column as variance (√pk used for the edge). Confirmed: target is Parkinson **variance**. We do
NOT switch to volatility. Horizon target = point `pk[t+h]` (the delivered panel forms `y = pk[t+h]`, not a
mean-over-window — verified in `pack_fold`/`build_enriched_panel`). Tests pin this alignment.

## Data flow (per panel × horizon)

```
enriched CSVs ──> frozen_universe (train-frac screen, delivered)
             ──> build_enriched_panel (delivered): panel.pk [T,N] variance,
                     panel.feats [T,N,5]=[pk,har_w,har_m,market_pk,vol_z], anchors, target_dates
read_aligned_columns (new, tested): daily_return, log_range, volume, zero_range_flag -> [T,N] each
build_sm_features (new): per-ticker causal stock features + cross-sectional market features -> F_sm [T,N,Fsm]
make_folds (delivered, lb10/folds_target=7) + assert_no_leakage (delivered)
for each fold:
    pack_fold (delivered): D with har5_tr/va/te, y_tr/va/te, tmask_*, d_va/d_te, t_mean
    _har_ols_preds (delivered): HAR + HAR-X floored train/val/test forecasts  (== edge_hmatched HAR-X)
    directed_vol2pk_hmatched (delivered edge_hmatched, read-only): train-only adjacency A [N,N]
    build_graph_features (new): neighbor_pk / neighbor_shock / neighbor_return = A @ base[t] -> G [T,N,3]
    oof_harx (new, tested): chronological expanding-window HAR-X over TRAIN anchors -> harx_oof [Atr,N]
    residual target z_tr = log((y_tr+eps)/(harx_oof+eps)) on train cells with valid OOF
    for model in {direct, base(stock), +market, +graph}:
        tune XGBoost on val (small grid, early stopping) by val QLIKE of reconstructed forecast
        pick alpha-shrinkage / z-clip guardrail on val (residual models); direct uses log target
        forecast test -> yhat, floored with shared nfloor -> pooled _pred_dict
pool over folds -> metrics (QLIKE/MSE/RMSE/MAE/R2) + non-lock/lock-only/top-1% + win-rates
DM (date-clustered, delivered RMR._dm_all) for predeclared comparisons
overfit evidence: train/val/test fit metrics + fit_diagnostics for the XGBoost learner
result JSON -> results/xgboost_residual/xgb_<market>_h<h>.json
```

## Causal features (all ≤ feature_date t)

- **Stock** (per-ticker on its OWN trading series, reindexed to the union calendar — same convention as the
  delivered per-ticker `har_weekly/har_monthly`): `pk` at t and lags 1,2,3,5,10,22; rolling mean 5/10/22/44;
  rolling std 22; rolling median 22 + MAD 22; EWMA span 5 and 22; ratio `pk_t/(roll_mean_22+eps)`; rolling
  linear slope over 22; `daily_return`, `abs_return`, negative-return indicator; `log_range`; reused
  `har_weekly`, `har_monthly`, `volume_zscore`.
- **Market** (cross-sectional at date t over valid nodes, broadcast to all nodes): reused `market_pk`; cross-
  sectional mean/std/MAD of pk; market return (cs mean daily_return); fraction up / fraction down; market
  volume shock (cs mean volume_zscore).
- **Graph** (per fold, train-only adjacency `A[target j, source i]`): `neighbor_pk[j,t]=Σ_i A[j,i]·pk_i(t)`,
  `neighbor_shock[j,t]=Σ_i A[j,i]·vol_z_i(t)`, `neighbor_return[j,t]=Σ_i A[j,i]·ret_i(t)`. `A` is built with
  `last_row = last_train_anchor + horizon` (train-only) — graph-cutoff test pins this.

NaN in long-lag features on early rows is left as NaN (XGBoost `hist` routes NaN natively); masked
(invalid) cells are excluded before fitting.

## Leakage-safe OOF HAR-X residual target

`oof_harx(har5_tr, y_tr, mask_tr, cfg)`: TRAIN anchors are chronological. After a warm-up prefix
(`OOF_WARMUP_FRAC`), split the remainder into `OOF_SPLITS` contiguous blocks; for each block fit a 5-feature
OLS on all valid cells with anchor index **strictly before** the block start and predict the block's valid
cells. Warm-up rows get no OOF prediction and are excluded from residual training. Final forecast on
val/test uses the fold-train HAR-X (`harx["va"]`/`harx["te"]`), consistent with the walk-forward protocol.
`yhat = HARX_forecast · exp(alpha·clip(zhat, -c, c))`, floored. Cutoff test verifies each OOF prediction
used only earlier anchors.

## Hyperparameters (val-only)

Small curated candidate list `XGB_GRID` (config); per model per horizon, fit each candidate with early
stopping on val (RMSE of the training target), reconstruct the val forecast, select by **val QLIKE**.
Guardrail `alpha ∈ ALPHA_GRID` (shrinkage) and `c ∈ CLIP_GRID` (z-clip) chosen on val QLIKE; if val picks
`alpha=0` the correction adds no value (reported). One fixed seed (`XGB_SEED`); determinism verified by test.

## Metrics / inference

Standard QLIKE (all obs, primary) + MSE/RMSE/MAE/R2; non-lock conditional QLIKE (exclude lock cells: target
≤ `LIMIT_LOCK_MULT·floor` OR `zero_range_flag` at target date), lock-only QLIKE, lock QLIKE share, top-1%
date contribution; ticker win-rate and unique-date win-rate vs HAR-X. DM = date-clustered (RMR._dm_all) for:
XGB-residual vs HAR-X; +market vs base; +graph vs +market; direct-XGB vs HAR-X. VolGA vs HAR-X cited from
delivered `edge_hmatched` DM; VolGA vs XGB+graph reported as pooled point estimates (no cell-level DM — VolGA
cells need GPU). April-2025 + top-1% + non-lock robustness reported. Unadjusted multiple comparisons labelled
**exploratory**.

## Files

- `code/xgb_config.py` — all tunables (windows, floors, grids, seed, n_jobs, OOF params, lock mult).
- `code/xgb_features.py` — `read_aligned_columns`, `build_sm_features`, `build_graph_features`, helpers.
- `code/xgb_oof.py` — `oof_harx`, `residual_target`, `reconstruct`.
- `code/xgb_model.py` — `tune_direct`, `tune_residual` (grid + early stopping + guardrail on val).
- `code/run_xgboost_residual.py` — fold loop + ladder + metrics + DM + JSON (thin run/main pragma'd).
- `code/summarize_xgb.py` — summary/ablation/lock robustness tables + SHAP/permutation figure from JSONs.
- `test/test_*.py` — causality/alignment/OOF-cutoff/graph-cutoff/split-isolation/count/reproducibility/HAR-X
  parity with edge_hmatched/positive-forecast/lock-mask tests + a real-data smoke.

## Gate notes

- Overfit evidence emitted for the learned XGBoost models (`train_metrics`/`val_metrics`/`fit_diagnostics`);
  `check_overfit_evidence` detects `xgb` as learned.
- No config-hardcode: every tunable in `xgb_config.py`. No pragma-hidden logic: real logic in tested
  helpers; only argparse `main()` + data/JSON I/O glue carries `# pragma: no cover`.
