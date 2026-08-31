# VolGA multi-horizon walk-forward (VN100, clean enriched data) — requirements

## Objective
Extend the expanding-window walk-forward (delivered `baselines/2026-08-30_walkforward_harx_lstm`)
into a **3-model, multi-horizon, VolGA-enabled** experiment on the **clean enriched** VN100 panel.

Models (leave-one-out comparison):
1. **HAR-X** — 5-feature OLS, refit per fold (parsimonious baseline).
2. **LSTM** — no-graph 5-feature LSTM (delivered `MaskedRichNet(use_graph=False)`).
3. **VolGA** — LSTM + 2-hop weighted-GAT over a per-fold **TRAIN-ONLY** vol→PK Top-5 graph
   (delivered `MaskedRichNet(use_graph=True)`, `gat_layers=2` default).

`VolGA − LSTM` isolates the graph's marginal value; both are compared against HAR-X.

## Scope (this pass = B1, CODE only)
Deliver gate-green, working code + a fast smoke run. Do **NOT** launch the multi-hour full sweep
(that is a separate B2 pass). Reuse the 2026-08-30 and 2026-08-21 machinery read-only — do not
reimplement HAR/LSTM/GAT/DM/evidence helpers.

## Input (NEW: clean enriched data — compute-once causal columns)
- `data/processed_enriched/vn100/<ticker>.csv` — PRE-COMPUTED causal columns. The 5 node features are
  read DIRECTLY (no recompute): `[parkinson_variance, har_weekly, har_monthly, market_pk,
  volume_zscore_22]`. Forecast TARGET = `parkinson_variance` at t+h (formed at train time, not stored).
- Node universe: the tickers that survive a fixed 0.80-split node screen (≥ `MIN_TRAIN_ROWS` valid
  train anchors), frozen for all folds — reproduces the delivered 102-node VN100 universe.

## Params (config/CLI so B2 can sweep)
- lookback = 22 (`--lookback`, default 22).
- horizon ∈ {1,5,10,22}, one at a time (`--horizon`).
- 22 folds: `K = ceil(OOS_anchors / 22)` (monthly-style retrain, per
  `docs/reports/2026-08-30_window_fold_lstm_deepresearch.md`; NOT a literal K=22 cadence).
- val tail = `pc.WF_VAL_TAIL` (66); 5 seeds = `pc.SEEDS`; VN100.
- Training: epochs=16 + early-stop + dropout + weight_decay + grad_clip + LR-sched (delivered defaults).
- OUT path encodes the horizon: `results/walkforward_volga/walkforward_volga_vn100_h{H}.json`
  (fixes the delivered bug where OUT was hardcoded to `_h1`).
- CLI: `--horizon --lookback --epochs --smoke --out --no-gpu-wait --batch`.

## Leakage safety (acceptance-critical)
- Per fold, the vol→PK adjacency AND every per-node feature/target scaler are estimated on the fold's
  TRAIN window ONLY, then frozen for val/test forecasting.
- Expanding train, `horizon`-length purge before the forecast block, forecast disjoint from train/val
  (`assert_no_leakage`, reused).

## Performance (CLAUDE.md ENFORCED)
The delivered `run_masked_rich.train_masked_rich` is ALREADY batched: it processes `[B, N, seq, 5]`
tensors on GPU with a mask-aware masked-MSE loss, per-node train-only scalers, `ReduceLROnPlateau`,
early stop, grad-clip; tensors are kept on-device (host↔device copy only at eval). This pass REUSES
that batched trainer unchanged (correctness/no-leakage first) rather than re-batching a batch=1 loop.
The perf conclusion + measured smoke runtime are recorded in the summary report.

## Success criteria / go-no-go
- [ ] Enriched reader returns the 5 features + target with correct shapes + causal NaN handling;
      fails loud if a whole feature is missing for a valid ticker (no all-zero silent degradation).
- [ ] Per-fold graph + scalers depend only on train rows (perturbing future rows leaves them unchanged).
- [ ] All 3 models run on a tiny fixture and produce finite, floored predictions.
- [ ] `--horizon 5` shifts the target by 5 and the OUT filename carries `h5`.
- [ ] Real-data smoke (1–2 folds, 2 epochs, 1 seed, h1, small VN100 slice) produces metrics + fit
      evidence + learning curves.
- [ ] result.json schema carries `train_metrics`/`val_metrics`/`metrics` + `fit_diagnostics` +
      `learning_curves` for `LSTM` and `LSTM_wGAT_vol2pk` (passes the pre-push overfit-evidence gate).
- [ ] Tests pass; C0 line = 100% / C1 branch ≥ 95% on changed lines; `ruff --select F` clean.
- [ ] Normal pre-push gate passes (no QG_SKIP); pushed to origin/master.
- **NO-GO** if any leakage test fails, if a feature is silently zeroed, or if the gate needs QG_SKIP.
