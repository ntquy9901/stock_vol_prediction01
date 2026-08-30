# Walk-forward HAR-X vs no-graph LSTM (VN100 h1) — requirements

## Objective
Answer one question: does **periodic retraining** (expanding-window walk-forward) change the
fixed-split verdict that the no-graph LSTM is significantly **worse** than HAR-X on QLIKE?

Fixed-split reference (delivered `results/masked_rich_floor1e2/vn100_h1/result.json`, VN100 h1,
102 nodes, 454 test dates):
- LSTM QLIKE 0.5784, HAR-X QLIKE 0.5115 (5-seed ensemble).
- Date-clustered Diebold-Mariano LSTM-vs-HAR-X on QLIKE: p = 1.14e-3, favours HAR-X (mean_diff +0.0668).

(The task brief quotes an older fixed-split LSTM QLIKE of 0.6229; the report cites the current
delivered `result.json` value 0.5784 and the same DM p = 1.1e-3 verdict.)

## Input
- Processed VN100 Parkinson-variance panel: `submission/soict_lstm_gat/data/vn100/*_processed.csv`
  (104 files; columns `date, parkinson_volatility` = VARIANCE sigma^2).
- Raw OHLCV for the volume feature: `data/raw/prices/vn100_vnstock/*_ohlcv.csv`.
- Node universe = the **102 tickers** that the delivered fixed-split `build_masked_rich` keeps
  (obtained once from the fixed 0.80/0.10 split, then frozen for all folds — identical universe,
  identical OOS region).

## Output
- `results/walkforward_harx_lstm/walkforward_vn100_h1.json`:
  pooled-OOS QLIKE/MSE/RMSE/MAE/R2 for HAR-X and LSTM (ensemble + per-seed), date-clustered DM
  (LSTM vs HAR-X) on the pooled walk-forward OOS, per-fold breakdown, and over/under-fit evidence
  (train/val/test metrics + fit verdict + learning curves per fold).
- `docs/reports/2026-08-30_walkforward_harx_lstm.md`: the headline — walk-forward DM vs fixed-split
  DM side by side, and the one-sentence answer.

## Design parameters (approved — implement exactly)
- OOS region = the delivered fixed-split TEST region (last 10% of anchors = 454 dates).
- Retrain cadence K = 66 trading days; ~7 folds tiling the OOS region contiguously.
- Per fold r (anchor position): train = `[0, r-h-val)`, val tail = `[r-h-val, r-h)` (66),
  purge gap h between val and forecast, forecast = `[r, r+K)`.
- h = 1, val = 66, epochs = 16, patience 5, seeds {42,123,2026,7,2024} (LSTM 5-seed ensemble),
  lookback = 10, batch = 32.
- Per-ticker feature + target scalers **refit TRAIN-ONLY each fold**.
- HAR-X: 5-feature OLS closed-form refit each fold. LSTM: delivered no-graph MaskedRichNet
  (`use_graph=False`), 16 epochs, early-stop on the val tail, 5 seeds ensembled.
- QLIKE floor IDENTICAL across HAR-X and LSTM (`cfg.qlike_floor`); per-node positivity floor
  `1e-2 * t_mean + 1e-12` for both (mirrors the delivered runner; prior H2 bug).

## No-leakage acceptance criteria (testable)
1. No forecast-region target date appears in any fold's train or val target dates.
2. The fold scalers are fit on TRAIN rows only (forecast/val anchors never enter the scaler fit).
3. Fold train windows are strictly increasing (expanding) and end at least `h+val` positions
   before their forecast start (purge gap = h).

## Success criteria / go-no-go
- All fold + leakage + runner unit tests pass (pytest, unique test names, stubbed training on a
  tiny real slice).
- Over/under-fit evidence captured per fold and pooled; fit verdict recorded.
- Pooled OOS forecast dates == the 454 fixed-split test dates (directly comparable).
- Report states, honestly, whether walk-forward narrows/flips or preserves the HAR-X advantage.
- Full pre-push gate green (C0 line 100% / C1 branch >=95% on changed lines, ruff --select F clean);
  commit + push to origin/master.

## Non-goals
- No graph (no-graph LSTM only). No hyper-parameter search. No other panels/horizons.
