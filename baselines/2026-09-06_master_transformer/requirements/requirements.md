# MASTER (Market-Guided Stock Transformer) for daily Parkinson-variance forecasting

**Baseline:** `2026-09-06_master_transformer`
**Source model:** MASTER, Li et al., AAAI 2024, arXiv:2312.15235; reference code SJTU-DMTai/MASTER (MIT), vendored in `_master_ref/`.

## Objective
Adapt the MASTER architecture (market-guided feature gate + intra-stock temporal attention + **dense**
inter-stock self-attention across the whole cross-section + temporal aggregation + linear decoder) to
forecast one-step / multi-step daily **Parkinson variance** (`parkinson_variance`, i.e. sigma^2) on the
VN30 and VN100 panels, and honestly measure whether its dense cross-sectional attention beats the
existing baselines HAR-X (OLS), no-graph LSTM, and VolGA (LSTM + sparse horizon-matched vol->PK GAT).

The strong prior on this project is that graph/attention has repeatedly failed to beat HAR-X on the
noisy daily target and VolGA only wins at h1. A NEGATIVE result (MASTER also fails to beat HAR-X, or
its dense attention does not beat VolGA's sparse edge) is a valid, publishable finding. The narrative is
not to be optimised.

## Inputs
- Enriched causal panels `data/processed_enriched/{vn30,vn100}/<ticker>.csv` with the 5 node features
  `[parkinson_variance, har_weekly, har_monthly, market_pk, volume_zscore_22]` (read via the delivered
  `wf_enriched_panel.build_enriched_panel`, READ-ONLY).
- Per anchor (trading date t) the MASTER sample is `[N, T, D]`:
  - N = stocks in the frozen universe, T = lookback (10, canonical), D = 5 stock features + M market features.
  - Stock features are the per-node train-only standardized features from `pack_fold` (`D.X_*`).
  - M = 4 causal market features broadcast across stocks (cross-sectional mean/dispersion of Parkinson
    variance, mean volume z-shock, causal rolling z of market vol) — the gate input at the last timestep.
- Target = `parkinson_variance` at t+h per stock, masked to valid stocks (`tmask`).

## Outputs
- `results/master_transformer/master_<market>_h<h>.json` per (market, horizon): model/config, git commit,
  panel, horizon, date ranges, ticker-date + unique-date counts, metrics (MSE/RMSE/MAE/QLIKE/R2 +
  non-lock conditional QLIKE + win-rate vs HAR-X), DM (date-clustered) MASTER vs HAR-X / LSTM / VolGA,
  train/val/test fit evidence + learning curves, seeds/device/versions.
- `docs/reports/2026-09-06_master_transformer_report.md` (method, [N,T,D]+gate adaptation, results table,
  overfit evidence, limitations, GO/NO-GO). Objective style.

## Success criteria / go-no-go
- **Fairness (mandatory, not a "win" criterion):** MASTER, HAR-X, LSTM, VolGA compared on the SAME
  walk-forward folds, seeds, lookback (lb10, folds_target=7), per-node train-only scalers, and the SAME
  shared QLIKE positivity floor. Leakage guards (`assert_no_leakage`) pass. Per-node scalers fit on TRAIN
  only. No Softplus/ReLU forcing positivity — linear output + inverse-transform + positivity floor
  (proven pattern). If any of these fail the result is BLOCKED, not reported.
- **GO** if MASTER beats HAR-X on pooled QLIKE with a date-clustered DM p < 0.05 in favour of MASTER on
  at least the h1 cross-sectional-structure horizon on at least one panel, AND its dense attention beats
  VolGA (DM favours MASTER). Otherwise **NO-GO** (report honestly, including mixed results and exact
  p-values).
- Overfit gate: every learned model (MASTER, LSTM, VolGA) result carries train/val/test metrics +
  fit_diagnostics + learning_curves, and each verdict is `ok` (not over/under-fit).

## Non-goals
- SP500 (heavy, runs on Colab A100) — skip locally, note it.
- Beating the state of the art on stock RANKING (MASTER's original task); here the task is variance
  regression with QLIKE, a different objective.
