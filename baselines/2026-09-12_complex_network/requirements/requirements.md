# Requirements — Complex-Network Topology for Market-Index Volatility (2026-09-12)

Spec for this baseline. Full protocol and every deviation from the source paper are in
`docs/reports/2026-09-12_complex_network_gbm_design.md` (source of truth). Source paper: Nguyen, Dinh &
Nguyen, "Complex Network Built From Stock Price Returns and Volumes to Predict Market Volatility and Volume,"
*Complexity*, 2026, DOI 10.1155/cplx/5670093.

## Objective

Replicate the paper's complex-network method on our data and test whether the seven global network-topology
metrics carry volatility information, in two experiments:

- **Experiment A (PRIMARY, faithful Section 2.4):** predict the FUTURE market-index volatility (and average
  return, average log volume) from the 7 topology metrics of a rolling combined return+volume network. Models
  = Linear Regression + Random Forest. Score = out-of-sample R² and RMSE under a causal expanding
  walk-forward.
- **Experiment B (secondary, thesis extension):** broadcast the same 7 metrics to every stock and test whether
  they add incremental value to the per-stock gamma-GBM (Parkinson-variance forecast). Score = pooled
  per-observation QLIKE + date-clustered Diebold-Mariano at h ∈ {1,5,10,22}.

## Inputs

- Stock panels (network nodes): `data/processed_enriched/{hose,sp500_clean}/*.csv` via `FM.load(market)`
  (`daily_return`, own-history HAR/RQ/MR block, `parkinson_variance`).
- Real `ln(volume)` for the volume-correlation matrix (Refinement 1, replaces `volume_zscore_22`): raw OHLCV
  `data/raw/prices/{hose_vnstock,sp500_clean}/<TK>_ohlcv.csv` via `volume_io.load_log_volume(market, tickers)`.
- Market index (Experiment A target): `data/raw/prices/_market_index/vnindex.csv` (HOSE) and `gspc.csv`
  (S&P 500) — real, cross-checked series (design doc §3.1).
- Tunable constants: `code/config.py` (WIN=66 headline / WIN_ROBUST=132, STEP=22, L=22, THR=0.5, ALPHA=0.7,
  α-grid, RF_KW, horizons, RAW_VOL_DIR).

## Outputs

- `results/gamma_gbm/complex_network_index_<market>.json` — Experiment A: `n_windows`,
  `n_tickers_per_window` {min,median,max} (Refinement 2), headline per model×target (`r2_oos`, `rmse_oos`,
  `train_r2`, `n_test`, `n_train_final`, `feature_importance` per topology metric — Refinement 4),
  `fit_diagnostics`, `robustness` (α-grid at WIN, win=132 at ALPHA — Refinement 3).
- `results/gamma_gbm/complex_network_<market>.json` — Experiment B: per-horizon GBM vs GBM+topo QLIKE,
  `gain_pct`, `dm_p`, `train_metrics`/`test_metrics`/`fit_diagnostics` (overfit evidence).
- Printed compact tables for both experiments.

## Acceptance criteria

1. Code isolated to this baseline (read-only imports of shared modules); no edits outside it.
2. `pytest test/` green with 100% line / ≥95% branch coverage on changed `code/*.py`.
3. Leakage controls verified (see `code_review/`): causal correlation windows, strictly-future targets,
   target_end embargo in the walk-forward, scaler fit on train only, identical QLIKE floor for the DM test.
4. Both experiments run on HOSE (and S&P 500 if runtime permits) producing the JSON outputs above with real
   numbers — no fabricated values.

## Go / No-Go

- **Experiment A — GO** when a faithful OOS R² for `idx_vol` is produced and reported together with
  `n_windows` and the LR/RF split. A low or negative R² is a valid, reportable outcome (the paper's headline
  used ΔT=125 and the real 2015-2024 VNIndex; our WIN=66 monthly-window sample is smaller). The go criterion is
  that the number is real and honestly reported, not that it matches the paper's 0.56.
- **Experiment B — GO/NO-GO on the science, not the run:** GO only if GBM+topo beats GBM under the
  date-clustered DM test (negative gain and p below the usual threshold) at one or more horizons; otherwise
  NO-GO (topology adds no incremental value on our per-stock QLIKE). Either result is reported honestly.
