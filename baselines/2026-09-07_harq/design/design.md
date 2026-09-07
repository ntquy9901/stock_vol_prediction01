# Design — HARQ walk-forward

## Data flow
1. `build_enriched_panel(files, LOOKBACK, horizon, frozen_universe(...))` → panel with `feats [T,N,5]`
   (channel 0 = daily Parkinson variance), `anchors`, `pk`, `target_dates` (delivered, unchanged).
2. `quarticity_panel(feats)` → `harq [T,N] = feats[:,:,0] * sqrt(rolling5(feats[:,:,0]^2))` — causal, reuses
   only channel 0.
3. `make_folds(...)` → the exact canonical expanding-window folds; `assert_no_leakage`.
4. Per fold: `pack_fold` → `D` (train-only scalers). Fit HAR-X OLS on `D.har5_tr` (masked train rows) and
   HAR-X-Q OLS on `[D.har5_tr, harq@train-anchors]`; predict test; floor per node; pool via `RMR._pred_dict`.
5. Pool across folds; `RMR._metrics` (5 metrics) + `RMR._dm_all` (date-clustered DM, HAR-X-Q vs HAR-X).

## Key decisions
- **Reuse, don't reimplement.** Panel, folds, scalers, per-node floor, metrics and DM are the delivered
  components — HAR-X is byte-identical to the canonical HAR-X (verified: baseline QLIKE matches). The ONLY
  addition is the HARQ column, so the comparison is a clean single-feature ablation.
- **HARQ term is causal.** `rolling5(pk^2)` with `min_periods=1` uses only days ≤ t; extracted at the same
  anchors as `har5`, `nan_to_num`'d identically.
- **Daily-only HARQ.** Interacting only the daily term (not weekly/monthly) — the full HARQ-F was numerically
  unstable and non-significant in the linear probe (`scripts/eda/harq_deepdive.py`).
- **RQ proxy.** True realized quarticity needs intraday returns; with daily OHLC we proxy it by the rolling
  mean of squared daily variance (variance-of-variance). Documented as a proxy.

## Gates (SDD)
- Simplicity: one added regressor, pure linear OLS, no new data, no network.
- Anti-abstraction: imports the delivered `wf_enriched_panel` / `wf_folds` / `run_masked_rich` directly.
- Performance: fully vectorised numpy; loop is folds×horizons×markets of OLS (seconds); no GPU, no batch=1.

## Files
- `code/harq_walkforward.py` — quarticity_panel / _ols_predict / run / main.
- `test/test_harq.py` — pytest: HARQ causality + formula, OLS recovery, no-leakage of the extra column, and a
  real-data smoke that asserts HAR-X reproduces canonical and HAR-X-Q ≤ HAR-X at h5.
