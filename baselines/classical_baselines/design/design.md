# Design — classical econometric baselines

## Data flow
1. `build_manifest(horizon)` — reuse `trackb_code.run_pilot.build_screening_inputs(False, None, 'P0',
   horizon)` which yields the pooled manifest (val/test = 14418/14464 at h5) and the per-ticker
   `PreprocessorStore` used by the ladder. This guarantees the SAME observations, targets, split,
   and per-ticker scalers as P0-G1 (obs-alignment invariant checked at runtime + in a test).
2. Raw series: `load_vol_series(ticker)` (full Parkinson vol from processed CSV) and
   `load_return_series(ticker)` (log returns from raw OHLCV close). Rolling HAR components computed
   once per ticker on the continuous series (trailing-only -> leakage-safe).
3. Each baseline emits `{(ticker_id, target_date): pred_raw}` covering exactly every val/test key.
4. `evaluate_baseline` round-trips raw predictions through the ladder store's per-ticker
   `target_scaler` (transform -> `evaluate_records` inverse-transforms back, exact float round-trip)
   so the 6 metrics are computed by the IDENTICAL ladder code (`train.evaluate_records`), including
   the positivity gate and per-ticker directional accuracy.

## Baseline math (all forecast the single-day vol at target_date = obs_date + h)
- Persistence: pred = sigma_park[obs_date].
- EWMA/RiskMetrics: v_t = lambda v_{t-1} + (1-lambda) sigma_park[t]^2, v_0 = sigma_park[0]^2,
  lambda=0.94; pred = sqrt(v[obs_date]) (RiskMetrics flat h-step forecast).
- HAR: OLS y ~ [sigma_d, sigma_w, sigma_m]; sigma_d=sigma[t], sigma_w=mean 5d, sigma_m=mean 22d.
  Fit per ticker on that ticker's train samples; predict val/test.
- HARQ (proxy): OLS y ~ [sigma_d, sigma_w, sigma_m, sigma_d*sqrt(RQ_d)] with RQ_d = sigma_d^2
  (daily range proxy). Flagged approximation; canonical intraday RQ unavailable.
- log-HAR: OLS ln(y) ~ [ln sigma_d, ln sigma_w, ln sigma_m]; pred = exp(fit).
- GARCH/GJR/EGARCH: `arch_model` on 100x log returns, `fit(last_obs=train_count)` (train-only
  params), `forecast(horizon=h, start=first val origin)`; pred_vol = sqrt(variance[h])/100 at the
  obs origin. GARCH/GJR analytic; EGARCH simulation (1000 paths, fixed seed).

## Positivity floor
Predictions floored at 1e-8 (matches the ladder positivity basis) before scoring so the
`evaluate_records` non-positive gate (<=1%) is satisfied; persistence/EWMA/GARCH are already >0,
only HAR/HARQ OLS can dip negative on rare rows.

## Design decisions / gates
- Simplicity Gate: one module, no new abstractions; reuse trackb manifest + `evaluate_records`.
- Anti-Abstraction Gate: use `arch` and `sklearn.LinearRegression` directly.
- Leakage: GARCH params train-only (frozen); HAR/HARQ/log-HAR fit on train samples only; EWMA and
  persistence are causal (trailing-only). Val/test obs identical to ladder.

## Files
- `code/classical_baselines.py` — library + `run_all(horizon, out_dir)` orchestrator.
- `code/run_classical_baselines.py` — CLI entry (sys.path bootstrap) -> writes canonical JSON/MD.
- `test/test_classical_baselines.py` — metric-correctness, obs-alignment, GARCH smoke, baseline math.
