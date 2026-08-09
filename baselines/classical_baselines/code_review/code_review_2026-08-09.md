# Code review — classical econometric baselines (2026-08-09)

Adversarial 3-layer review (Blind Hunter / Edge Case Hunter / Acceptance Auditor) of the classical
baseline suite (`code/classical_baselines.py`, `code/run_classical_baselines.py`). All findings
below were resolved before the run was accepted; each is backed by a test or a runtime check.

## Blind Hunter (hidden correctness bugs)

- **B1 — Target is Parkinson VARIANCE, not vol (unit mismatch in GARCH/EWMA).** The processed
  `parkinson_volatility` column is numerically sigma^2 = (ln(H/L))^2/(4 ln 2) (verified corr=1.0,
  ratio=1.0 vs raw OHLCV; median ~1.3e-4). The first GARCH implementation returned a VOL forecast
  (`sqrt(var)/100`), giving test R^2 = -19 (a ~50x scale error). Fixed to return the raw-return
  VARIANCE (`var/1e4`). RiskMetrics EWMA likewise now smooths the variance series directly
  (`v_t = lam v_{t-1} + (1-lam) RV_t`) instead of `sqrt(EWMA(RV^2))`. Persistence/HAR/HARQ/logHAR
  were already unit-consistent (they forecast the same series they read).
- **B2 — Date-format mismatch dropped GARCH coverage silently.** VPB and VRE store tz-aware
  timestamps (`2017-08-18 00:00:00+07:00`) while obs keys are `YYYY-MM-DD`; a naive `astype(str)`
  made 100% of their return dates fail to align, which surfaced as `min() arg is empty`.
  Fixed by normalizing OHLCV dates to the local calendar date via `pd.to_datetime(...).dt.strftime`
  (NOT `utc=True`, which would shift +07:00 midnights to the previous day). Regression test
  `test_ohlcv_dates_normalized_to_date_only`.
- **B3 — log(0) in log-HAR.** Days with H==L give Parkinson variance exactly 0, so `ln(sigma)`
  produced `-inf` and crashed `LinearRegression` at predict time. Fixed by flooring HAR components
  at 1e-8 in `_har_design`.

## Edge Case Hunter (boundaries)

- **E1 — Missing raw OHLCV (LPB).** LPB has no price file anywhere, so return-based GARCH cannot
  cover it. The GARCH family is scored on the 32-ticker subset (14247 val / 14292 test) and the
  discrepancy is reported explicitly (`garch_excluded_tickers`, per-baseline `n_obs`); the
  vol-only baselines keep exact 14418/14464 ladder alignment. No fabricated fallback predictions.
- **E2 — Zero-RV days inflate persistence QLIKE.** Persistence predicts RV[obs_date]; on H==L days
  RV=0 is floored to 1e-8, and QLIKE (ratio - log ratio - 1) explodes for the few obs whose target
  is non-zero but prediction is ~0. Persistence QLIKE (~2050 val / ~4151 test) is therefore
  dominated by a handful of degenerate days and is not a meaningful ranking metric; its
  RMSE/MAE/R^2/DirAcc remain valid. Documented in the report.
- **E3 — Positivity gate.** `evaluate_records` rejects >1% non-positive raw predictions; all
  predictions are floored at 1e-8 before scoring (matches the ladder positivity basis).
- **E4 — Full obs coverage enforced.** `assert_full_coverage` fails if any baseline misses or
  duplicates an observation key (test `test_assert_full_coverage_detects_missing`); run against
  every baseline inside `run_all`.

## Acceptance Auditor (does it meet the spec?)

- **A1 — Same observations as the ladder.** `build_manifest` reuses the ladder's manifest path;
  the smoke test asserts val=14418 / test=14464 at h5 (exact ladder counts).
- **A2 — Same scorer.** `evaluate_baseline` round-trips raw predictions through the ladder store's
  per-ticker `target_scaler` and calls the identical `train.evaluate_records`; the round-trip is
  lossless (test `test_roundtrip_is_lossless_on_raw_scale`) and reproduces `evaluate_records`
  exactly (test `test_metric_correctness_matches_ladder_scorer`, abs<=1e-12).
- **A3 — Leakage safety.** GARCH params are estimated on the train sample only (`last_obs`), frozen
  for val/test forecasts; HAR/HARQ/logHAR fit on train samples only; EWMA/persistence are causal
  (trailing-only). Val/test observation set is identical to the ladder's.
- **A4 — HARQ honesty.** The daily dataset has no intraday realized quarticity; HARQ uses a
  documented range proxy RQ_d = RV_d^2 and is labelled an approximation, not the canonical HARQ.

## Verdict
No open HIGH/MEDIUM findings. Every reported number is produced by a real fit/forecast on real data
and scored by the ladder's own metric function. Minor/accepted: persistence QLIKE degeneracy (E2,
documented, not fixed by design); GARCH 32/33-ticker coverage (E1, documented discrepancy).
