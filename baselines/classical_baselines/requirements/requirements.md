# Classical econometric volatility baselines (Track-B paper)

## Goal
Provide a suite of CLASSICAL econometric volatility baselines evaluated on the EXACT same
data / target / temporal split / held-out observations as the consistent Track-B ladder
(`docs/reports/ladder_consistent_h5_2026-08-09_154402.json`), so they drop directly into a
combined paper table alongside the deep-model rungs P0 -> P1 -> P2 -> P3 -> G1.

## Input / basis (must match the ladder)
- Data: `data/processed/<TICKER>_processed.csv` (`date`, `parkinson_volatility`); 33 tickers.
- Raw prices for return-based models: `data/raw/prices/<TICKER>_ohlcv.csv` (`date,open,high,low,close,volume`).
- Target: Parkinson volatility `shift(-h)` (single day at `obs_date + h`), horizon h=5 (primary).
- Split: leakage-safe chronological 70/15/15 (per-ticker), built via `build_pooled_manifest`.
- Observation set: EXACTLY `pooled.samples['val']` (14418) and `pooled.samples['test']` (14464)
  for h=5 — same (ticker_id, target_date) keys and same raw targets used to score P0-G1.
- Metrics: the SAME 6 metrics via the ladder's `evaluate_records` (mse, rmse, mae, r2, qlike,
  directional_accuracy — per-ticker chronological direction).

## Baselines
- Persistence / random walk: pred = Parkinson vol at obs_date.
- EWMA / RiskMetrics (lambda=0.94) on the Parkinson variance proxy (flat multi-step forecast).
- HAR: per-ticker OLS on [sigma_d, sigma_w, sigma_m] -> vol at obs_date+h (anchoring; P0 is the
  pooled deep-pipeline HAR anchor, reported from the ladder JSON).
- HARQ (Bollerslev-Patton-Quaedvlieg 2016): implemented with a DAILY range-based realized-
  quarticity proxy RQ_d = sigma_d^2 (no intraday returns in this daily dataset -> the canonical
  5-min RQ is not identified; this is an approximation, flagged as such and NOT the canonical HARQ).
- log-HAR: per-ticker OLS on log-volatility, exp() back to raw scale.
- GARCH-family (per-ticker, `arch` package): GARCH(1,1), GJR-GARCH, EGARCH on 100x log-returns
  (close-to-close). Params estimated on the train sample only (frozen), h-step marginal-variance
  forecast at each obs origin -> daily return vol, compared to the Parkinson RV proxy (same daily
  return-vol scale). GARCH/GJR analytic multi-step; EGARCH simulation multi-step.

## Success criteria (go/no-go)
- Every baseline scored on EXACTLY the 14418 val / 14464 test observations (obs-alignment test).
- 6-metric val+test table for every baseline, same schema as the ladder canonical file.
- TDD: metric-correctness test (matches `evaluate_records`), obs-alignment test, per-ticker GARCH
  smoke on a tiny real-data slice. `pytest` green, ruff clean.
- Canonical `docs/reports/classical_baselines_h5_<ts>.{json,md}` + note comparing each baseline to
  P3/G1.
- No fabricated numbers: every reported value comes from a real fit/forecast on real data.

## Non-goals
- Retuning the deep models. Retraining P0-G1. Touching `.worktrees/` or `archive/`.
