# Requirements — regime / change-point features (Tier-1 shock detection)

## Objective
Test whether adding causal market-regime features improves volatility forecasts, using the strongest baseline
(HAR-X) as the probe. Motivated by the shock-detection deep-research (`docs/reports/2026-09-06_shock_detection_research.md`):
regime/change-point signals were the highest-leverage, lowest-effort daily-data add (Marcucci 2005 MRS-GARCH;
regime-switching HAR), because the high-frequency jump estimators (bipower/HARQ) need intraday returns we do not have.

## Input / output
- Input: the enriched daily panel per market (S&P 500, VN100, VN30); regime features derived causally from the
  market-level Parkinson-variance aggregate.
- Output: `results/regime_features/regime_<market>_h<h>.json` with QLIKE (and all metrics) for HAR-X and
  HAR-X+regime, plus a date-clustered Diebold-Mariano test (HAR-X+regime vs HAR-X).

## Success criteria (go/no-go)
- **GO** (justifies integrating regime features into the LSTM/VolGA input, in_dim 5→8): HAR-X+regime attains a
  DM-significant QLIKE improvement over HAR-X on at least one market at the short horizons.
- **NO-GO**: no DM-significant improvement anywhere. Then the deep-model integration (which changes the model
  input dimension and requires a full retrain) is not worth the cost, and the result is reported as an honest
  negative.

## Constraints
- Causal only: every regime feature uses data up to day $t$ (unit-tested for no look-ahead).
- Hard isolation (§3.F): reuse the delivered panel/fold/HAR machinery read-only; do not modify other baselines.
- Same leakage-free walk-forward, folds, and QLIKE floor as the main VolGA experiment.
