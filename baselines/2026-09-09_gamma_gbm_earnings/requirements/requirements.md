# gamma-GBM + earnings + calm-calibration — requirements

## Goal
Improve the committed S&P 500 gamma-GBM (baselines/2026-09-08_gamma_gbm) by attacking the two remaining
test-set issues identified by the diagnostic (`docs/reports/2026-09-09_gbm_sp500_diagnostic.html` +
residual analysis):
- **Issue A — storm D9 under-forecast (40% of remaining error).** ~18% of D9 storms are SCHEDULED
  earnings spikes (fixable): adding a forward-looking earnings feature cut D9-near-earnings QLIKE 3.31→0.91
  in a single-split probe. ~69% of D9 are UNSCHEDULED shocks (tariff/short-seller) — structurally
  unfixable (no advance signal).
- **Issue B — calm D0 over-forecast (~16% of remaining error, bias ~4x).** A QLIKE-optimal calm-floor
  calibration on validation targets this.

## Method
- Base: gamma-GBM (loss='gamma' = QLIKE) on 5 HAR + RQ + 5 decline features, canonical walk-forward.
- (a) CLEAN earnings feature: `earn_prox = max(0, 1 - days_to_nearest_earnings/5)` (a ramp that is 0 far from
  earnings → adds no noise on the ~76% far cells, which hurt h10 in the raw-distance probe) + `earn_soon`
  (binary within 3 days). Distance measured from the TARGET day (t+h) to the nearest scheduled earnings —
  forward-looking (schedule known ahead). Earnings dates from yfinance (`results/gamma_gbm/sp500_earnings.parquet`).
- (b) Calm-floor calibration: per forecast-quantile bin, the QLIKE-optimal multiplier c*=mean(y/f) fit on the
  VALIDATION split, applied to test (clipped [0.3, 3]). No test peeking.

## Output / metrics
- 5 metrics for HAR-X / GBM / GBM+earn / GBM+earn+cal; date-clustered DM vs HAR-X AND vs GBM-base.
  `results/gamma_gbm_earnings/gbmE_sp500_clean_h{h}.json`.

## Success criteria (go / no-go) — RESULT
- MUST met: HAR-X reproduces canonical exactly; GBM+earn beats GBM-base.
- **GO met (earnings):** walk-forward S&P 500 h5 — HAR-X 0.4550 → GBM 0.4378 → **GBM+earn 0.4008
  (+8.5% vs GBM-base, +11.9% vs HAR-X, DM p<0.001)**. The clean earnings ramp is a large, significant gain.
- **Calibration (b) = NEGATIVE control (do NOT use):** GBM+earn+cal 0.4125 is WORSE than GBM+earn 0.4008 — the
  QLIKE-optimal-on-val calm calibration does not generalise (val→test regime shift). Reported as a documented
  negative; the shipped model is GBM+earn.
- Honest ceiling: the unscheduled-shock ~69% of D9 is structural (no advance signal); and the 33% market-common
  + 11% sector-common residual is a CONTEMPORANEOUS shock NOT predictable from origin-time market/sector
  features (a separate probe `scripts/eda/market_sector_gbm.py` found +market/+sector features do not help) —
  so a sector/market GNN would not add forecasting value. Earnings (forward-looking, scheduled) is the only
  exploitable lever.

## Non-goals
- No VN (GBM overfits VN; earnings there needs a VN earnings source + goes into HARQ/stack). No hyperparameter
  tuning on test.
