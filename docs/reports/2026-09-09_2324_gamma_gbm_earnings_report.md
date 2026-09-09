# gamma-GBM + forward-looking earnings beats HAR-X by ~11% on S&P 500 (walk-forward)

Adding a forward-looking earnings-calendar feature to the S&P 500 gamma-GBM gives the largest, significant
improvement found — attacking exactly the diagnostic-identified bottleneck (scheduled-earnings storm spikes).

## Result (canonical walk-forward, 7 folds; HAR-X reproduces canonical exactly)
| market | h | HAR-X | GBM | GBM+earn | +earn vs GBM | +earn vs HAR-X | DM p (vs GBM) |
|---|---|---|---|---|---|---|---|
| S&P 500 | 5 | 0.4550 | 0.4378 | **0.4008** | **+8.5%** | **+11.9%** | **0.000** ✓ |
| S&P 500 | 10 | 0.4797 | 0.4624 | **0.4278** | **+7.5%** | **+10.8%** | **0.000** ✓ |

- The **clean earnings ramp** `earn_prox=max(0,1-days_to_earnings/5)` (0 far from any earnings) is significant
  at BOTH horizons — the raw-distance version was only p=0.10 at h10 (far-cell noise); the ramp removes it.
- Mechanism (single-split decomposition): D9-near-earnings QLIKE 3.31→0.91; earnings fixes the ~18% of storm
  cells that are scheduled-earnings spikes.

## What did NOT help (honest negatives, documented)
- **Calm-floor calibration (b):** QLIKE-optimal-on-val multiplier makes it WORSE (h5 0.4125 > 0.4008; h10
  0.4481 > 0.4278) — does not generalise (val→test shift). Reported as a negative control; shipped model = GBM+earn.
- **Market + sector factor features** (`scripts/eda/market_sector_gbm.py`): the residual has 33% market-common
  + 11% sector-common structure (5x above a shuffled-sector null), BUT adding causal market/sector aggregates
  at the forecast origin does NOT reduce it (+market −0.0%/−2.7%, +sector +0.9%/−0.2%, n.s.). The common part
  is a CONTEMPORANEOUS shock, not a persistent regime knowable at t−h — so it is not forecastable, and a
  sector/market GNN (which also uses origin-time info) would not add forecasting value. This also explains why
  VolGA's correlation-edge graph never helped.

## The ceiling, stated precisely
Only FORWARD-LOOKING, scheduled information is exploitable:
- **Earnings** (scheduled, known ahead) → +7.5–8.5% (this baseline).
- **Unscheduled shocks** (tariff/short-seller: ~69% of D9) and **contemporaneous common co-movement**
  (market/sector) → structurally unpredictable from the past → the remaining error.
- Per-stock implied vol would be the next forward-looking lever (SP500 has liquid options) but its historical
  series is a paid data product.

## Deliverable
`baselines/2026-09-09_gamma_gbm_earnings/` (SDD, 8 tests, 100% line+branch, 3-lens review: 1 CRITICAL indexing
bug FOUND & FIXED + regression test, 1 test-gap fixed, calibration=documented negative). Earnings dates
`results/gamma_gbm/sp500_earnings.parquet` (yfinance). Results `results/gamma_gbm_earnings/gbmE_sp500_clean_h{5,10}.json`.

## Full S&P 500 picture (best model per horizon)
HAR-X → gamma-GBM (+3.6–13.3%, all h) → **gamma-GBM+earnings (+10.8–11.9% vs HAR-X at h5/h10)**. On VN, GBM
overfits — HARQ / VolGA+HAR-X-Q stack remain the winners there.
