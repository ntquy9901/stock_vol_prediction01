# Design — gamma-GBM + earnings + calm-calibration

## Data flow (per canonical walk-forward fold)
1. Reuse `gamma_gbm_walkforward` (G): `extra_feature_panels` (RQ + decline), `_design` (5 HAR + 6 extras),
   `_harx_ols` (canonical HAR-X). Same panel/folds/floor.
2. `earnings_panels(panel, earn)` → `earn_prox [n_anchor,N]` = `max(0, 1 - dist/RAMP)`, `earn_soon` = `dist<=SOON`,
   where `dist` = days from each anchor's TARGET date (`panel.target_dates`) to the nearest scheduled earnings of
   that ticker (vectorized searchsorted). Forward-looking; ramp is 0 far from earnings (no far-cell noise).
3. Fit gamma GBM on masked TRAIN rows for two feature sets: base (11) and +earn (13); predict val + test.
4. `_calibrate(pred_val, y_val, pred_test)`: per forecast-quantile bin, `c*=mean(y_val/f_val)` (QLIKE-optimal
   multiplier, clipped [0.3,3]) applied to test → `GBM+earn+cal`.
5. Pool HAR-X / GBM / GBM+earn / GBM+earn+cal via `RMR._pred_dict`; `_metrics` + `_dm_all` vs HAR-X and vs GBM.

## Key decisions
- **Clean ramp, not raw distance.** The raw `earn_dist` probe helped h5 (+6.6%) but only p=0.10 at h10 because
  it added noise on the 76% far cells; `earn_prox` is exactly 0 beyond RAMP days → keeps the +48% near-earnings
  lift without the far-cell dilution.
- **Forward-looking, causal.** Distance uses the scheduled earnings calendar (known ahead); measured to the
  TARGET day. Caveat: yfinance dates are actual announcement dates (a rare few may differ from the pre-announced
  date) — a minor, standard assumption; documented.
- **QLIKE-optimal calibration.** For QLIKE, argmin_c QLIKE(y, c·f) = mean(y/f); applied per forecast bin, fit on
  val only. Corrects systematic calm-decile over-forecast without a test peek.
- **Reuse the reviewed base.** GBM base is byte-identical to the committed gamma-GBM; earnings and calibration
  are additive, so each contribution is a clean ablation (GBM → +earn → +cal).

## Gates (SDD)
- Simplicity: 2 engineered features + a 1-parameter-per-bin calibration; reuses the committed GBM.
- Anti-abstraction: imports `gamma_gbm_walkforward`, the delivered panel/folds/metrics/DM, and sklearn.
- Performance: vectorised panels; per-fold GBM (CPU, minutes on SP500); no GPU, no per-item loop.

## Files
- `code/gbm_earnings_walkforward.py` — earnings_panels / _design / _calibrate / run / main.
- `test/test_gbm_earnings.py` — earnings proximity causality + ramp formula, calibration = QLIKE-optimal
  multiplier, design column counts, real-data smoke (HAR-X reproduces canonical; +earn does not worsen QLIKE).
