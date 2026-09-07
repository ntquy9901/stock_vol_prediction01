# Deep + HARQ stack — requirements

## Goal
Beat HAR-X on QLIKE at h5/h10 (and h1) on VN100 and VN30 by a larger, significant margin than any single
model, by combining two decorrelated forecasts: VolGA (deep, nonlinear; numerically lowest QLIKE but
seed-noisy → not significant alone) and HAR-X-Q (linear, measurement-error-corrected; small but consistent
significant gain, from `baselines/2026-09-07_harq`).

## Method
- `stack = w·VolGA + (1−w)·HAR-X-Q`, both positive variance forecasts, floored.
- **PRIMARY: w = 0.5** (equal weight) — an a-priori 1/N ensemble, no fitting, no overfitting.
- **SECONDARY: w fit on the VALIDATION split** (pooled grid) per market×horizon, applied to TEST — reported
  as a control; expected to overfit (the forecast-combination puzzle).
- HAR-X / HAR-X-Q recomputed per cell on the exact canonical folds via the reviewed `harq_walkforward`;
  VolGA read from the canonical `qlike_anchor` cells (5-seed ensemble). Shared QLIKE floor; date-clustered DM.

## Success criteria (go / no-go)
- MUST: the equal-weight stack never worse than HAR-X on QLIKE.
- GO: equal-weight stack QLIKE < HAR-X with DM p<0.05 at a majority of horizons on both markets.
- Achieved (equal weight): significant at h1/h5/h10 on BOTH VN100+VN30 (6/8 cells), gains +1.1..+3.6%;
  h22 numerically best (+0.99% / +1.59%) but not significant. The stack also beats BOTH members at h5/h10.
- Honest control: the val-fit weight leans VolGA-heavy (0.75–1.0) and does WORSE out-of-sample than equal
  weight — the forecast-combination puzzle (DeMiguel et al. 2009; Smith & Wallis 2009). Report equal weight.

## Non-goals
- No new training; VolGA and HAR-X-Q are frozen forecasts. No test-set weight tuning.
