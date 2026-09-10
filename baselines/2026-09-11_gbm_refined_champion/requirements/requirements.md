# Requirements — Refined SP500 champion (GBM + earnings + estimators + spike)

## Goal
Consolidate three separately-validated feature probes into ONE committed walk-forward model on S&P 500, measured
apples-to-apples against HAR-X and against the committed GBM+earn champion
(`baselines/2026-09-09_gamma_gbm_earnings`), so the refined model can replace GBM+earn as the SP500 champion with
full evidence (6 metrics, date-clustered DM, overfit evidence) rather than as loose `scripts/eda/` probes.

## Input / output
- **Input:** enriched SP500 clean panel (`data/processed_enriched/sp500_clean/*.csv`, 497 tickers, OHLCV +
  Parkinson/GK/RS/YZ variance + HAR channels + volume z-score), and scheduled earnings dates
  (`results/gamma_gbm/sp500_earnings.parquet`, yfinance).
- **Output:** `results/gbm_refined_champion/gbmR_sp500_clean_h{1,5,10,22}.json` — per horizon: 6 metrics
  (MSE/RMSE/MAE/R²/QLIKE/DirAcc) for HAR-X, GBM+earn, GBM+refined; date-clustered DM (refined vs HAR-X, refined vs
  GBM+earn); overfit evidence (train vs test QLIKE + gap%).

## Model
- **GBM+refined** = HistGradientBoostingRegressor(loss='gamma') on: 5 HAR-X features + 6 log-vol-decline extras
  (rq, mr_change, mr_slope5/10, mr_dev5, mr_z22) + **3 estimators** (Garman-Klass, Rogers-Satchell,
  Yang-Zhang n=20, origin-time) + **4 asymmetric-earnings** (earn_prox, earn_soon, earn_pre, earn_post) +
  **4 spike-propensity** (vov22, spike_rate63, spike_mag63, accel10). Total 22 features.
- Compared against: **HAR-X** (canonical OLS) and **GBM+earn** (committed champion = HAR5 + extras + earn_prox +
  earn_soon).

## Acceptance criteria (go / no-go)
- **GO** if GBM+refined beats GBM+earn on pooled QLIKE at ≥3 of 4 horizons, with date-clustered DM p<0.05 at ≥2
  horizons, AND does not significantly worsen any horizon, AND overfit gap (train→test QLIKE) < ~15% (no
  overfitting), AND still beats HAR-X on QLIKE at all horizons (retaining the committed edge).
- **CONDITIONAL GO** if it beats GBM+earn on QLIKE at all/most horizons but DM significance is marginal — report
  as an incremental refinement, keep GBM+earn as the headline.
- **NO-GO** (keep GBM+earn) if refined does not beat GBM+earn, or overfits, or degrades a horizon significantly.

## Leakage constraints (mandatory)
- Earnings distances measured from the TARGET day t+h to SCHEDULED dates known at t (forward-looking, legitimate).
- Estimators/spike/extras use only data up to origin day t (trailing windows; estimators aligned to anchor day t).
- Walk-forward: per-fold train/val/forecast chronological; target-interval embargo at boundaries
  (`assert_no_leakage`); HAR-X OLS fit on train only; no test peeking for any feature or hyperparameter.

## Non-goals
- No graph / cross-stock features (exhaustively shown NO-GO on SP500 and VN this cycle).
- No new hyperparameter search (reuse the committed gamma-GBM config for a fair like-for-like comparison).
- VN markets excluded (no per-firm earnings dates available → earnings/asym-earnings not buildable).
