# HARQ walk-forward baseline — requirements

## Goal
Beat HAR-X on QLIKE at the long horizons (h5/h10/h22, and h1) on VN100 and VN30 by adding the HARQ
measurement-error correction (Bollerslev, Patton & Quaedvlieg 2016) to HAR-X, evaluated on the EXACT
canonical walk-forward folds. This is the first lever (after regime-blend / long-memory / VIX /
mean-reversion, all non-significant at long h) that produces a statistically significant improvement.

## Method
- HAR-X = OLS on the 5 canonical features `[pk, har_w, har_m, market_pk, volume_zscore_22]` (unchanged).
- HAR-X-Q = HAR-X + one HARQ term: `har_daily * sqrt(RQ)`, `RQ = rolling5 mean of pk^2` (a realized-
  quarticity proxy; daily OHLC only). The interaction attenuates the noisy daily term when recent
  variance-of-variance is high. Daily-only form (the full HARQ-F interacting all terms was unstable).
- Both linear OLS refit per fold on TRAIN valid rows; per-node positivity floor `POS_FLOOR_FRAC*mean+EPS`;
  target `pk[t+h]`. Reuses the delivered panel + folds + scalers unchanged.

## Inputs
- `data/processed_enriched/{vn100,vn30}/*.csv` (via the delivered `build_enriched_panel` + `make_folds`).

## Output / metrics
- Five metrics (MSE/RMSE/MAE/QLIKE/R²) for HAR-X and HAR-X-Q per market×horizon.
- Date-clustered Diebold–Mariano (QLIKE/SE/AE) HAR-X-Q vs HAR-X. Results `results/harq/harq_{market}_h{h}.json`.

## Success criteria (go / no-go)
- MUST: HAR-X baseline reproduces the canonical HAR-X QLIKE exactly (validates the walk-forward replication).
- GO: HAR-X-Q QLIKE < HAR-X with DM p < 0.05 at a majority of horizons on both markets.
- Achieved: 7/8 cells significant (VN100 all h; VN30 h1/h5/h10). Gains small (+0.12..+0.80%) but significant.
  Honest: VN30 h22 not significant (−0.04%); report as such.

## Non-goals
- No deep model here (VolGA+HARQ is a follow-up needing a 6th node feature). No intraday RQ (proxy only).
