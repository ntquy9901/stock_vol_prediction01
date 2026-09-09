# Gamma-loss GBM baseline — requirements

## Goal
Beat HAR-X on QLIKE using a nonlinear, QLIKE-trained gradient-boosted model, on the exact canonical
walk-forward folds. Motivation: QLIKE equals the gamma unit deviance (QLIKE(y,f)=y/f−log(y/f)−1), so a
gradient-boosting regressor with `loss='gamma'` trains directly on the evaluation metric, while trees
capture nonlinear interactions of the HAR features with a realized-quarticity (measurement-error) proxy and
causal "vol-is-declining" features that a linear HAR-X cannot.

## Method
- HAR-X = OLS on the 5 canonical features (unchanged, reproduces canonical exactly).
- GBM = `HistGradientBoostingRegressor(loss='gamma')` on 11 features: the 5 HAR features + RQ proxy
  (`har_daily·... ` via `sqrt(rolling5(pk²))`) + 5 log-vol decline features (change / slope5 / slope10 /
  dev5 / z22). One GBM pooled over all tickers per fold, refit on TRAIN, predicted on TEST; per-node
  positivity floor; target `pk[t+h]` (floored positive for the gamma loss).
- Reuses the delivered panel + folds + HAR-X OLS; date-clustered Diebold–Mariano GBM vs HAR-X.

## Inputs
- `data/processed_enriched/{sp500_clean,vn100,vn30}/*.csv` via `build_enriched_panel` + `make_folds`.

## Output / metrics
- Five metrics for HAR-X and GBM per market×horizon; DM (QLIKE/SE/AE). `results/gamma_gbm/gbm_{market}_h{h}.json`.

## Success criteria (go / no-go)
- MUST: HAR-X reproduces the canonical QLIKE (validates replication).
- Result (honest, data-size dependent):
  - **SP500 (480 tickers, large): GBM BEATS HAR-X significantly** — h5 +3.78% (p=0.0000), h10 +3.61% (p=0.039).
  - **VN100/VN30 (104/33 tickers, small): GBM is WORSE than HAR-X** (overfits the noisy, heavy-tailed,
    small panel; e.g. VN100 h5 −7.3% p=0.000). On VN the parsimonious HAR-X / HARQ / the VolGA+HAR-X-Q stack
    remain the winners.
- Conclusion: the gamma-GBM is the SP500 solution; it is NOT a VN solution. Report both honestly (no
  universal winner — consistent with Dudek et al. 2024).

## Non-goals
- No hyperparameter tuning on test. No deep model here (the VolGA+GBM ensemble is a separate follow-up).
