# VN100: why the deep LSTM underperforms the linear HAR-X (data-mining)

_Read-only mining of the delivered `masked_rich` pipeline across train/val/test. Deep = LSTM (5 feats, no graph); Linear = HAR-X (5-feat OLS). Target = Parkinson variance; QLIKE floor 1e-08 shared; fits on train only._

## Executive summary

On VN100 the deep LSTM's one robust deficit against the parsimonious linear HAR-X is on QLIKE (h1 test 0.5650 vs 0.5115; delivered date-clustered DM p=1.1e-3): on squared error the two are near-parity (h1 test MSE 2.365e-07 vs 2.367e-07). Decomposing QLIKE by target magnitude localises the bulk of the h1 gap to the high-volatility deciles — the top four deciles carry ~76% of it and the top two alone ~56% — where the LSTM under-predicts the volatility spikes slightly more than HAR-X and QLIKE penalises tail under-prediction asymmetrically. Ranked by evidence: (1) tail spike-miss under an asymmetric QLIKE (primary, but a MODEST relative effect — both models smooth heavily, var(pred)/var(actual)=0.24 for the LSTM vs 0.24 for HAR-X); (2) loss-metric mismatch — the LSTM is MSE-competitive yet QLIKE-deficient, the MSE-trained / QLIKE-scored signature; (3) HAR's parsimonious basis is near-optimal for a low signal-to-noise, strongly persistent target; (4) only a mild overfitting signal — both models' test loss sits BELOW their train loss (the test regime is lower-variance), the LSTM merely generalises marginally worse than HAR-X.

## Ranked conclusion

1. **Tail spike-miss under an asymmetric QLIKE (primary).** The LSTM's QLIKE deficit is localised to the high-volatility deciles (top-4 ~76%, top-2 ~56% of the h1 gap): it under-predicts spikes slightly more than HAR-X, and QLIKE punishes tail under-prediction asymmetrically. The effect is modest and relative — both models smooth heavily (var(pred)/var(actual) ~ 0.24 for both).
2. **Loss-metric mismatch (contributing).** MSE-trained, QLIKE-scored: the LSTM is MSE-competitive but QLIKE-deficient — the mismatch surfaces in the tail region QLIKE up-weights. HAR-X keeps the MSE/RMSE edge in the delivered 5-seed pipeline.
3. **HAR inductive-bias near-optimality (contributing).** Strong persistence + low signal-to-noise (~0.2 of the target is forecastable) favour the high-bias linear basis; HAR in-sample vs OOS R^2 is stable.
4. **Overfitting (mild, not dominant).** Both models' TEST loss is below their TRAIN loss (lower-variance test regime); the LSTM only generalises marginally worse than HAR-X.

## Quantitative evidence (h1 test)

- test QLIKE: LSTM 0.5650 vs HAR-X 0.5115 (the robust, significant gap)
- test MSE: LSTM 2.365e-07 vs HAR-X 2.367e-07 (near-parity — deficit is QLIKE-specific)
- QLIKE-gap concentration: top-4 deciles ~76%, top-2 deciles ~56% of the gap
- prediction-variance ratio var(pred)/var(actual): LSTM 0.235, HAR-X 0.239 (both << 1 — both compress; actual = 1.0)
- top-decile signed bias mean(pred-actual): LSTM -1.042e-03, HAR-X -1.033e-03 (both negative; LSTM slightly more = worse spike-miss)
- h1 HAR in-sample R^2 0.2109 / OOS R^2 0.2226 / signal-to-noise 0.2673
- h1 lag-1 target autocorrelation 0.328 (slow decay = persistence)

## Per-split metrics (retrained pipeline)

| h | model | train MSE | train QLIKE | test MSE | test MAE | test QLIKE | QLIKE gap (test-train) |
|---|---|---|---|---|---|---|---|
| h1 | HAR-X | 4.872e-07 | 0.9039 | 2.367e-07 | 2.898e-04 | 0.5115 | -0.3923 |
| h1 | LSTM | 4.820e-07 | 0.9022 | 2.365e-07 | 2.873e-04 | 0.5650 | -0.3372 |
| h5 | HAR-X | 5.345e-07 | 0.9611 | 2.606e-07 | 3.160e-04 | 0.5633 | -0.3977 |
| h5 | LSTM | 5.323e-07 | 0.9615 | 2.669e-07 | 3.191e-04 | 0.5813 | -0.3802 |
| h10 | HAR-X | 5.508e-07 | 0.9900 | 2.754e-07 | 3.306e-04 | 0.6023 | -0.3877 |
| h10 | LSTM | 5.479e-07 | 0.9846 | 2.821e-07 | 3.322e-04 | 0.6216 | -0.3630 |
| h22 | HAR-X | 5.728e-07 | 1.0287 | 2.891e-07 | 3.486e-04 | 0.6405 | -0.3883 |
| h22 | LSTM | 5.653e-07 | 1.0085 | 2.992e-07 | 3.504e-04 | 0.6559 | -0.3526 |

_QLIKE gap (test-train) is NEGATIVE for both models: the test regime is lower-variance, so neither model grossly overfits; the LSTM's train->test relationship is only marginally worse than HAR-X's._

## Caveats

- The deep model genuinely loses on QLIKE (significant at h1); near-parity on squared error; wins only MAE at short horizons. Delivered pipeline gives HAR-X the MSE/RMSE edge at every horizon.
- Over-smoothing is a modest, RELATIVE effect (both models compress ~equally); the LSTM's extra tail under-prediction shows mainly in the QLIKE decile decomposition.
- Evidence is correlational (measured patterns consistent with the mechanism, not a causal proof).
- LSTM metrics are of the seed-averaged (ensemble) prediction (the delivered `metrics` / DM basis), not the per-seed-mean paper headline (`metrics_per_seed`, generally slightly higher).
- Single delivered configuration (lookback 10, 5 seeds, 20-epoch early-stopped LSTM).
- Graph/horizon-decay studied separately; out of scope here.
