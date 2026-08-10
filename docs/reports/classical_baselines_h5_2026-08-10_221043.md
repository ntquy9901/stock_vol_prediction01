# Classical econometric volatility baselines (h5)

One basis for every baseline: leakage-safe chronological 70/15/15 split, Parkinson-volatility target shift(-h), and the EXACT pooled val/test observations (same keys + raw targets) scored by the same `train.evaluate_records` as the deep-model ladder P0-G1.

Observation set: val=14418, test=14464, tickers=33. Metrics via the ladder `evaluate_records` scorer.

## VAL metrics

| baseline | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| Persistence | 3.36036e-06 | 0.00183313 | 0.000578995 | 0.596149 | 2050.83 | 48.20 |
| EWMA | 2.18086e-06 | 0.00147677 | 0.000475058 | 0.737902 | 0.526228 | 48.61 |
| HAR | 2.39922e-06 | 0.00154894 | 0.0005155 | 0.711659 | 0.528577 | 48.67 |
| HARQ | 2.39898e-06 | 0.00154886 | 0.000513964 | 0.711688 | 0.523225 | 48.93 |
| logHAR | 2.21514e-06 | 0.00148833 | 0.000450251 | 0.733783 | 0.693417 | 48.05 |
| GARCH | 8.57967e-06 | 0.00292911 | 0.000852663 | -0.0311145 | 1.93983 | 49.60 |
| GJR-GARCH | 8.57689e-06 | 0.00292863 | 0.000854061 | -0.0307797 | 1.8755 | 49.66 |
| EGARCH | 8.5412e-06 | 0.00292253 | 0.000854806 | -0.0264909 | 1.86983 | 49.16 |

## TEST metrics

| baseline | mse | rmse | mae | r2 | qlike | dir_acc |
|---|---|---|---|---|---|---|
| Persistence | 7.68559e-06 | 0.00277229 | 0.000722742 | 0.658 | 4151.22 | 48.01 |
| EWMA | 5.33929e-06 | 0.00231069 | 0.000610615 | 0.762408 | 0.600625 | 48.03 |
| HAR | 5.24277e-06 | 0.00228971 | 0.00063117 | 0.766703 | 0.579291 | 48.40 |
| HARQ | 5.2402e-06 | 0.00228915 | 0.000628925 | 0.766817 | 0.573674 | 48.38 |
| logHAR | 5.62759e-06 | 0.00237225 | 0.0005932 | 0.749579 | 0.779422 | 48.83 |
| GARCH | 2.23971e-05 | 0.00473256 | 0.0011592 | 0.00335494 | 1.75138 | 48.49 |
| GJR-GARCH | 2.24496e-05 | 0.00473811 | 0.00116291 | 0.00101609 | 1.81412 | 48.40 |
| EGARCH | 2.25369e-05 | 0.00474731 | 0.00116734 | -0.00286811 | 1.86327 | 48.64 |

## Notes
- **HARQ**: HARQ uses a DAILY range-based realized-quarticity proxy RQ_d = sigma_d^2 because the dataset is daily OHLCV (no intraday returns); the canonical BPQ-2016 5-min RQ is not identified. This is an approximation, not the canonical HARQ.
- **target_units**: The processed `parkinson_volatility` column is numerically the Parkinson VARIANCE estimator sigma^2 = (ln(H/L))^2 / (4 ln 2) (verified corr=1.0 vs raw OHLCV, median ~1.3e-4). Every baseline forecasts this daily realized-variance quantity; EWMA smooths it directly and GARCH forecasts the conditional return variance (same units).
- **GARCH_family**: GARCH(1,1)/GJR/EGARCH fit per ticker on 100x close-to-close log returns; params estimated on the train sample only (frozen). The h-step marginal conditional variance (percent^2) is divided by 1e4 to recover the raw-return variance, directly comparable to the Parkinson variance target.
- **P0_anchor**: P0 (pooled HAR on standardized features, from the ladder) is the deep-pipeline HAR anchor; the HAR row here is a per-ticker OLS on raw volatility.
- **GARCH_coverage**: GARCH/GJR/EGARCH cover all 33 of 33 tickers on the full 14418 val / 14464 test observation set (exact ladder alignment), same as the vol-only baselines.
- **LPB_provenance**: LPB raw OHLCV was recovered from the SSI iBoard API (2020-11-09..2026-08-10); Parkinson variance recomputed from its High/Low reproduces LPB_processed.csv (median |diff| 4.8e-6). SSI uses a different price-adjustment convention than the other tickers (immaterial for return-GARCH: log-returns are scale-invariant except on a few ex-dividend days). This lifts the GARCH family from 32/33 to 33/33 tickers.
- **GARCH_calendar_gap**: 18 LPB observations fall on holidays (Tet 2025, Apr/May 2025, New Year 2026) present in the processed series but absent from the SSI trading calendar; for those the GARCH forecast is carried forward from the last trading origin (persistent conditional variance; 0.12% of test observations).
