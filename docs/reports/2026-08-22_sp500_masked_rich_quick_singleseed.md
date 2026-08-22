# S&P 500 quick check — masked-rich panel (single seed), for discussion

Date: 2026-08-22. Status: exploratory single-seed run (NOT the paper's submitted result; the submitted
paper is Vietnam-only). Purpose: preview whether the deep/graph models behave on a large market before a
full multi-seed S&P 500 study.

## Setup
- Design: same masked union-of-dates panel and five node features as the paper's VN Tables 1-2
  (`masked-union-panel-rich-5feat-weighted-gat`), lookback = 10, per-node positivity floor 1e-2 x train mean,
  date-clustered Diebold-Mariano inference.
- Universe: S&P 500 constituents, 442 nodes after the min-train-rows drop, 1624 test dates, 717,768 masked
  test observations at h1.
- Seeds: **1 (seed 42)** — single-seed exploratory run, `--no-corr` (only the directed volume->Parkinson
  graph, no symmetric-correlation comparator). GARCH not computed for this panel.
- Models: HAR (classic 3-feature OLS: daily/weekly/monthly), HAR-X (5-feature OLS = the 3 HAR features +
  market PK + volume z-score; this is what the paper labels "HAR"), LSTM (5-feature, no graph), LSTM+GAT
  (directed volume->Parkinson Top-5 weighted 2-hop). The paper's submitted tables use the 5-feature model
  (HAR-X) as "HAR"; both are shown here for completeness.
- Source JSONs: `results/masked_rich_floor1e2/sp500_h{1,5,10,22}/result.json`.

## All metrics (single seed)
Bold marks the best value per horizon per metric. Lower is better for MSE/RMSE/MAE/QLIKE; higher for R2.

| h | Model | MSE (×10⁻⁷) | RMSE (×10⁻⁴) | MAE (×10⁻⁴) | QLIKE | R² |
|---|---|---:|---:|---:|---:|---:|
| **1** | HAR | 6.276 | 7.922 | **2.359** | **0.3622** | 0.3079 |
|  | HAR-X | **6.171** | **7.856** | 2.361 | 0.4078 | **0.3195** |
|  | LSTM | 6.367 | 7.979 | 2.453 | 0.4110 | 0.2979 |
|  | LSTM+GAT | 6.384 | 7.990 | 2.466 | 0.4064 | 0.2960 |
| **5** | HAR | 8.086 | 8.992 | 2.679 | 0.4349 | 0.1084 |
|  | HAR-X | 8.062 | 8.979 | 2.691 | **0.4303** | 0.1110 |
|  | LSTM | **7.850** | **8.860** | **2.573** | 1.0433 | **0.1344** |
|  | LSTM+GAT | 7.902 | 8.890 | 2.637 | 0.7907 | 0.1287 |
| **10** | HAR | 8.937 | 9.454 | 2.829 | 0.4824 | 0.0165 |
|  | HAR-X | 8.948 | 9.459 | 2.842 | **0.4775** | 0.0153 |
|  | LSTM | 8.505 | 9.222 | 2.767 | 0.8276 | 0.0640 |
|  | LSTM+GAT | **8.497** | **9.218** | **2.701** | 1.0110 | **0.0650** |
| **22** | HAR | 9.647 | 9.822 | 2.991 | **0.5993** | -0.0603 |
|  | HAR-X | 9.623 | 9.810 | 2.983 | 0.6150 | -0.0576 |
|  | LSTM | 9.399 | 9.695 | 3.259 | 0.8515 | -0.0330 |
|  | LSTM+GAT | **9.172** | **9.577** | **2.952** | 1.3198 | **-0.0080** |

## Diebold-Mariano contrasts (date-clustered; p-value, favoured model)
A "better" favour means the first-named model has the lower loss.

| h | LSTM vs HAR (QLIKE) | (MAE) | LSTM+GAT vs LSTM (QLIKE) | (MAE) |
|---|---|---|---|---|
| 1 | 0.000 (HAR) | 0.000 (HAR) | 0.226 (GAT) | 0.002 (LSTM) |
| 5 | 0.000 (HAR) | 0.021 (LSTM) | 0.000 (GAT) | 0.000 (LSTM) |
| 10 | 0.017 (HAR) | 0.408 (LSTM) | 0.035 (LSTM) | 0.000 (GAT) |
| 22 | 0.066 (HAR) | 0.024 (HAR) | 0.192 (LSTM) | 0.000 (GAT) |

## Reading
- **QLIKE:** HAR / HAR-X have the lowest QLIKE at every horizon. The deep models' QLIKE is inflated at
  h5/h10/h22 (LSTM 0.83-1.04, LSTM+GAT 0.79-1.32), consistent with a positivity-floor tail-collapse
  artifact at the S&P 500 variance scale (the 1e-2 floor tuned on the VN panels appears too low here). QLIKE
  for the deep models on this panel is therefore not yet trustworthy and should be re-checked with a higher /
  scale-aware floor before drawing conclusions.
- **Point error (MSE / RMSE / MAE / R2):** HAR-X leads at h1; the LSTM and LSTM+GAT lead from h5 onward
  (LSTM lowest MAE at h5 and significantly lower than HAR, p=0.021; LSTM+GAT lowest MSE/RMSE/MAE/R2 at h10 and
  h22). So on point-error metrics the deep models do improve over HAR at the longer horizons on this large
  market — the opposite of the VN panels where HAR led point error from h5 onward.
- **Graph (LSTM+GAT vs no-graph LSTM):** mixed. It does not help at h1 (QLIKE ns, MAE worse). It lowers the
  QLIKE blow-up at h5 (p<0.001) but raises it at h10 (p=0.035 worse). It gives a significantly lower MAE than
  the no-graph LSTM at h10 and h22 (p<0.001). No consistent QLIKE gain over the no-graph LSTM.

## Caveats before using these numbers
1. **Single seed only** — no multi-seed mean or dispersion; a full study needs >=3 seeds.
2. **QLIKE floor artifact** — the deep-model QLIKE values are inflated by the floor; re-run with a
   scale-aware floor before comparing QLIKE across models on S&P 500.
3. **No GARCH** on this panel yet; **--no-corr** (correlation-graph comparator not run).
4. This is **not** in the submitted paper (Vietnam-only); it is preview material for a possible later
   cross-market version.
