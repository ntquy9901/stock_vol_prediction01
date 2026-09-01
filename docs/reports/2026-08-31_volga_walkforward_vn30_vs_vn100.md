# VolGA Walk-Forward Cross-Market Comparison - VN30 vs VN100 (full error-metric set)

Date: 2026-09-01

## 1. Scope and question

Both markets are evaluated with the identical VolGA walk-forward protocol. The question is whether the vol->PK spillover graph adds statistically significant forecasting value over an otherwise identical no-graph LSTM, and how that value depends on the size and composition of the stock universe. VolGA denotes the LSTM augmented with a vol->PK graph attention layer (Top-5 directed spillover edges). The universes are VN30 (31 nodes) and VN100 (102 nodes). All error metrics are reported: MSE, RMSE, MAE, QLIKE and R2. QLIKE is the primary loss; the others are secondary point-error metrics.

## 2. Methodology (identical for both markets)

- Target: Parkinson-variance (sigma squared, not sigma), clean enriched panels (`data/processed_enriched/vn30`, `data/processed_enriched/vn100`).
- Design: expanding-window walk-forward, 22 monthly-retrain folds, 5 seeds [42, 123, 2026, 7, 2024], lookback 22, validation tail 66.
- Graph is rebuilt per fold from training data only; feature scalers are fit per fold on training data only.
- Models compared: HAR, HAR-X, LSTM (no graph), VolGA (LSTM + vol->PK graph).
- Inference: date-clustered Diebold-Mariano test on QLIKE, squared error and absolute error; QLIKE positivity floor 1e-08.
- Retrain cadence: VN30 K=16, VN100 K=21. All other settings identical.
- Directional accuracy is not reported (not computed by the suite; unreliable for variance targets due to anti-persistence).

## 3. Full per-horizon error metrics by market

Best model per metric per market per horizon in bold. MSE/RMSE/MAE/QLIKE lower is better; R2 higher is better.

### VN30

| Horizon | Model | MSE | RMSE | MAE | QLIKE | R2 |
|---|---|---|---|---|---|---|
| h1 | HAR | 1.947e-07 | 4.413e-04 | 2.431e-04 | **0.4952** | 0.2271 |
| h1 | HAR-X | 1.938e-07 | 4.402e-04 | 2.402e-04 | 0.4967 | 0.2310 |
| h1 | LSTM | 1.905e-07 | 4.365e-04 | 2.378e-04 | 0.5219 | 0.2439 |
| h1 | VolGA | **1.890e-07** | **4.348e-04** | **2.362e-04** | 0.5146 | **0.2497** |
| h5 | HAR | 2.171e-07 | 4.659e-04 | 2.637e-04 | 0.5944 | 0.1453 |
| h5 | HAR-X | 2.156e-07 | 4.644e-04 | 2.614e-04 | **0.5937** | 0.1511 |
| h5 | LSTM | **2.130e-07** | **4.616e-04** | **2.564e-04** | 0.6327 | **0.1613** |
| h5 | VolGA | 2.135e-07 | 4.621e-04 | 2.573e-04 | 0.6150 | 0.1595 |
| h10 | HAR | 2.367e-07 | 4.865e-04 | 2.784e-04 | 0.6412 | 0.1066 |
| h10 | HAR-X | 2.363e-07 | 4.861e-04 | 2.771e-04 | 0.6394 | 0.1083 |
| h10 | LSTM | **2.351e-07** | **4.849e-04** | 2.778e-04 | **0.6336** | **0.1125** |
| h10 | VolGA | 2.358e-07 | 4.856e-04 | **2.758e-04** | 0.6485 | 0.1101 |
| h22 | HAR | 2.756e-07 | 5.250e-04 | 3.017e-04 | 0.7012 | 0.0682 |
| h22 | HAR-X | **2.751e-07** | **5.245e-04** | 3.015e-04 | **0.6987** | **0.0699** |
| h22 | LSTM | 2.823e-07 | 5.313e-04 | **3.010e-04** | 0.7056 | 0.0456 |
| h22 | VolGA | 2.829e-07 | 5.319e-04 | 3.011e-04 | 0.7053 | 0.0434 |

### VN100

| Horizon | Model | MSE | RMSE | MAE | QLIKE | R2 |
|---|---|---|---|---|---|---|
| h1 | HAR | 2.369e-07 | 4.867e-04 | 2.906e-04 | 0.4983 | 0.2229 |
| h1 | HAR-X | 2.363e-07 | 4.861e-04 | 2.850e-04 | 0.5004 | 0.2248 |
| h1 | LSTM | 2.355e-07 | 4.853e-04 | 2.821e-04 | 0.5025 | 0.2275 |
| h1 | VolGA | **2.348e-07** | **4.846e-04** | **2.816e-04** | **0.4916** | **0.2297** |
| h5 | HAR | 2.625e-07 | 5.123e-04 | 3.151e-04 | 0.5671 | 0.1403 |
| h5 | HAR-X | **2.602e-07** | **5.101e-04** | 3.113e-04 | **0.5610** | **0.1477** |
| h5 | LSTM | 2.621e-07 | 5.119e-04 | 3.068e-04 | 0.5763 | 0.1416 |
| h5 | VolGA | 2.615e-07 | 5.114e-04 | **3.067e-04** | 0.5705 | 0.1436 |
| h10 | HAR | 2.752e-07 | 5.246e-04 | 3.269e-04 | 0.6005 | 0.0986 |
| h10 | HAR-X | **2.748e-07** | **5.242e-04** | 3.247e-04 | **0.6001** | **0.1001** |
| h10 | LSTM | 2.767e-07 | 5.261e-04 | **3.185e-04** | 0.6096 | 0.0936 |
| h10 | VolGA | 2.774e-07 | 5.267e-04 | 3.192e-04 | 0.6149 | 0.0916 |
| h22 | HAR | 2.885e-07 | 5.371e-04 | 3.425e-04 | 0.6392 | 0.0565 |
| h22 | HAR-X | **2.883e-07** | **5.370e-04** | 3.428e-04 | **0.6388** | **0.0570** |
| h22 | LSTM | 2.921e-07 | 5.404e-04 | 3.370e-04 | 0.6479 | 0.0448 |
| h22 | VolGA | 2.913e-07 | 5.397e-04 | **3.352e-04** | 0.6434 | 0.0473 |

## 4. Per-seed dispersion (LSTM and VolGA, 5 seeds)

### VN30

| Horizon | Model | Metric | mean | std | min | max |
|---|---|---|---|---|---|---|
| h1 | LSTM | MSE | 1.910e-07 | 2.889e-10 | 1.906e-07 | 1.913e-07 |
| h1 | LSTM | RMSE | 4.370e-04 | 3.306e-07 | 4.366e-04 | 4.374e-04 |
| h1 | LSTM | MAE | 2.381e-04 | 5.045e-07 | 2.374e-04 | 2.390e-04 |
| h1 | LSTM | QLIKE | 0.5422 | 0.0162 | 0.5138 | 0.5607 |
| h1 | LSTM | R2 | 0.2421 | 0.0011 | 0.2406 | 0.2434 |
| h1 | VolGA | MSE | 1.895e-07 | 2.962e-10 | 1.890e-07 | 1.900e-07 |
| h1 | VolGA | RMSE | 4.353e-04 | 3.402e-07 | 4.348e-04 | 4.359e-04 |
| h1 | VolGA | MAE | 2.366e-04 | 8.935e-07 | 2.353e-04 | 2.378e-04 |
| h1 | VolGA | QLIKE | 0.5434 | 0.0063 | 0.5311 | 0.5494 |
| h1 | VolGA | R2 | 0.2480 | 0.0012 | 0.2460 | 0.2497 |
| h5 | LSTM | MSE | 2.135e-07 | 8.250e-10 | 2.121e-07 | 2.145e-07 |
| h5 | LSTM | RMSE | 4.621e-04 | 8.933e-07 | 4.605e-04 | 4.631e-04 |
| h5 | LSTM | MAE | 2.569e-04 | 1.888e-06 | 2.543e-04 | 2.598e-04 |
| h5 | LSTM | QLIKE | 0.6635 | 0.0221 | 0.6442 | 0.7069 |
| h5 | LSTM | R2 | 0.1595 | 0.0032 | 0.1556 | 0.1651 |
| h5 | VolGA | MSE | 2.139e-07 | 7.951e-10 | 2.126e-07 | 2.149e-07 |
| h5 | VolGA | RMSE | 4.625e-04 | 8.600e-07 | 4.610e-04 | 4.635e-04 |
| h5 | VolGA | MAE | 2.577e-04 | 9.361e-07 | 2.560e-04 | 2.585e-04 |
| h5 | VolGA | QLIKE | 0.6468 | 0.0227 | 0.6206 | 0.6828 |
| h5 | VolGA | R2 | 0.1580 | 0.0031 | 0.1541 | 0.1632 |
| h10 | LSTM | MSE | 2.356e-07 | 4.060e-10 | 2.351e-07 | 2.362e-07 |
| h10 | LSTM | RMSE | 4.853e-04 | 4.182e-07 | 4.849e-04 | 4.860e-04 |
| h10 | LSTM | MAE | 2.781e-04 | 1.782e-06 | 2.752e-04 | 2.805e-04 |
| h10 | LSTM | QLIKE | 0.6360 | 0.0043 | 0.6298 | 0.6412 |
| h10 | LSTM | R2 | 0.1109 | 0.0015 | 0.1083 | 0.1124 |
| h10 | VolGA | MSE | 2.362e-07 | 9.123e-10 | 2.351e-07 | 2.375e-07 |
| h10 | VolGA | RMSE | 4.860e-04 | 9.384e-07 | 4.849e-04 | 4.873e-04 |
| h10 | VolGA | MAE | 2.762e-04 | 1.248e-06 | 2.752e-04 | 2.785e-04 |
| h10 | VolGA | QLIKE | 0.6946 | 0.0981 | 0.6355 | 0.8897 |
| h10 | VolGA | R2 | 0.1085 | 0.0034 | 0.1037 | 0.1126 |
| h22 | LSTM | MSE | 2.828e-07 | 9.773e-10 | 2.809e-07 | 2.835e-07 |
| h22 | LSTM | RMSE | 5.318e-04 | 9.200e-07 | 5.300e-04 | 5.325e-04 |
| h22 | LSTM | MAE | 3.014e-04 | 1.978e-06 | 2.991e-04 | 3.038e-04 |
| h22 | LSTM | QLIKE | 0.7097 | 0.0052 | 0.6994 | 0.7141 |
| h22 | LSTM | R2 | 0.0437 | 0.0033 | 0.0414 | 0.0502 |
| h22 | VolGA | MSE | 2.835e-07 | 2.484e-10 | 2.831e-07 | 2.837e-07 |
| h22 | VolGA | RMSE | 5.324e-04 | 2.333e-07 | 5.321e-04 | 5.326e-04 |
| h22 | VolGA | MAE | 3.015e-04 | 1.484e-06 | 3.003e-04 | 3.043e-04 |
| h22 | VolGA | QLIKE | 0.7090 | 0.0037 | 0.7044 | 0.7147 |
| h22 | VolGA | R2 | 0.0415 | 0.0008 | 0.0407 | 0.0427 |

### VN100

| Horizon | Model | Metric | mean | std | min | max |
|---|---|---|---|---|---|---|
| h1 | LSTM | MSE | 2.362e-07 | 4.274e-10 | 2.355e-07 | 2.369e-07 |
| h1 | LSTM | RMSE | 4.860e-04 | 4.397e-07 | 4.853e-04 | 4.867e-04 |
| h1 | LSTM | MAE | 2.827e-04 | 1.477e-06 | 2.806e-04 | 2.846e-04 |
| h1 | LSTM | QLIKE | 0.5307 | 0.0200 | 0.5070 | 0.5584 |
| h1 | LSTM | R2 | 0.2251 | 0.0014 | 0.2230 | 0.2274 |
| h1 | VolGA | MSE | 2.355e-07 | 5.606e-10 | 2.345e-07 | 2.361e-07 |
| h1 | VolGA | RMSE | 4.853e-04 | 5.779e-07 | 4.843e-04 | 4.859e-04 |
| h1 | VolGA | MAE | 2.822e-04 | 2.607e-06 | 2.793e-04 | 2.871e-04 |
| h1 | VolGA | QLIKE | 0.5030 | 0.0038 | 0.4979 | 0.5071 |
| h1 | VolGA | R2 | 0.2274 | 0.0018 | 0.2255 | 0.2308 |
| h5 | LSTM | MSE | 2.625e-07 | 4.215e-10 | 2.620e-07 | 2.631e-07 |
| h5 | LSTM | RMSE | 5.123e-04 | 4.113e-07 | 5.119e-04 | 5.130e-04 |
| h5 | LSTM | MAE | 3.072e-04 | 1.931e-06 | 3.043e-04 | 3.104e-04 |
| h5 | LSTM | QLIKE | 0.5895 | 0.0152 | 0.5727 | 0.6102 |
| h5 | LSTM | R2 | 0.1403 | 0.0014 | 0.1382 | 0.1418 |
| h5 | VolGA | MSE | 2.621e-07 | 5.448e-10 | 2.614e-07 | 2.628e-07 |
| h5 | VolGA | RMSE | 5.120e-04 | 5.320e-07 | 5.113e-04 | 5.127e-04 |
| h5 | VolGA | MAE | 3.071e-04 | 2.171e-06 | 3.033e-04 | 3.092e-04 |
| h5 | VolGA | QLIKE | 0.5780 | 0.0129 | 0.5666 | 0.6018 |
| h5 | VolGA | R2 | 0.1415 | 0.0018 | 0.1392 | 0.1439 |
| h10 | LSTM | MSE | 2.775e-07 | 3.582e-10 | 2.770e-07 | 2.780e-07 |
| h10 | LSTM | RMSE | 5.267e-04 | 3.400e-07 | 5.263e-04 | 5.273e-04 |
| h10 | LSTM | MAE | 3.192e-04 | 3.127e-07 | 3.187e-04 | 3.196e-04 |
| h10 | LSTM | QLIKE | 0.6319 | 0.0283 | 0.6069 | 0.6813 |
| h10 | LSTM | R2 | 0.0912 | 0.0012 | 0.0894 | 0.0928 |
| h10 | VolGA | MSE | 2.779e-07 | 1.087e-09 | 2.769e-07 | 2.798e-07 |
| h10 | VolGA | RMSE | 5.272e-04 | 1.030e-06 | 5.262e-04 | 5.290e-04 |
| h10 | VolGA | MAE | 3.197e-04 | 1.503e-06 | 3.171e-04 | 3.214e-04 |
| h10 | VolGA | QLIKE | 0.6312 | 0.0325 | 0.6052 | 0.6951 |
| h10 | VolGA | R2 | 0.0897 | 0.0036 | 0.0835 | 0.0932 |
| h22 | LSTM | MSE | 2.928e-07 | 1.064e-09 | 2.911e-07 | 2.944e-07 |
| h22 | LSTM | RMSE | 5.411e-04 | 9.831e-07 | 5.396e-04 | 5.426e-04 |
| h22 | LSTM | MAE | 3.375e-04 | 1.567e-06 | 3.347e-04 | 3.393e-04 |
| h22 | LSTM | QLIKE | 0.6511 | 0.0068 | 0.6400 | 0.6610 |
| h22 | LSTM | R2 | 0.0423 | 0.0035 | 0.0371 | 0.0479 |
| h22 | VolGA | MSE | 2.921e-07 | 1.146e-09 | 2.905e-07 | 2.936e-07 |
| h22 | VolGA | RMSE | 5.404e-04 | 1.060e-06 | 5.390e-04 | 5.418e-04 |
| h22 | VolGA | MAE | 3.357e-04 | 2.200e-06 | 3.322e-04 | 3.385e-04 |
| h22 | VolGA | QLIKE | 0.6461 | 0.0028 | 0.6433 | 0.6506 |
| h22 | VolGA | R2 | 0.0448 | 0.0037 | 0.0399 | 0.0498 |

## 5. Graph marginal value - VolGA vs no-graph LSTM (date-clustered DM, all three bases)

| Market | Horizon | QLIKE p | QLIKE favors | SE p | SE favors | AE p | AE favors |
|---|---|---|---|---|---|---|---|
| VN30 | h1 | 0.179 | VolGA | 0.032 | VolGA | 0.005 | VolGA |
| VN30 | h5 | 0.112 | VolGA | 0.338 | LSTM | 0.052 | LSTM |
| VN30 | h10 | 0.265 | LSTM | 0.484 | LSTM | 0.010 | VolGA |
| VN30 | h22 | 0.928 | VolGA | 0.484 | LSTM | 0.963 | LSTM |
| VN100 | h1 | 0.008 | VolGA | 0.232 | VolGA | 0.274 | VolGA |
| VN100 | h5 | 0.011 | VolGA | 0.064 | VolGA | 0.661 | VolGA |
| VN100 | h10 | 0.229 | LSTM | 0.310 | LSTM | 0.251 | LSTM |
| VN100 | h22 | 0.107 | VolGA | 0.147 | VolGA | 0.016 | VolGA |

The QLIKE, squared-error and absolute-error tests do not always agree. On VN30 the graph is not significant on QLIKE at any horizon, yet the squared-error and absolute-error tests can reach significance at short horizons; on VN100 the graph is significant on QLIKE at the two short horizons. Reporting all three bases shows the marginal-value verdict is loss-function-dependent.

## 6. Deep models vs HAR-X (date-clustered DM, QLIKE)

| Horizon | VN30 VolGA-vs-HARX p | VN100 VolGA-vs-HARX p | VN30 LSTM-vs-HARX p | VN100 LSTM-vs-HARX p |
|---|---|---|---|---|
| h1 | 0.157 | 0.177 | 0.097 | 0.809 |
| h5 | 0.397 | 0.585 | 0.222 | 0.427 |
| h10 | 0.717 | 0.520 | 0.648 | 0.622 |
| h22 | 0.788 | 0.842 | 0.796 | 0.717 |

All QLIKE p-values exceed 0.05. No deep model, with or without the graph, statistically outperforms the HAR-X econometric baseline on QLIKE at any horizon on either universe.

## 7. Interpretation

QLIKE (scale-invariant, tail-sensitive) is the primary loss; RMSE, MAE, MSE and R2 are secondary point-error metrics, with R2 ranking matching MSE/RMSE by construction. On point error, the deep models often lead on RMSE, MAE and R2 (they reduce mean and tail point error), while on QLIKE an econometric baseline frequently leads because QLIKE penalises proportional error. This is why the metric-by-metric winner in Section 3 is not uniform and why the DM verdict in Section 5 depends on the loss basis.

The graph carries statistically significant marginal value over the no-graph LSTM on VN100 at short horizons (QLIKE basis) but not on VN30 on QLIKE at any horizon. This ordering is not explained by correlation magnitude: VN30 has higher average pairwise return correlation than VN100, yet it is the smaller VN30 universe on which the graph fails to add significant QLIKE value. The result is consistent with the project EDA finding that graph value is driven by node breadth, liquidity and the presence of stably estimable edges rather than correlation magnitude alone.

## 8. Limitations

- Single 5-seed run per market and horizon; results are not averaged over independent data vintages.
- Two Vietnamese universes only (VN30, VN100); no external market is included in this comparison.
- VN30 contains only 31 nodes, which limits the statistical power of both the DM test and the per-fold graph construction on that universe.
- Retrain cadence differs between markets (VN30 K=16, VN100 K=21) as a function of available history; all other configuration is identical.

## 9. Sources

- `results/walkforward_volga/walkforward_volga_vn30_h{1,5,10,22}.json`
- `results/walkforward_volga/walkforward_volga_vn100_h{1,5,10,22}.json`
- `docs/reports/2026-08-31_volga_walkforward_vn30_dashboard.html`
- `docs/reports/2026-08-31_volga_walkforward_vn100_dashboard.html`
