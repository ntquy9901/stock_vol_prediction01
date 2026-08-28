# Fairest-test results (masked panel + 5 features + directed vol->PK + weighted-GAT 2-hop + HAR-X)

VN30 + VN100, 5 seeds, all metrics + date-clustered DM. Floor 1e-3*t_mean (QLIKE floor-caveat: the
LSTM QLIKE tail-collapse at some short horizons is a floor artifact — under a sensible 1e-2 floor all
models tie on QLIKE, see 2026-08-22_lstm_qlike_blowup_diagnosis.md; MSE/RMSE/MAE/R2 are floor-invariant).

## VN100

### vn100 h1 (N=102, test_dates=454, obs=46308, 5 seeds)

| Model | MSE(e-6) | RMSE(e-3) | MAE(e-4) | QLIKE | R2 |
|---|---:|---:|---:|---:|---:|
| HAR (3f lin) | 0.237 | 0.487 | 2.932 | 0.5004 | 0.2226 |
| HAR-X (5f lin) | 0.237 | 0.487 | 2.898 | 0.5327 | 0.2236 |
| LSTM (5f) | 0.237 | 0.487 | 2.821 | 1.0238 | 0.2224 |
| LSTM+GAT (5f) | 0.236 | 0.486 | 2.819 | 0.5411 | 0.2251 |

DM date-clustered: **HARX vs HAR** (qlike:p=0.116(B), se:p=0.851(A), ae:p=0.014(A)); **LSTM vs HARX** (qlike:p=0.000(B), se:p=0.801(B), ae:p=0.000(A)); **LSTM+GAT vs LSTM** (qlike:p=0.000(A), se:p=0.131(A), ae:p=0.675(A))

### vn100 h5 (N=102, test_dates=453, obs=46206, 5 seeds)

| Model | MSE(e-6) | RMSE(e-3) | MAE(e-4) | QLIKE | R2 |
|---|---:|---:|---:|---:|---:|
| HAR (3f lin) | 0.263 | 0.513 | 3.193 | 0.5694 | 0.1392 |
| HAR-X (5f lin) | 0.261 | 0.510 | 3.160 | 0.5633 | 0.1466 |
| LSTM (5f) | 0.264 | 0.514 | 3.090 | 0.5900 | 0.1361 |
| LSTM+GAT (5f) | 0.263 | 0.513 | 3.103 | 0.5691 | 0.1371 |

DM date-clustered: **HARX vs HAR** (qlike:p=0.097(A), se:p=0.010(A), ae:p=0.000(A)); **LSTM vs HARX** (qlike:p=0.095(B), se:p=0.075(B), ae:p=0.000(A)); **LSTM+GAT vs LSTM** (qlike:p=0.022(A), se:p=0.445(A), ae:p=0.002(B))

### vn100 h10 (N=102, test_dates=453, obs=46206, 5 seeds)

| Model | MSE(e-6) | RMSE(e-3) | MAE(e-4) | QLIKE | R2 |
|---|---:|---:|---:|---:|---:|
| HAR (3f lin) | 0.276 | 0.525 | 3.322 | 0.6024 | 0.0967 |
| HAR-X (5f lin) | 0.275 | 0.525 | 3.306 | 0.6023 | 0.0978 |
| LSTM (5f) | 0.280 | 0.529 | 3.282 | 0.6071 | 0.0837 |
| LSTM+GAT (5f) | 0.280 | 0.529 | 3.276 | 0.6072 | 0.0822 |

DM date-clustered: **HARX vs HAR** (qlike:p=0.978(A), se:p=0.644(A), ae:p=0.171(A)); **LSTM vs HARX** (qlike:p=0.597(B), se:p=0.022(B), ae:p=0.201(A)); **LSTM+GAT vs LSTM** (qlike:p=0.886(B), se:p=0.176(B), ae:p=0.064(A))

### vn100 h22 (N=102, test_dates=452, obs=46104, 5 seeds)

| Model | MSE(e-6) | RMSE(e-3) | MAE(e-4) | QLIKE | R2 |
|---|---:|---:|---:|---:|---:|
| HAR (3f lin) | 0.289 | 0.538 | 3.489 | 0.6405 | 0.0546 |
| HAR-X (5f lin) | 0.289 | 0.538 | 3.486 | 0.6405 | 0.0544 |
| LSTM (5f) | 0.298 | 0.546 | 3.468 | 0.6518 | 0.0257 |
| LSTM+GAT (5f) | 0.300 | 0.548 | 3.563 | 0.6544 | 0.0178 |

DM date-clustered: **HARX vs HAR** (qlike:p=0.932(A), se:p=0.783(B), ae:p=0.442(A)); **LSTM vs HARX** (qlike:p=0.429(B), se:p=0.002(B), ae:p=0.611(A)); **LSTM+GAT vs LSTM** (qlike:p=0.658(B), se:p=0.056(B), ae:p=0.000(B))

## VN30

### vn30 h1 (N=31, test_dates=326, obs=10106, 5 seeds)

| Model | MSE(e-6) | RMSE(e-3) | MAE(e-4) | QLIKE | R2 |
|---|---:|---:|---:|---:|---:|
| HAR (3f lin) | 0.195 | 0.441 | 2.416 | 0.5100 | 0.2231 |
| HAR-X (5f lin) | 0.193 | 0.439 | 2.389 | 0.5206 | 0.2308 |
| LSTM (5f) | 0.193 | 0.439 | 2.408 | 1.2733 | 0.2297 |
| LSTM+GAT (5f) | 0.191 | 0.437 | 2.367 | 0.8190 | 0.2368 |

DM date-clustered: **HARX vs HAR** (qlike:p=0.475(B), se:p=0.443(A), ae:p=0.089(A)); **LSTM vs HARX** (qlike:p=0.000(B), se:p=0.912(B), ae:p=0.214(B)); **LSTM+GAT vs LSTM** (qlike:p=0.000(A), se:p=0.003(A), ae:p=0.000(A))

### vn30 h5 (N=31, test_dates=323, obs=10013, 5 seeds)

| Model | MSE(e-6) | RMSE(e-3) | MAE(e-4) | QLIKE | R2 |
|---|---:|---:|---:|---:|---:|
| HAR (3f lin) | 0.215 | 0.464 | 2.603 | 0.5962 | 0.1442 |
| HAR-X (5f lin) | 0.214 | 0.463 | 2.583 | 0.5965 | 0.1497 |
| LSTM (5f) | 0.216 | 0.465 | 2.632 | 0.7214 | 0.1397 |
| LSTM+GAT (5f) | 0.215 | 0.463 | 2.651 | 0.6066 | 0.1467 |

DM date-clustered: **HARX vs HAR** (qlike:p=0.968(B), se:p=0.129(A), ae:p=0.026(A)); **LSTM vs HARX** (qlike:p=0.126(B), se:p=0.128(B), ae:p=0.000(B)); **LSTM+GAT vs LSTM** (qlike:p=0.141(A), se:p=0.024(A), ae:p=0.015(B))

### vn30 h10 (N=31, test_dates=323, obs=10013, 5 seeds)

| Model | MSE(e-6) | RMSE(e-3) | MAE(e-4) | QLIKE | R2 |
|---|---:|---:|---:|---:|---:|
| HAR (3f lin) | 0.230 | 0.480 | 2.738 | 0.6423 | 0.1018 |
| HAR-X (5f lin) | 0.230 | 0.480 | 2.733 | 0.6428 | 0.1028 |
| LSTM (5f) | 0.231 | 0.481 | 2.721 | 0.6588 | 0.0995 |
| LSTM+GAT (5f) | 0.233 | 0.482 | 2.725 | 0.6565 | 0.0931 |

DM date-clustered: **HARX vs HAR** (qlike:p=0.920(B), se:p=0.614(A), ae:p=0.553(A)); **LSTM vs HARX** (qlike:p=0.249(B), se:p=0.528(B), ae:p=0.340(A)); **LSTM+GAT vs LSTM** (qlike:p=0.506(A), se:p=0.030(B), ae:p=0.625(B))

### vn30 h22 (N=31, test_dates=321, obs=9951, 5 seeds)

| Model | MSE(e-6) | RMSE(e-3) | MAE(e-4) | QLIKE | R2 |
|---|---:|---:|---:|---:|---:|
| HAR (3f lin) | 0.228 | 0.477 | 2.789 | 0.6431 | 0.0696 |
| HAR-X (5f lin) | 0.227 | 0.477 | 2.785 | 0.6422 | 0.0723 |
| LSTM (5f) | 0.238 | 0.488 | 2.904 | 0.6550 | 0.0271 |
| LSTM+GAT (5f) | 0.238 | 0.488 | 2.884 | 0.6548 | 0.0261 |

DM date-clustered: **HARX vs HAR** (qlike:p=0.807(A), se:p=0.273(A), ae:p=0.741(A)); **LSTM vs HARX** (qlike:p=0.318(B), se:p=0.010(B), ae:p=0.001(B)); **LSTM+GAT vs LSTM** (qlike:p=0.937(A), se:p=0.723(B), ae:p=0.021(A))

