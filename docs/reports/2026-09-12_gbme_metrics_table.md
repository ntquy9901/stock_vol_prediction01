# Volatility forecasting — HAR / HARQ / GBM / GBME(GBM+Earning) / GBME(GBM+Earning)+graph (all metrics, DM-bolded)

GBM = gamma-loss Gradient Boosting Machine (HistGradientBoostingRegressor) on own-history features; GBME(GBM+Earning) = GBM + External earnings features; GBME(GBM+Earning)+graph adds the cross-firm correlation-graph neighbour aggregate. QLIKE is the primary loss (equal to the gamma deviance up to a constant); Diebold-Mariano tests use per-trading-date mean loss.

### S&P 500

MSE ×10⁻⁷ · RMSE/MAE ×10⁻⁴ · QLIKE unscaled. **Bold** on MSE/RMSE/MAE = numerically best (lowest) value in that horizon; **bold** on QLIKE = significantly best by date-clustered Diebold-Mariano (p<0.05).

| h | Model | MSE | RMSE | MAE | QLIKE |
|---|-------|-----|------|-----|-------|
| 1 | HAR | 5.754 | 7.586 | 2.132 | 0.3697 |
| 1 | HARQ | 5.648 | 7.515 | 2.151 | 0.3780 |
| 1 | GBM | 5.557 | 7.454 | 2.089 | 0.3570 |
| 1 | GBME(GBM+Earning) | **5.397** | **7.346** | 2.017 | **0.3172** |
| 1 | GBME(GBM+Earning)+graph | 5.424 | 7.365 | **2.010** | 0.3158 |
| | | | | | |
| 5 | HAR | 6.141 | 7.836 | 2.297 | 0.4364 |
| 5 | HARQ | 6.132 | 7.831 | 2.309 | 0.4377 |
| 5 | GBM | 6.117 | 7.821 | 2.255 | 0.4116 |
| 5 | GBME(GBM+Earning) | **5.972** | **7.728** | **2.180** | **0.3705** |
| 5 | GBME(GBM+Earning)+graph | 6.063 | 7.787 | 2.196 | 0.3715 |
| | | | | | |
| 10 | HAR | 6.214 | 7.883 | 2.356 | 0.4615 |
| 10 | HARQ | 6.205 | 7.877 | 2.358 | 0.4614 |
| 10 | GBM | 6.262 | 7.913 | 2.312 | 0.4307 |
| 10 | GBME(GBM+Earning) | **6.118** | **7.822** | **2.239** | **0.3912** |
| 10 | GBME(GBM+Earning)+graph | 6.240 | 7.899 | 2.272 | 0.3939 |
| | | | | | |
| 22 | HAR | 6.327 | 7.954 | 2.419 | 0.4855 |
| 22 | HARQ | 6.311 | 7.944 | 2.415 | 0.4831 |
| 22 | GBM | 6.233 | 7.895 | 2.359 | 0.4464 |
| 22 | GBME(GBM+Earning) | 6.098 | 7.809 | **2.277** | **0.4074** |
| 22 | GBME(GBM+Earning)+graph | **6.087** | **7.802** | 2.300 | 0.4110 |
| | | | | | |

*DM (QLIKE):* GBME vs GBM h1 p=0.000 · GBME vs GBM h5 p=0.000 · GBME vs GBM h10 p=0.000 · GBME vs GBM h22 p=0.000 (all significant → GBME QLIKE bolded).

### HOSE

MSE ×10⁻⁷ · RMSE/MAE ×10⁻⁴ · QLIKE unscaled. **Bold** on MSE/RMSE/MAE = numerically best (lowest) value in that horizon; **bold** on QLIKE = significantly best by date-clustered Diebold-Mariano (p<0.05).

| h | Model | MSE | RMSE | MAE | QLIKE |
|---|-------|-----|------|-----|-------|
| 1 | HAR | 6.604 | 8.126 | 5.193 | 1.8117 |
| 1 | HARQ | 6.389 | 7.993 | 5.062 | 1.7854 |
| 1 | GBM | **5.399** | **7.348** | **3.944** | **1.5679** |
| 1 | GBME(GBM+Earning) | 5.477 | 7.401 | 3.966 | 1.5730 |
| 1 | GBME(GBM+Earning)+graph | 5.555 | 7.453 | 3.980 | 1.5783 |
| | | | | | |
| 5 | HAR | 6.626 | 8.140 | 5.227 | 1.8218 |
| 5 | HARQ | 6.519 | 8.074 | 5.152 | 1.8075 |
| 5 | GBM | 5.796 | 7.613 | 4.265 | **1.6478** |
| 5 | GBME(GBM+Earning) | **5.795** | **7.613** | 4.276 | 1.6485 |
| 5 | GBME(GBM+Earning)+graph | 5.815 | 7.625 | **4.256** | 1.6494 |
| | | | | | |
| 10 | HAR | 6.660 | 8.161 | 5.250 | 1.8277 |
| 10 | HARQ | 6.589 | 8.118 | 5.198 | 1.8180 |
| 10 | GBM | 5.966 | 7.724 | 4.425 | **1.6842** |
| 10 | GBME(GBM+Earning) | 5.962 | 7.721 | 4.441 | 1.6862 |
| 10 | GBME(GBM+Earning)+graph | **5.959** | **7.720** | **4.407** | 1.6837 |
| | | | | | |
| 22 | HAR | 6.716 | 8.195 | 5.282 | 1.8355 |
| 22 | HARQ | 6.677 | 8.171 | 5.250 | 1.8299 |
| 22 | GBM | 6.211 | 7.881 | 4.615 | **1.7263** |
| 22 | GBME(GBM+Earning) | 6.211 | 7.881 | 4.627 | 1.7262 |
| 22 | GBME(GBM+Earning)+graph | **6.207** | **7.878** | **4.604** | 1.7267 |
| | | | | | |

*DM (QLIKE):* GBME vs GBM h1 p=0.000 · GBME vs GBM h5 p=0.214 · GBME vs GBM h10 p=0.122 · GBME vs GBM h22 p=0.919 (earnings does not beat GBM → own-history GBM QLIKE bolded).
