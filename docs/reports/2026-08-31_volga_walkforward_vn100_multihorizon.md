# VolGA Walk-Forward Evaluation on VN100 - Multi-Horizon Report (full error-metric set)

Date: 2026-09-01

## Scope

Out-of-sample walk-forward comparison of four volatility forecasters on the VN100 clean enriched panel across four horizons (h1, h5, h10, h22). Models, in reporting order: HAR (heterogeneous autoregressive baseline), HAR-X (HAR with exogenous enriched features), LSTM (sequence model, no graph), and VolGA (the LSTM augmented with a volatility-to-Parkinson graph attention layer over Top-5 directed spillover edges).

Target: Parkinson variance (sigma squared, not sigma). Primary loss: QLIKE (scale-invariant, tail-sensitive, lower is better) with positivity floor 1e-08. Secondary point-error metrics: MSE, RMSE, MAE (lower is better) and R2 (higher is better). Directional accuracy is not reported: the walk-forward suite did not compute it, and for variance targets it has known anti-persistence issues; this is stated explicitly rather than omitted silently.

## Design

- Expanding-window walk-forward, 22 folds, monthly retrain cadence (K=21).
- 5 seeds: [42, 123, 2026, 7, 2024]. Nodes: 102. Lookback: 22. Validation tail: 66.
- Leakage controls: per-fold train-only graph construction, per-fold train-only feature scalers, strictly forward test windows, date-clustered Diebold-Mariano inference.
- Training: epochs 16, patience 5, min_epochs 5, batch_size 32 (batched training, not per-sample).

## 1. Full per-horizon error metrics (all models, all metrics)

Best model per metric per horizon in bold. MSE/RMSE/MAE/QLIKE lower is better; R2 higher is better.

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

## 2. Per-seed dispersion (LSTM and VolGA, 5 seeds)

Mean, standard deviation, min and max across the 5 seeds for each metric. HAR and HAR-X are closed-form (seed-independent). These are per-seed statistics, so the QLIKE mean differs slightly from the prediction-pooled QLIKE in Section 1.

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

## 3. Diebold-Mariano significance on all three loss bases (QLIKE, squared error, absolute error)

Date-clustered Diebold-Mariano test for every model pair on each loss basis. Reporting all three bases shows whether the verdict is loss-function-dependent.

VolGA vs LSTM (A=VolGA, B=LSTM):

| Horizon | QLIKE p | QLIKE favors | SE p | SE favors | AE p | AE favors |
|---|---|---|---|---|---|---|
| h1 | 0.008 | VolGA | 0.232 | VolGA | 0.274 | VolGA |
| h5 | 0.011 | VolGA | 0.064 | VolGA | 0.661 | VolGA |
| h10 | 0.229 | LSTM | 0.310 | LSTM | 0.251 | LSTM |
| h22 | 0.107 | VolGA | 0.147 | VolGA | 0.016 | VolGA |

VolGA vs HAR-X (A=VolGA, B=HAR-X):

| Horizon | QLIKE p | QLIKE favors | SE p | SE favors | AE p | AE favors |
|---|---|---|---|---|---|---|
| h1 | 0.177 | VolGA | 0.274 | VolGA | 0.000 | VolGA |
| h5 | 0.585 | HAR-X | 0.431 | HAR-X | 0.005 | VolGA |
| h10 | 0.520 | HAR-X | 0.203 | HAR-X | 0.032 | VolGA |
| h22 | 0.842 | HAR-X | 0.369 | HAR-X | 0.094 | VolGA |

LSTM vs HAR-X (A=LSTM, B=HAR-X):

| Horizon | QLIKE p | QLIKE favors | SE p | SE favors | AE p | AE favors |
|---|---|---|---|---|---|---|
| h1 | 0.809 | HAR-X | 0.516 | LSTM | 0.002 | LSTM |
| h5 | 0.427 | HAR-X | 0.236 | HAR-X | 0.008 | LSTM |
| h10 | 0.622 | HAR-X | 0.323 | HAR-X | 0.017 | LSTM |
| h22 | 0.717 | HAR-X | 0.274 | HAR-X | 0.178 | LSTM |

HAR-X vs HAR (A=HAR-X, B=HAR):

| Horizon | QLIKE p | QLIKE favors | SE p | SE favors | AE p | AE favors |
|---|---|---|---|---|---|---|
| h1 | 0.728 | HAR | 0.639 | HAR-X | 0.000 | HAR-X |
| h5 | 0.079 | HAR-X | 0.001 | HAR-X | 0.000 | HAR-X |
| h10 | 0.923 | HAR-X | 0.438 | HAR-X | 0.013 | HAR-X |
| h22 | 0.664 | HAR-X | 0.468 | HAR-X | 0.397 | HAR |

## 4. Error-metric interpretation

QLIKE is the primary loss: scale-invariant and tail-sensitive, suited to variance targets whose magnitude varies across stocks and regimes. RMSE and MSE weight large errors quadratically, MAE weights them linearly, and R2 rescales MSE against target variance, so the R2 ranking matches MSE and RMSE by construction.

On VN100 the QLIKE-best model coincides with the RMSE-best and MAE-best model at h1 (VolGA best on all three). The ranking flips between metrics at h5 (QLIKE favors HAR-X, RMSE HAR-X, MAE VolGA); h10 (QLIKE favors HAR-X, RMSE HAR-X, MAE LSTM); h22 (QLIKE favors HAR-X, RMSE HAR-X, MAE VolGA). Where the ranking flips, the point-error metrics tend to favour the deep models (which reduce mean and tail point error), while QLIKE can still favour an econometric baseline because it penalises proportional error rather than absolute error. The DM tables in Section 3 show this explicitly: on some pairs and horizons the squared-error and absolute-error tests reach significance while the QLIKE test does not, so the marginal-value verdict is loss-function-dependent.

## 5. Fit evidence (over/under-fit fold counts, out of 22)

| Horizon | Model | n_ok | n_overfit | n_underfit |
|---|---|---|---|---|
| h1 | HAR | 19 | 3 | 0 |
| h1 | HAR-X | 17 | 5 | 0 |
| h1 | LSTM | 18 | 4 | 0 |
| h1 | VolGA | 18 | 4 | 0 |
| h5 | HAR | 18 | 4 | 0 |
| h5 | HAR-X | 19 | 3 | 0 |
| h5 | LSTM | 18 | 4 | 0 |
| h5 | VolGA | 18 | 4 | 0 |
| h10 | HAR | 15 | 7 | 0 |
| h10 | HAR-X | 16 | 6 | 0 |
| h10 | LSTM | 17 | 5 | 0 |
| h10 | VolGA | 16 | 6 | 0 |
| h22 | HAR | 16 | 6 | 0 |
| h22 | HAR-X | 16 | 6 | 0 |
| h22 | LSTM | 17 | 5 | 0 |
| h22 | VolGA | 17 | 5 | 0 |

Per-fold, per-seed LSTM and VolGA learning curves (train/val MSE plus early-stop best epoch) are stored in the source JSON; HAR and HAR-X are closed-form.

## 6. Conclusion

- Graph marginal value (VolGA vs no-graph LSTM, QLIKE DM): significant for VolGA at h1, h5.
- Point QLIKE winners by horizon: h1 VolGA, h5 HAR-X, h10 HAR-X, h22 HAR-X.
- No deep model significantly beats HAR-X on QLIKE at any horizon (see Section 3).
- The verdict can differ across loss bases; squared-error and absolute-error DM results are reported in full alongside QLIKE.

## 7. Limitations

- Single walk-forward run with 5 seeds; seed dispersion is captured but broader seed ensembles are not.
- VN100 universe only; the VN30 walk-forward is reported in the companion cross-market document.
- Point-metric differences among the models are small relative to the date-clustered DM standard errors, so ranking by point metric alone is not decisive.
- Target is Parkinson variance; results are specific to this estimator and the clean enriched panel.

## Source data

- `results/walkforward_volga/walkforward_volga_vn100_h1.json`
- `results/walkforward_volga/walkforward_volga_vn100_h5.json`
- `results/walkforward_volga/walkforward_volga_vn100_h10.json`
- `results/walkforward_volga/walkforward_volga_vn100_h22.json`

Companion HTML dashboard: `docs/reports/2026-08-31_volga_walkforward_vn100_dashboard.html`.
