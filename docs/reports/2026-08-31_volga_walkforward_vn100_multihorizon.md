# VolGA Walk-Forward Evaluation on VN100 - Multi-Horizon Report

Date: 2026-08-31

## Scope

Out-of-sample walk-forward comparison of four volatility forecasters on the VN100 clean
enriched panel across four horizons (h1, h5, h10, h22). Models, in reporting order:

1. HAR - heterogeneous autoregressive (daily/weekly/monthly realized-variance baseline).
2. HAR-X - HAR with exogenous enriched features.
3. LSTM - sequence model on the enriched panel, no graph.
4. VolGA - the LSTM augmented with a volatility-to-Parkinson (vol to PK) graph attention layer
   over Top-5 directed spillover edges.

Target: Parkinson variance (sigma squared, not sigma). Loss/evaluation metric: QLIKE
(lower is better) with a positivity floor of 1e-08. Auxiliary metrics: RMSE, MAE, R2.

## Design

- Expanding-window walk-forward, 22 folds, monthly retrain cadence (K=21).
- 5 seeds: [42, 123, 2026, 7, 2024]. Nodes: 102. Lookback: 22. Validation tail: 66.
- Per-horizon out-of-sample test observations approximately 46,300.
- Leakage controls: per-fold train-only graph construction, per-fold train-only feature scalers,
  strictly forward test windows, date-clustered Diebold-Mariano inference.
- Training: epochs 16, patience 5, min_epochs 5,
  batch_size 32 (batched training, not per-sample).

## 1. Pooled QLIKE (lower is better)

Best model per horizon in bold.

| Horizon | HAR | HAR-X | LSTM | VolGA |
|---|---|---|---|---|
| h1 | 0.4983 | 0.5004 | 0.5025 | **0.4916** |
| h5 | 0.5671 | **0.5610** | 0.5763 | 0.5705 |
| h10 | 0.6005 | **0.6001** | 0.6096 | 0.6149 |
| h22 | 0.6392 | **0.6388** | 0.6479 | 0.6434 |

Full metric table:

| Horizon | Model | QLIKE | RMSE | MAE | R2 |
|---|---|---|---|---|---|
| h1 | HAR | 0.4983 | 4.867e-04 | 2.906e-04 | 0.2229 |
| h1 | HAR-X | 0.5004 | 4.861e-04 | 2.850e-04 | 0.2248 |
| h1 | LSTM | 0.5025 | 4.853e-04 | 2.821e-04 | 0.2275 |
| h1 | VolGA | 0.4916 | 4.846e-04 | 2.816e-04 | 0.2297 |
| h5 | HAR | 0.5671 | 5.123e-04 | 3.151e-04 | 0.1403 |
| h5 | HAR-X | 0.5610 | 5.101e-04 | 3.113e-04 | 0.1477 |
| h5 | LSTM | 0.5763 | 5.119e-04 | 3.068e-04 | 0.1416 |
| h5 | VolGA | 0.5705 | 5.114e-04 | 3.067e-04 | 0.1436 |
| h10 | HAR | 0.6005 | 5.246e-04 | 3.269e-04 | 0.0986 |
| h10 | HAR-X | 0.6001 | 5.242e-04 | 3.247e-04 | 0.1001 |
| h10 | LSTM | 0.6096 | 5.261e-04 | 3.185e-04 | 0.0936 |
| h10 | VolGA | 0.6149 | 5.267e-04 | 3.192e-04 | 0.0916 |
| h22 | HAR | 0.6392 | 5.371e-04 | 3.425e-04 | 0.0565 |
| h22 | HAR-X | 0.6388 | 5.370e-04 | 3.428e-04 | 0.0570 |
| h22 | LSTM | 0.6479 | 5.404e-04 | 3.370e-04 | 0.0448 |
| h22 | VolGA | 0.6434 | 5.397e-04 | 3.352e-04 | 0.0473 |

## 2. Graph marginal value: VolGA vs no-graph LSTM (date-clustered DM, QLIKE)

| Horizon | p-value | mean QLIKE diff | favors | significant |
|---|---|---|---|---|
| h1 | 0.008 | -0.01085 | VolGA | yes |
| h5 | 0.011 | -0.00582 | VolGA | yes |
| h10 | 0.229 | +0.00529 | LSTM | no |
| h22 | 0.107 | -0.00447 | VolGA | no |

The vol to PK graph is a significant marginal contributor at h1 and h5, and not significant at
h10 and h22. This horizon pattern is consistent with volatility-spillover findings in which
cross-asset transmission is strongest at short lead times.

## 3. Deep models vs HAR-X (date-clustered DM, QLIKE)

VolGA vs HAR-X:

| Horizon | p-value | mean QLIKE diff | favors | significant |
|---|---|---|---|---|
| h1 | 0.177 | -0.00881 | VolGA | no |
| h5 | 0.585 | +0.00945 | HAR-X | no |
| h10 | 0.520 | +0.01477 | HAR-X | no |
| h22 | 0.842 | +0.00456 | HAR-X | no |

LSTM vs HAR-X:

| Horizon | p-value | mean QLIKE diff | favors | significant |
|---|---|---|---|---|
| h1 | 0.809 | +0.00204 | HAR-X | no |
| h5 | 0.427 | +0.01527 | HAR-X | no |
| h10 | 0.622 | +0.00948 | HAR-X | no |
| h22 | 0.717 | +0.00904 | HAR-X | no |

Neither deep model significantly outperforms HAR-X at any horizon. LSTM is statistically
indistinguishable from HAR-X (equivalence) across all horizons.

## 4. Fit evidence (over/under-fit fold counts, out of 22)

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

No underfit folds at any horizon. Overfit folds are a minority and concentrate in the same
market windows across all four models, indicating regime difficulty rather than model-specific
instability. Per-fold, per-seed LSTM and VolGA learning curves (train/val MSE plus early-stop
best epoch) are stored in the source JSON; HAR and HAR-X are closed-form.

## 5. Conclusion

- The vol to PK graph adds statistically significant marginal value over the no-graph LSTM at
  short horizons (h1, h5) and not at longer horizons (h10, h22).
- No deep model (LSTM or VolGA) significantly outperforms the HAR-X baseline at any horizon.
- On point QLIKE, HAR-X attains the lowest value at h5, h10 and h22, and VolGA at h1.

## 6. Limitations

- Single walk-forward run with 5 seeds; seed dispersion is captured but broader
  seed ensembles are not.
- VN100 universe only; a VN30 walk-forward is pending and not included here.
- Point-QLIKE differences among HAR, HAR-X and the deep models are small relative to the
  date-clustered DM standard errors, so ranking by point metric alone is not decisive.
- Target is Parkinson variance; results are specific to this estimator and the clean enriched panel.

## Source data

- `results/walkforward_volga/walkforward_volga_vn100_h1.json`
- `results/walkforward_volga/walkforward_volga_vn100_h5.json`
- `results/walkforward_volga/walkforward_volga_vn100_h10.json`
- `results/walkforward_volga/walkforward_volga_vn100_h22.json`

Companion HTML dashboard: `docs/reports/2026-08-31_volga_walkforward_vn100_dashboard.html`.
