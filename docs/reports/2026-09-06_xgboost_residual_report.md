# XGBoost residual-ratio volatility study — results (2026-09-06)

## Summary (standard QLIKE, Parkinson variance)

| Panel | h | HAR-X | XGB-direct | XGB-resid-base | +market | +graph | VolGA(ref) | DM base vs HAR-X (p, favors) | base ticker/date win |
|---|---|---|---|---|---|---|---|---|---|
| vn100 | 1 | 0.5000 | 1.1305 | 0.5028 | 0.5021 | 0.5019 | 0.4879 | 0.501, B | 0.34/0.46 |
| vn100 | 5 | 0.5607 | 0.8652 | 0.5750 | 0.5749 | 0.5754 | 0.5662 | 0.343, B | 0.20/0.37 |
| vn100 | 10 | 0.5999 | 0.9264 | 0.6133 | 0.6151 | 0.6149 | 0.6127 | 0.502, B | 0.18/0.28 |
| vn100 | 22 | 0.6385 | 0.9873 | 0.6582 | 0.6595 | 0.6593 | 0.6476 | 0.594, B | 0.33/0.41 |
| vn30 | 1 | 0.4801 | 0.7660 | 0.4792 | 0.4787 | 0.4791 | 0.4740 | 0.791, A | 0.42/0.30 |
| vn30 | 5 | 0.5602 | 0.8256 | 0.5797 | 0.5786 | 0.5779 | 0.5672 | 0.489, B | 0.21/0.41 |
| vn30 | 10 | 0.6091 | 0.8975 | 0.6261 | 0.6254 | 0.6258 | 0.6061 | 0.598, B | 0.24/0.33 |
| vn30 | 22 | 0.6782 | 1.1169 | 0.7432 | 0.7597 | 0.7450 | 0.6882 | 0.465, B | 0.18/0.32 |

## Feature-group ablation

| Panel | h | base | +market | +graph | DM market vs base (p,fav) | DM graph vs market (p,fav) |
|---|---|---|---|---|---|---|
| vn100 | 1 | 0.5028 | 0.5021 | 0.5019 | 0.359, A | 0.401, A |
| vn100 | 5 | 0.5750 | 0.5749 | 0.5754 | 0.951, A | 0.257, B |
| vn100 | 10 | 0.6133 | 0.6151 | 0.6149 | 0.363, B | 0.455, A |
| vn100 | 22 | 0.6582 | 0.6595 | 0.6593 | 0.165, B | 0.581, A |
| vn30 | 1 | 0.4792 | 0.4787 | 0.4791 | 0.531, A | 0.050, B |
| vn30 | 5 | 0.5797 | 0.5786 | 0.5779 | 0.598, A | 0.364, A |
| vn30 | 10 | 0.6261 | 0.6254 | 0.6258 | 0.516, A | 0.468, B |
| vn30 | 22 | 0.7432 | 0.7597 | 0.7450 | 0.277, B | 0.315, A |

## Limit-lock robustness (XGB_resid_base)

| Panel | h | QLIKE (all) | non-lock | lock-only | lock QLIKE share | n_lock | QLIKE excl top-1% dates | (model=XGB_resid_base) |
|---|---|---|---|---|---|---|---|---|
| vn100 | 1 | 0.5028 | 0.4716 | 9.942 | 0.065 | 153 | 0.4567 | |
| vn100 | 5 | 0.5750 | 0.5471 | 9.026 | 0.052 | 153 | 0.5154 | |
| vn100 | 10 | 0.6133 | 0.5853 | 9.085 | 0.049 | 153 | 0.5549 | |
| vn100 | 22 | 0.6582 | 0.6303 | 9.057 | 0.045 | 153 | 0.5939 | |
| vn30 | 1 | 0.4792 | 0.4449 | 10.300 | 0.075 | 45 | 0.4283 | |
| vn30 | 5 | 0.5797 | 0.5503 | 8.884 | 0.054 | 45 | 0.4858 | |
| vn30 | 10 | 0.6261 | 0.5953 | 8.976 | 0.053 | 47 | 0.5422 | |
| vn30 | 22 | 0.7432 | 0.7145 | 8.662 | 0.042 | 46 | 0.6290 | |

## GO/NO-GO tally

```
{
  "n_panel_horizon": 8,
  "n_lower_qlike_than_harx": 1,
  "n_dm_significant_better": 0,
  "verdict": "NO-GO"
}
```

## Method

Expanding-window walk-forward on the delivered enriched panels (VN30, VN100), canonical split lb10 /
folds_target=7 — identical to the delivered `edge_hmatched` runs, so the HAR-X baseline scored here equals
the delivered HAR-X (verified: `test_harx_pooled_parity_with_edge_hmatched`) and VolGA/LSTM are comparable.
Target = Parkinson **variance** `pk[t+h]`; shared QLIKE floor + per-node positivity floor across every
model. Features are strictly causal (per-ticker stock history, cross-sectional market aggregates, and
train-only horizon-matched `directed_vol2pk` neighbour aggregates). The residual learner is trained on a
chronological out-of-fold HAR-X log-ratio target `z = log((y+eps)/(HARX_OOF+eps))`; the final forecast is
`HARX_forecast * exp(alpha*clip(zhat))` with `alpha`/`clip` chosen on validation QLIKE. XGBoost is CPU-only
(`hist`, one deterministic seed). All p-values are date-clustered Diebold–Mariano and, because many
panels×horizons×models are compared, are **exploratory** (unadjusted for multiplicity).

## Key results

- **HAR-X + XGBoost residual (primary):** does NOT improve OOS QLIKE. The point estimate is lower than
  HAR-X in only 1 of 8 panel-horizons (vn30 h1: 0.4792 vs 0.4801) and that difference is not significant
  (DM p=0.791). In the other 7 it is worse (DM favors HAR-X, none significant). Ticker and unique-date win
  rates versus HAR-X are below 0.5 everywhere (0.18–0.42 / 0.28–0.46) — HAR-X wins the majority of both
  tickers and dates in every cell.
- **Non-lock / top-1%-date robustness:** the non-improvement persists after excluding limit-lock ticker-date
  observations and after excluding the top-1% dates by QLIKE contribution (the ordering base ≥ HAR-X is
  unchanged). The result is not an artefact of limit-lock degeneracy (lock QLIKE share is only 4–8%).
- **Market and graph features:** add no value on top of the residual base — no ablation step is a
  significant improvement; on several cells the guardrail selects `alpha=0` when market/graph features are
  present, i.e. the correction collapses to plain HAR-X.
- **Direct XGBoost (log target):** far worse than HAR-X at every horizon (QLIKE 0.77–1.13 vs 0.48–0.68) and
  fails the fit gate on 4 of 8 panels — overfit at vn100 h1 (val→test QLIKE +48%) and underfit (negative
  R²) at the long horizons vn100 h10/h22 and vn30 h22. A flexible tree on the raw log-variance target does
  not generalise; the HAR structure is doing the real work.
- **VolGA (reference, from `edge_hmatched`):** the strongest deep model, but only marginally below HAR-X at
  h1 and above HAR-X at longer horizons — consistent with the project finding that the graph helps
  daily-horizon squared/abs error, not QLIKE.

## April-2025 shock cluster

20 unique April-2025 forecast-target dates, ~5.6–5.7% of ticker-date cells limit-locked.
QLIKE is ~3× the overall level (≈1.5–1.6). The elevation persists on the non-lock subset (≈1.0–1.1), so it
is a genuine market/economic shock, not merely Parkinson-proxy collapse under price limits. XGB_resid_graph
is slightly WORSE than HAR-X in the cluster (1.615 vs 1.581 on VN30; 1.539 vs 1.459 on VN100), with an
underprediction rate near 0.42 for both — the nonlinear correction does not help during the shock.

## Limitations

- VolGA / LSTM were not retrained (CPU-only mandate; the GPU is in use by a concurrent transformer agent).
  Their QLIKE and the VolGA-vs-HAR-X DM are cited from the delivered `edge_hmatched` artifacts; a cell-level
  DM of VolGA vs XGB+graph is not available (needs VolGA per-cell predictions).
- Multiple comparisons are unadjusted (exploratory).
- Single deterministic XGBoost seed (tree model); cross-machine bit-reproducibility not guaranteed.
- CatBoost/LightGBM not evaluated (not installed; the required ladder did not warrant them).

## GO/NO-GO decision

**NO-GO** for HAR-X + XGBoost residual-ratio and for direct XGBoost. No predeclared primary comparison is
significant; the point-estimate improvement is limited to one cell, is not significant, does not survive the
win-rate check, and market/graph features add nothing. Direct XGBoost overfits/underfits and harms OOS
QLIKE. HAR-X remains the parsimonious champion among tabular models; VolGA remains the reference deep model.
Per the guide, a more complex Transformer is not warranted by this XGBoost outcome.
