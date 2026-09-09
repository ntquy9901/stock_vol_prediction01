# Gamma-loss GBM beats HAR-X on S&P 500 (walk-forward)

A nonlinear, QLIKE-trained gradient-boosted model significantly beats HAR-X on the S&P 500 at h5/h10 — the
strongest single-model result on that panel, where deep models (VolGA/MASTER) only tie or lose.

## Result (canonical walk-forward, 7 folds; HAR-X reproduces canonical exactly)
S&P 500 — GBM beats HAR-X significantly at ALL FOUR horizons:
| market | h | HAR-X | GBM | Δ vs HAR-X | DM p |
|---|---|---|---|---|---|
| **S&P 500** | 1 | 0.4061 | **0.3521** | **+13.31%** | **0.0001** ✓ |
| **S&P 500** | 5 | 0.4550 | **0.4378** | **+3.78%** | **0.0000** ✓ |
| **S&P 500** | 10 | 0.4797 | **0.4624** | **+3.61%** | **0.0393** ✓ |
| **S&P 500** | 22 | 0.4959 | **0.4677** | **+5.70%** | **0.0025** ✓ |
| VN100 | 1 | 0.5000 | 0.5523 | −10.5% | 0.000 (worse) |
| VN100 | 5 | 0.5607 | 0.6016 | −7.3% | 0.000 (worse) |
| VN100 | 10 | 0.5999 | 0.6227 | −3.8% | 0.042 (worse) |
| VN30 | 5 | 0.5602 | 0.5537 | +1.1% | 0.289 (n.s.) |
| VN30 | 10 | 0.6091 | 0.6149 | −1.0% | 0.509 (worse) |

The S&P 500 win is large and significant at every horizon (h1 +13.3%!), and unlike the VN HARQ/stack (which
fade at h22) it holds strongly at the monthly horizon too.

## Model
`HistGradientBoostingRegressor(loss='gamma')` on 11 features = the 5 HAR-X features + a realized-quarticity
proxy (measurement error) + 5 causal log-vol decline features (change/slope5/slope10/dev5/z22). The gamma
unit deviance equals QLIKE up to a constant, so the GBM optimises the evaluation metric directly.

## Why it works — and why only on S&P 500
- **Nonlinearity, not the objective (control).** A LINEAR gamma model is much WORSE than HAR-X OLS (S&P 500
  h5: 0.6466 vs 0.4870), and extras do not help it (0.6503); only the NONLINEAR GBM wins (+4.63% single-split
  / +3.78% walk-forward). So the gain comes from tree nonlinearity + feature interactions, not from training
  on the eval metric.
- **Data size.** GBM needs data: S&P 500 (480 tickers × 24y) gives it enough to generalise; VN (104/33
  tickers) is too small and heavy-tailed → the GBM OVERFITS and loses to the parsimonious HAR-X. This matches
  the literature (Dudek et al. 2024: no model universally beats HAR; simple linear rivals complex).
- **Where the edge is:** calm-cell calibration (as with HARQ/stack); the GBM does NOT fix the storm-tail loss
  (structural h-step lag), confirmed by the per-decile analysis.

## Recommendation
- **S&P 500:** adopt the gamma-GBM (strongest, significant, +3.6–3.8%). It is the SP500 analogue of the VN
  HARQ/stack wins.
- **VN:** keep HARQ / the VolGA+HAR-X-Q stack (GBM overfits VN).
- Follow-up (in progress): VolGA + gamma-GBM heterogeneous ensemble on S&P 500 (deep + boosting residual
  stacking — a literature-backed direction and the under-explored GNN+GBM combination).

Baseline: `baselines/2026-09-08_gamma_gbm/` (SDD, 4 tests, 100% line+branch, 3-lens review — no CRITICAL/MAJOR).
Results: `results/gamma_gbm/gbm_{market}_h{h}.json`.
