# VolGA Walk-Forward Cross-Market Comparison — VN30 vs VN100

Date: 2026-09-01

## 1. Scope and question

Both markets are evaluated with the identical VolGA walk-forward protocol. The question is
whether the vol→PK spillover graph adds statistically significant forecasting value over an
otherwise identical no-graph LSTM, and how that value depends on the size and composition of
the stock universe. VolGA denotes the LSTM augmented with a vol→PK graph attention layer
(Top-5 directed spillover edges). The two universes are VN30 (31 nodes) and VN100 (102 nodes).

## 2. Methodology (identical for both markets)

- Target: Parkinson-variance (σ², not σ), clean enriched panels
  (`data/processed_enriched/vn30`, `data/processed_enriched/vn100`).
- Design: expanding-window walk-forward, 22 monthly-retrain folds, 5 seeds
  [42, 123, 2026, 7, 2024], lookback 22, validation tail 66.
- Graph is rebuilt per fold from training data only; feature scalers are fit per fold on
  training data only.
- Models compared: HAR, HAR-X, LSTM (no graph), VolGA (LSTM + vol→PK graph).
- Inference: date-clustered Diebold-Mariano test on QLIKE, QLIKE positivity floor 1e-08.
- Retrain cadence: VN30 K=16, VN100 K=21. All other settings identical.

## 3. Per-horizon pooled QLIKE (lower is better)

Pooled over all folds, seeds and nodes. Best per market and horizon marked with an asterisk.

| Horizon | Model | VN30 QLIKE | VN100 QLIKE |
|---|---|---|---|
| h1  | HAR   | 0.4952* | 0.4983  |
| h1  | HAR-X | 0.4967  | 0.5004  |
| h1  | LSTM  | 0.5219  | 0.5025  |
| h1  | VolGA | 0.5146  | 0.4916* |
| h5  | HAR   | 0.5944  | 0.5671  |
| h5  | HAR-X | 0.5937* | 0.5610* |
| h5  | LSTM  | 0.6327  | 0.5763  |
| h5  | VolGA | 0.6150  | 0.5705  |
| h10 | HAR   | 0.6412  | 0.6005  |
| h10 | HAR-X | 0.6394  | 0.6001* |
| h10 | LSTM  | 0.6336* | 0.6096  |
| h10 | VolGA | 0.6485  | 0.6149  |
| h22 | HAR   | 0.7012  | 0.6392  |
| h22 | HAR-X | 0.6987* | 0.6388* |
| h22 | LSTM  | 0.7056  | 0.6479  |
| h22 | VolGA | 0.7053  | 0.6434  |

On point QLIKE, an econometric baseline (HAR or HAR-X) wins on VN30 at h1, h5, h22 and on
VN100 at h5, h10, h22. On VN100 the only horizon where a deep model attains the best pooled
QLIKE is h1 (VolGA), and on VN30 the only such horizon is h10 (LSTM).

## 4. Graph marginal value — VolGA vs no-graph LSTM (date-clustered DM, QLIKE)

| Horizon | VN30 p-value | VN30 favors | VN100 p-value | VN100 favors |
|---|---|---|---|---|
| h1  | 0.179 | VolGA | 0.008 | VolGA (significant) |
| h5  | 0.112 | VolGA | 0.011 | VolGA (significant) |
| h10 | 0.265 | LSTM  | 0.229 | LSTM  |
| h22 | 0.928 | VolGA | 0.107 | VolGA |

On VN100 the vol→PK graph is a statistically significant marginal contributor over the
no-graph LSTM at the two short horizons (h1 p=0.008, h5 p=0.011) and not significant at the
longer horizons. On VN30 the graph is not a statistically significant contributor at any
horizon; the smallest p-value is 0.112 at h5.

## 5. Deep models vs HAR-X (date-clustered DM, QLIKE)

Neither deep model significantly beats HAR-X at any horizon on either market.

| Horizon | VN30 VolGA-vs-HARX p | VN100 VolGA-vs-HARX p | VN30 LSTM-vs-HARX p | VN100 LSTM-vs-HARX p |
|---|---|---|---|---|
| h1  | 0.157 | 0.177 | 0.097 | 0.809 |
| h5  | 0.397 | 0.585 | 0.222 | 0.427 |
| h10 | 0.717 | 0.520 | 0.648 | 0.622 |
| h22 | 0.788 | 0.842 | 0.796 | 0.717 |

All p-values exceed 0.05. No deep model, with or without the graph, statistically outperforms
the HAR-X econometric baseline at any horizon on either universe.

## 6. Interpretation

The graph carries statistically significant marginal value over the no-graph LSTM on VN100 at
short horizons (h1, h5) but not on VN30 at any horizon. This ordering is not explained by
correlation magnitude: the VN30 universe has higher average pairwise return correlation than
VN100, yet it is the smaller VN30 universe on which the graph fails to add significant value.
The result is consistent with the project EDA finding that graph value is driven by node
breadth, liquidity, and the presence of stably estimable edges, rather than by correlation
magnitude alone. With only 31 nodes, the per-fold train-only vol→PK edge set on VN30 is both
smaller and less stable across retrain folds, which reduces the effective signal the attention
layer can exploit, whereas the 102-node VN100 panel supplies a broader and more stable edge set
at short horizons.

On both markets the graph effect fades with horizon, becoming non-significant by h10 on both
universes, which is consistent with cross-asset volatility transmission being strongest at short
lead times.

Across both markets, the econometric baselines remain competitive: no deep model beats HAR-X at
any horizon under the date-clustered DM test, and HAR or HAR-X wins the pooled point QLIKE at
most horizons.

## 7. Limitations

- Single 5-seed run per market and horizon; results are not averaged over independent data
  vintages.
- Two Vietnamese universes only (VN30, VN100); no external market is included in this
  comparison.
- VN30 contains only 31 nodes, which limits the statistical power of both the DM test and the
  per-fold graph construction on that universe.
- Retrain cadence differs between markets (VN30 K=16, VN100 K=21) as a function of available
  history; all other configuration is identical.

## 8. Sources

- `results/walkforward_volga/walkforward_volga_vn30_h{1,5,10,22}.json`
- `results/walkforward_volga/walkforward_volga_vn100_h{1,5,10,22}.json`
- `docs/reports/2026-08-31_volga_walkforward_vn30_dashboard.html`
- `docs/reports/2026-08-31_volga_walkforward_vn100_dashboard.html`
