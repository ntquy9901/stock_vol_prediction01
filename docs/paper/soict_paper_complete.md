# Multi-Horizon Stock Volatility Forecasting: HAR, LSTM and Graph-Attention Models on Vietnamese Equities

*SOICT submission (markdown draft; transcribed to the SOICT/Overleaf LaTeX template in
`docs/paper/soict_harlstmgat.tex`). Architecture diagram: `docs/paper/diagrams/soict_harlstmgat.svg`.*

---

## Abstract

We evaluate whether a deep sequence model or a cross-sectional graph-attention branch changes multi-horizon
daily Parkinson-variance forecasts relative to a linear Heterogeneous Autoregressive (HAR) baseline on the
Vietnamese equity market — VN30 and VN100 — under panel-correct inference. Three
models share the same five node features: a linear HAR, a
five-feature LSTM, and a five-feature LSTM+GAT (directed volume→Parkinson Top-5 weighted two-hop attention).
We report five metrics (MSE, RMSE, MAE, QLIKE, R²) with equal weight and assess significance with the
date-clustered Diebold–Mariano (DM) test; predictions are floored at 1e-2 times the per-node training mean.
The primary target is short-term (horizons h ∈ {1, 5}); h ∈ {10, 22} are an extended study. HAR has the
lowest QLIKE on both panels at every horizon, and no learned model has a significantly lower QLIKE than HAR
(date-clustered DM, p ≥ 0.05). At h1 and h5 the deep models have the lowest MAE on VN100 (LSTM versus HAR,
p<0.001), and the LSTM+GAT has a significantly lower QLIKE than the no-graph LSTM in both panels (VN100 h1
p<0.001, h5 p=0.022; VN30 h1 p<0.001, h5 p=0.031), reaching HAR's QLIKE level at VN100 h5 (0.5690 versus
0.5633). At h10 and h22 HAR has the lowest MSE, RMSE, QLIKE and R², and the LSTM+GAT-versus-LSTM QLIKE
difference is not significant (p ≥ 0.55). The ranking is metric- and horizon-dependent, so all five metrics
are reported with equal weight. Methodologically, a naive per-observation DM test on this cross-sectionally
dependent panel overstates significance by roughly the square root of the number of stocks (all stocks share the same trading days).
Every number is read from a stored results file.

---

## 1. Introduction

Daily volatility forecasting underpins risk management, position sizing and derivative pricing. The HAR
model of Corsi (2009) is a linear benchmark: a few ordinary-least-squares coefficients on daily, weekly and
monthly aggregates of realized volatility capture much of the long-memory structure of the series. Deep
sequence models and graph neural networks are frequently proposed to capture nonlinear temporal dynamics and
cross-sectional volatility spillovers, and the evidence that they out-of-sample beat HAR is mixed; positive
claims are often confounded by data-design choices, such as a common-date intersection that discards most of
the panel or a naive significance test on cross-sectionally dependent data.

This paper measures, under a single panel-correct evaluation on the Vietnamese equity market, whether the
temporal nonlinearity or the cross-sectional graph changes the forecast relative to a linear HAR. We ask two
questions:

1. Does a deep temporal LSTM over the shared feature set change the out-of-sample forecast relative to the
   linear HAR, and at which horizons and metrics?
2. Does a cross-sectional Graph Attention Network branch — here a directed volume→Parkinson weighted graph —
   change the out-of-sample forecast relative to the same-feature no-graph LSTM?

To answer these we use a **masked union-of-dates panel**: rather than intersecting to the dates on which every
ticker is present, we keep the union of trading dates, with node and target masks for tickers not yet listed
on a given date, so late-listed tickers are included without dropping whole dates and every model — including
the graph model — is trained and evaluated on the same panel with the same five features.

Our contributions are: (i) a multi-horizon benchmark on the masked union-of-dates VN30/VN100 panel that holds
the feature set and the data design fixed across HAR, an LSTM and an LSTM+GAT, so any difference is
attributable to a single component; (ii) per-horizon results on all five metrics with the date-clustered DM
test — HAR has the lowest QLIKE at every horizon with no learned model significantly lower, the deep models
have the lowest short-horizon MAE on VN100, and the LSTM+GAT has a significantly lower QLIKE than the no-graph
LSTM at h1 and h5 in both panels but not at h10 or h22; and (iii) a methodological contribution — the
date-clustered DM test, since a naive per-observation test overstates significance by a factor on the order
of the square root of the cross-section size. Every number reported here is taken from a stored results file.

---

## 2. Related Work

**HAR and classical volatility models.** The HAR model (Corsi, 2009) approximates the long memory of
realized volatility with a cascade of daily, weekly and monthly components, motivated by the heterogeneous
market hypothesis. Recent machine-learning work reports that rolling-window and specification choices, rather
than model class, often account for measured differences from HAR (Audrino and Chassot, 2025). Extensions
such as HARQ (Bollerslev, Patton and Quaedvlieg, 2016) exploit measurement error, and GARCH-family models
remain standard conditional-variance benchmarks. We use HAR as the linear baseline.

**Deep and graph models for volatility.** LSTMs (Hochreiter and Schmidhuber, 1997) are widely applied to
financial time series, and graph neural networks have been proposed to model cross-asset spillovers, for
example Graph Attention Networks (Veličković et al., 2018), multivariate realized-volatility GNNs (Chen and
Robert, 2022) and spillover-aware GNN-HAR hybrids (Zhang et al., 2025). Reported effects vary across studies
and data designs. Volatility spillover measurement itself has a long econometric tradition (Diebold and
Yilmaz, 2012).

**Graph construction.** The edge set is a design choice. Our graph uses a directed volume→Parkinson relation
(each node attends to its Top-5 volume-shock leaders, edge-weighted, over two hops), motivated by volume
leading volatility.

**Forecast evaluation.** Volatility is latent, so we use the QLIKE loss, which is robust to the volatility
proxy (Patton, 2011), alongside MSE-based losses. Statistical significance of forecast-accuracy differences
is assessed with the Diebold–Mariano test (Diebold and Mariano, 1995) using the small-sample correction of
Harvey, Leybourne and Newbold (1997).

---

## 3. Method

### 3.1 Target and features

The forecasting target is the Parkinson variance estimator (Parkinson, 1980) at day `t+h`, a non-negative
point forecast. The primary target is short-term: horizons `h ∈ {1, 5}` (one trading day and one trading week
ahead); horizons `h ∈ {10, 22}` (two weeks and roughly one trading month ahead) are reported as an extended
study. All three models use the same five node features per ticker at day `t`: the daily Parkinson variance,
its 5-day rolling mean (weekly), its 22-day rolling mean (monthly), a market Parkinson factor (the
cross-sectional average Parkinson variance) and a 20-day volume z-score.

### 3.2 Models

- **HAR** (baseline): pooled ordinary-least-squares regression on the five features, linear.
- **LSTM (5-feature)**: a two-layer LSTM (hidden size 64) over the lookback window of the five features. A
  single pooled model is trained over all tickers; each ticker is standardized with its own `StandardScaler`
  fit on training rows only; the output is linear and inverse-transformed to the physical scale for
  evaluation. Dropout, weight decay, gradient clipping, a `ReduceLROnPlateau` schedule and early stopping are
  used.
- **LSTM+GAT (5-feature)** (Figure 1): the LSTM temporal branch above, plus a Graph Attention Network
  spatial branch (Veličković et al., 2018) that reads the five node features at day `t` and attends over a
  directed volume→Parkinson Top-5, edge-weighted, two-hop graph estimated on training rows only. The two
  branch outputs are concatenated and passed to an MLP head. The leave-one-out comparison against the
  same-feature **LSTM** isolates the graph's marginal contribution.

![Figure 1: HAR-LSTM-GAT architecture](diagrams/soict_harlstmgat.png)

*Figure 1. Architecture of LSTM+GAT: a per-node LSTM temporal branch and a GAT spatial branch (directed
volume→Parkinson Top-5 weighted two-hop attention) are concatenated and passed to an MLP head; the
leave-one-out comparison removes the GAT branch to give the same-feature LSTM.*

### 3.3 Masked union-of-dates panel

A GAT operates over a common-date cross-section, which naively forces a common-date intersection design (keep
only dates on which every ticker is present) that discards most ticker-days and places the whole test set in
one recent period. We instead use a **masked union-of-dates panel**: the GAT operates over a common-date
cross-section with node and target masks for tickers not yet listed on a given date, so late-listed tickers
are included without dropping whole dates, and a **mask-aware loss** ignores absent tickers. Every model —
HAR, LSTM and LSTM+GAT — is trained and evaluated on the same panel with the same five features and the same
chronological split. The panel yields test sets of 46,308 masked observations over 454 test dates on VN100
(h1) and 10,106 observations over 326 test dates on VN30 (h1).

### 3.4 Training and evaluation protocol

Each configuration is run with five seeds, up to 20 epochs with early stopping, on GPU with parallel data
workers. We report five metrics — MSE, RMSE, MAE, QLIKE and R² — seed-averaged over the test set, with equal
weight. Predictions are floored at 1e-2 times the per-node training mean, applied identically to every model.

Statistical significance is assessed with the **date-clustered Diebold–Mariano test**: the per-observation
loss differential is collapsed to one value per calendar date before the DM statistic is computed, with the
Harvey–Leybourne–Newbold small-sample correction and a HAC lag of `h-1`. This is the panel-correct treatment
for a cross-section in which all tickers share each trading date; a naive per-observation DM over every
(ticker, date) row treats the effective sample as the number of tickers times the number of dates,
understates the loss-differential variance, and inflates the statistic by a factor on the order of the
square root of the cross-section size (roughly six on VN30 and ten on VN100). All significance statements use
the date-clustered test. We report two targeted contrasts — LSTM versus HAR and LSTM+GAT versus LSTM — on
QLIKE and MAE.

---

## 4. Data

Two equity panels are used, both with the Parkinson variance target:

| Panel | Tickers | Role | Source |
|---|---|---|---|
| VN30 | 31 | Primary (Vietnam) | Project processed data |
| VN100 | 102 | Vietnam breadth | vnstock |

Both panels use the panel of Section 3.3 with the five features, a single chronological
split and per-ticker standardization fit on training rows only.

---

## 5. Experiments

We compare HAR, the five-feature LSTM and the five-feature LSTM+GAT (directed volume→Parkinson weighted
two-hop) on VN30 and VN100 over the panel at horizons {1, 5, 10, 22}, reporting all
five metrics and the two targeted date-clustered DM contrasts (LSTM versus HAR, LSTM+GAT versus LSTM).

---

## 6. Results

Point-error metrics are reported in scaled units to avoid scientific notation: **MSE ×10⁻⁷, RMSE ×10⁻⁴, MAE
×10⁻⁴**. QLIKE and R² are unscaled. All values are the five-seed test-set means from the stored `result.json`
files (`results/masked_rich_floor1e2/`). Row order is HAR → LSTM → LSTM+GAT.

*Source: `docs/reports/2026-08-22_masked_rich_floor1e2_clean.md`.*

Tables 1 (VN100) and 2 (VN30) report all five metrics; Table 3 reports the two targeted date-clustered DM
contrasts on QLIKE and MAE. The primary horizons h1 and h5 appear first; the extended horizons h10 and h22
appear in the lower block of each table.

**Table 1. VN100 (46,308 masked obs over 454 test dates at h1; 5 seeds).** Lower is better for
MSE/RMSE/MAE/QLIKE, higher for R².

| h | Model | MSE (×10⁻⁷) | RMSE (×10⁻⁴) | MAE (×10⁻⁴) | QLIKE | R² |
|---|---|---:|---:|---:|---:|---:|
| 1 | HAR       | 2.367 | 4.865 | 2.898 | 0.5115 | 0.2236 |
| 1 | LSTM      | 2.370 | 4.869 | 2.821 | 0.5525 | 0.2224 |
| 1 | LSTM+GAT  | **2.362** | **4.860** | **2.819** | **0.5107** | **0.2251** |
| 5 | HAR       | **2.606** | **5.104** | 3.160 | **0.5633** | **0.1466** |
| 5 | LSTM      | 2.638 | 5.136 | **3.090** | 0.5841 | 0.1361 |
| 5 | LSTM+GAT  | 2.635 | 5.133 | 3.103 | 0.5690 | 0.1371 |
| 10 | HAR      | **2.754** | **5.248** | 3.306 | **0.6023** | **0.0978** |
| 10 | LSTM     | 2.798 | 5.289 | 3.282 | 0.6070 | 0.0837 |
| 10 | LSTM+GAT | 2.802 | 5.294 | **3.276** | 0.6072 | 0.0822 |
| 22 | HAR      | **2.891** | **5.377** | 3.486 | **0.6405** | **0.0544** |
| 22 | LSTM     | 2.979 | 5.458 | **3.468** | 0.6518 | 0.0257 |
| 22 | LSTM+GAT | 3.003 | 5.480 | 3.563 | 0.6544 | 0.0178 |

**Table 2. VN30 (10,106 masked obs over 326 test dates at h1; 5 seeds).** Same scaling as Table 1.

| h | Model | MSE (×10⁻⁷) | RMSE (×10⁻⁴) | MAE (×10⁻⁴) | QLIKE | R² |
|---|---|---:|---:|---:|---:|---:|
| 1 | HAR       | 1.927 | 4.389 | 2.389 | **0.5159** | 0.2308 |
| 1 | LSTM      | 1.929 | 4.393 | 2.407 | 0.6073 | 0.2297 |
| 1 | LSTM+GAT  | **1.912** | **4.372** | **2.366** | 0.5800 | **0.2368** |
| 5 | HAR       | **2.139** | **4.625** | **2.583** | **0.5965** | **0.1497** |
| 5 | LSTM      | 2.164 | 4.652 | 2.632 | 0.6402 | 0.1397 |
| 5 | LSTM+GAT  | 2.147 | 4.633 | 2.651 | 0.6059 | 0.1467 |
| 10 | HAR      | **2.301** | **4.797** | 2.733 | **0.6428** | **0.1028** |
| 10 | LSTM     | 2.310 | 4.806 | **2.721** | 0.6584 | 0.0995 |
| 10 | LSTM+GAT | 2.326 | 4.823 | 2.725 | 0.6564 | 0.0931 |
| 22 | HAR      | **2.272** | **4.766** | **2.785** | **0.6422** | **0.0723** |
| 22 | LSTM     | 2.383 | 4.881 | 2.904 | 0.6550 | 0.0271 |
| 22 | LSTM+GAT | 2.385 | 4.883 | 2.884 | 0.6548 | 0.0261 |

**Table 3. Targeted date-clustered DM contrasts (p-value; favoured model in parentheses; bold if p<0.05).**
H = HAR, L = LSTM, G = LSTM+GAT, on QLIKE and MAE.

| Panel | h | LSTM vs HAR (QLIKE) | (MAE) | LSTM+GAT vs LSTM (QLIKE) | (MAE) |
|---|---|---|---|---|---|
| VN100 | 1  | **0.028 (H)** | **<0.001 (L)** | **<0.001 (G)** | 0.684 (G) |
| VN100 | 5  | 0.148 (H) | **<0.001 (L)** | **0.022 (G)** | **0.002 (L)** |
| VN100 | 10 | 0.598 (H) | 0.201 (L) | 0.872 (L) | 0.064 (G) |
| VN100 | 22 | 0.429 (H) | 0.611 (L) | 0.658 (L) | **<0.001 (L)** |
| VN30  | 1  | **<0.001 (H)** | 0.221 (H) | **<0.001 (G)** | **<0.001 (G)** |
| VN30  | 5  | **0.037 (H)** | **<0.001 (H)** | **0.031 (G)** | **0.015 (L)** |
| VN30  | 10 | 0.260 (H) | 0.340 (L) | 0.547 (G) | 0.624 (L) |
| VN30  | 22 | 0.318 (H) | **0.001 (H)** | 0.936 (G) | **0.021 (G)** |

**Short-horizon results (h1, h5).**

- *VN100 h1.* The LSTM+GAT has the lowest value on all five metrics (MSE 2.362, RMSE 4.860, MAE 2.819, QLIKE
  0.5107, R² 0.2251). By date-clustered DM, the LSTM has a lower MAE than HAR (p<0.001), the LSTM has a higher
  QLIKE than HAR (p=0.028), and the LSTM+GAT has a lower QLIKE than the no-graph LSTM (p<0.001).
- *VN100 h5.* HAR has the lowest MSE, RMSE, QLIKE and highest R²; the LSTM has the lowest MAE (3.090). By DM,
  the LSTM has a lower MAE than HAR (p<0.001) and the LSTM+GAT has a lower QLIKE than the LSTM (p=0.022).
- *VN30 h1.* The LSTM+GAT has the lowest MSE, RMSE, MAE and highest R²; HAR has the lowest QLIKE (0.5159). By
  DM, the LSTM has a higher QLIKE than HAR (p<0.001), and the LSTM+GAT has a lower QLIKE (p<0.001) and lower
  squared error (p=0.003) than the no-graph LSTM.
- *VN30 h5.* HAR has the lowest value on all five metrics. By DM, the LSTM has a higher QLIKE (p=0.037) and
  higher MAE (p<0.001) than HAR, and the LSTM+GAT has a lower QLIKE than the LSTM (p=0.031).

Across the four short-horizon cells, no learned model has a significantly lower QLIKE than HAR; the LSTM+GAT
has a significantly lower QLIKE than the no-graph LSTM in all four (VN100 h1 p<0.001, h5 p=0.022; VN30 h1
p<0.001, h5 p=0.031), and its QLIKE equals HAR's level at VN100 h5 (0.5690 versus 0.5633) and VN100 h1 (0.5107
versus 0.5115).

**Extended-horizon results (h10, h22).** HAR has the lowest MSE, RMSE, QLIKE and highest R² on both panels at
both horizons; the LSTM has the lowest MAE at VN100 h22 (3.468) and VN30 h10 (2.721), and the LSTM+GAT the
lowest MAE at VN100 h10 (3.276). By date-clustered DM, the LSTM+GAT-versus-LSTM QLIKE difference is not
significant at h10 or h22 on either panel (p ≥ 0.55); the LSTM has a higher squared error than HAR at VN100
h10 (p=0.022) and h22 (p=0.002) and a higher MAE than HAR at VN30 h22 (p=0.001).

---

## 7. Discussion

On both panels HAR has the lowest QLIKE at every horizon, and no learned model has a significantly lower QLIKE
than HAR under the date-clustered DM test. At h1 and h5 the LSTM and LSTM+GAT have the lowest MAE on VN100
(LSTM versus HAR, p<0.001), and the LSTM+GAT has a significantly lower QLIKE and, on VN30, lower squared error
than the same-feature no-graph LSTM; at VN100 h1 and VN30 h1 the LSTM+GAT has the lowest value on four or five
of the metrics. At h10 and h22 HAR has the lowest MSE, RMSE, QLIKE and R², and the LSTM+GAT-versus-LSTM QLIKE
difference is not significant. The graph's effect on QLIKE relative to the no-graph LSTM is therefore present
at the short horizons and absent at the longer horizons; the deep models' MAE reduction is present at the
short horizons on VN100 and reverses at the longer horizons, where their squared error and MAE are higher than
HAR's.

The ranking is metric- and horizon-dependent, which is why all five metrics are reported with equal weight.
The date-clustered DM test matters for these conclusions: a naive per-observation test on this
cross-sectionally dependent panel inflates the statistic by a factor on the order of the square root of the
cross-section size, so a difference that is not significant per date can appear significant per observation.

---

## 8. Limitations

- **Positivity floor.** QLIKE at the short horizons depends on the per-node prediction floor; we fix a common
  floor (1e-2 times the per-node training mean) across all models and report MSE, RMSE, MAE and R² alongside
  QLIKE.
- **Single market.** The study covers the Vietnamese market (VN30, VN100).
- **Graph scope.** The study uses a directed volume→Parkinson weighted graph; other graph designs (for
  example volume-shock-threshold or sector graphs) were not tested and remain follow-up work.
- All reported numbers are read directly from stored `result.json` files.

---

## 9. Conclusion

For daily Parkinson-variance forecasting on the Vietnamese market, evaluated with the same features, the same
masked union-of-dates panel and the same date-clustered Diebold–Mariano inference, HAR has the lowest QLIKE at
every horizon on both panels and no learned model has a significantly lower QLIKE than HAR. At the short
horizons (h1, h5) the LSTM and LSTM+GAT have the lowest MAE on VN100, and the directed volume→Parkinson
weighted graph gives the LSTM+GAT a significantly lower QLIKE than the same-feature no-graph LSTM in both
panels, equal to HAR's QLIKE level at VN100 h5; at the extended horizons (h10, h22) HAR has the lowest MSE,
RMSE, QLIKE and R², and the graph's QLIKE difference from the no-graph LSTM is not significant. The ranking is
metric- and horizon-dependent, so all five metrics are reported with equal weight.

---

## References

1. Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial
   Econometrics*, 7(2), 174–196.
2. Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return.
   *The Journal of Business*, 53(1), 61–65.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
4. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention
   networks. In *International Conference on Learning Representations (ICLR)*.
5. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic
   Statistics*, 13(3), 253–263.
6. Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors.
   *International Journal of Forecasting*, 13(2), 281–291.
7. Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of
   Econometrics*, 160(1), 246–256.
8. Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting the errors: a simple approach for
   improved volatility forecasting (HARQ). *Journal of Econometrics*, 192(1), 1–18.
9. Diebold, F. X., & Yilmaz, K. (2012). Better to give than to receive: predictive directional measurement
   of volatility spillovers. *International Journal of Forecasting*, 28(1), 57–66.
10. Chen, Q., & Robert, C.-Y. (2022). Multivariate realized volatility forecasting with graph neural
    network. In *ACM ICAIF*. arXiv:2112.09015.
11. Zhang, C., Pu, X., Cucuringu, M., & Dong, X. (2025). Forecasting realized volatility with spillover
    effects: perspectives from graph neural networks (GNNHAR). arXiv:2308.01419.
12. Audrino, F., & Chassot, J. (2025). HARd to beat: the overlooked impact of rolling windows in the era of
    machine learning. *International Journal of Forecasting*. arXiv:2406.08041.

---

*Figure 1 appears in Section 3.2. Vector source: `docs/paper/diagrams/soict_harlstmgat.svg`; PDF/PNG
rendered by `diagrams/generate_arch.py`.*
