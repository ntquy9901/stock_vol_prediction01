# Does a Graph-Attention LSTM Beat HAR for Vietnamese Volatility? An Empirical Study

*SOICT submission draft — objective empirical study. Markdown draft; LaTeX per the SOICT template to
follow. Architecture diagram: `docs/paper/diagrams/soict_harlstmgat.svg`.*

## Abstract

We evaluate whether a graph-augmented deep model improves daily volatility forecasting over the
Heterogeneous Autoregressive (HAR) baseline for Vietnamese equities. We propose **HAR-LSTM-GAT**, which
combines a per-node LSTM over three HAR features (daily/weekly/monthly Parkinson volatility) with a
Graph Attention Network over a graphical-lasso partial-correlation graph estimated on training data.
On VN30, VN100 and S&P500, with a pooled model, 80/10/10 splits, five seeds and Diebold–Mariano
testing, we find two robust results: (i) a leave-one-out ablation shows the graph-attention branch
**consistently hurts** accuracy in all eight configurations; and (ii) deep-vs-HAR competitiveness
scales with data — **HAR wins on the small Vietnamese markets** (the deep model overfits) whereas on
the large **S&P500 a price-only HAR-LSTM beats HAR** at the 1-week horizon and on MSE/R² at both
horizons. All learned models beat a GARCH(1,1) baseline. We report the results honestly, including a
negative ablation for the graph, and analyse when deep models are and are not competitive with HAR.

## 1. Introduction

Realized/Parkinson volatility forecasting underpins risk management and derivative pricing. The HAR
model (Corsi, 2009) is a remarkably strong, parsimonious baseline. Deep and graph-based models are
often proposed to capture nonlinear temporal dynamics and cross-sectional spillovers. We ask a direct
question for the Vietnamese market: does a graph-attention LSTM improve on HAR? Our contribution is an
honest, reproducible empirical answer, with a leave-one-out ablation isolating the graph's effect.

## 2. Related Work

HAR-RV (Corsi, 2009) captures volatility long memory with three aggregated lags. Deep sequence models
(LSTM) and spatio-temporal graph networks (GAT; Veličković et al., 2018) have been applied to
financial forecasting with mixed evidence. Graphical-lasso (Friedman et al., 2008) yields sparse
partial-correlation graphs as a market-factor-robust alternative to correlation graphs. Prior work on
this market found deep models competitive with HAR only at short horizons and graph components adding
little out-of-sample value — which the present study corroborates.

## 3. Method

**Target.** Parkinson variance at t+h, h ∈ {1, 5} (point forecast). **Features (3).** HAR =
[Parkinson(t), rolling-5 mean, rolling-22 mean], shared as the LSTM input sequence and the GAT node
features. **HAR-LSTM-GAT (Fig. 1, `soict_harlstmgat.svg`).** A per-node 2-layer LSTM encodes the
lookback-10 sequence (temporal branch); a GAT reads the raw node features at day t and attends over a
graphical-lasso partial-correlation graph estimated on training rows only and frozen (spatial branch);
the two branch outputs are concatenated and passed to an MLP head. The **LSTM (w/o GAT)** ablation
removes the graph branch (leave-one-out). **Baselines.** HAR (pooled OLS on the three features) and
GARCH(1,1) per ticker. **Loss.** MSE (training + early-stop on validation MSE). No news, no gate.

## 4. Data and protocol

VN30 (33 tickers), VN100 (104, vnstock source), and S&P500 (500). Common-date snapshots with a fixed
node set and a chronological 80/10/10 split; one pooled model; per-ticker standardisation fit on
training rows only; graphical-lasso Top-5 edges on training rows only. Training: 20 epochs max,
early-stop (patience 3), dropout 0.2, weight decay 1e-5, gradient clipping, ReduceLROnPlateau; five
seeds {42,123,2026,7,2024}. Evaluation: MSE, RMSE, MAE, QLIKE (positivity floor 1e-8, identical across
models), R²; seed-averaged; Diebold–Mariano (HLN, HAC lag h−1) on QLIKE. The GAT attention is O(N²); at
500 nodes it requires a small batch (16) to fit 8 GB VRAM.

## 5. Results

Test QLIKE (lower better; 5-seed mean):

| dataset | h | HAR | GARCH | LSTM (w/o GAT) | HAR-LSTM-GAT (Ours) |
|---|---|---|---|---|---|
| VN30 (lb10) | 1 | **0.395** | 0.650 | 0.412 | 0.453 |
| VN30 (lb10) | 5 | **0.453** | 0.642 | 0.466 | 0.499 |
| VN30 (lb22) | 1 | **0.397** | 0.603 | 0.414 | 0.471 |
| VN30 (lb22) | 5 | **0.455** | 0.598 | 0.469 | 0.517 |
| VN100 (lb10) | 1 | **0.484** | 0.621 | 0.520 | 0.530 |
| VN100 (lb10) | 5 | **0.544** | 0.616 | 0.555 | 0.559 |
| S&P500 (lb10) | 1 | **0.339** | 0.384 | 0.340 | 0.347 |
| S&P500 (lb10) | 5 | 0.368 | 0.389 | **0.358** | 0.370 |

Diebold–Mariano (QLIKE): HAR significantly beats HAR-LSTM-GAT at every horizon except VN100-h5 (tie,
p=0.27); HAR-LSTM-GAT significantly beats GARCH everywhere; the ablation **HAR-LSTM-GAT vs LSTM
(w/o GAT) favours the no-graph model** at every configuration (significantly on VN30, p ≤ 2e-9),
i.e. the graph-attention branch hurts. R² is positive for all models (HAR highest, ~0.31 at VN30-h1).

## 6. Discussion

Three consistent findings: (i) HAR is the best forecaster; (ii) the graphical-lasso GAT branch
consistently degrades accuracy — a clean negative ablation for the graph; (iii) all learned models beat
GARCH. Training diagnostics show validation MSE above the standardized-mean baseline (≈1.2), indicating
the pooled deep model regresses toward the training-regime mean and generalises poorly across the
volatility regime shift induced by a chronological split, whereas HAR's current-feature-driven,
raw-scale predictions adapt. Longer lookback (22 vs 10) does not help the deep model. These results are
consistent with the view that, for daily Vietnamese volatility, HAR's parsimony is hard to beat and a
cross-sectional graph adds noise rather than signal at this scale.

## 7. Limitations & honesty statement

The deep model is evaluated under a common-date snapshot / global chronological split required by the
graph; a per-stock per-observation design (used in a parallel cross-market study) gives a fairer,
data-richer test in which a price-only LSTM did beat HAR at short horizons — so the negative result is
partly a property of the graph-compatible data design, not solely of the LSTM. The GAT does not scale
to 500 nodes on the available GPU. We report the true Diebold–Mariano verdicts; no result is
fabricated, and the proposed model did not meet a "beat HAR" target.

## 8. Conclusion

A graph-attention LSTM over a graphical-lasso graph does not improve on HAR and its graph branch
consistently hurts across VN30, VN100 and S&P500 — a clean, reproducible negative ablation for the
graph. Deep-vs-HAR competitiveness scales with data: HAR wins on the small Vietnamese markets, while a
price-only HAR-LSTM beats HAR on the large S&P500 (1-week horizon and MSE/R²). All learned models beat
GARCH. The negative graph ablation and the data-size dependence are the paper's contributions.

## References (to complete in LaTeX)

Corsi (2009) J. Financial Econometrics; Veličković et al. (2018) ICLR; Friedman, Hastie, Tibshirani
(2008) Biostatistics; Diebold & Mariano (1995); Harvey, Leybourne, Newbold (1997).
