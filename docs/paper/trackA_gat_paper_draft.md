# Stock Volatility Prediction for the Vietnamese Stock Market

---

## Abstract

Volatility forecasts set risk limits, margin, and option prices across the Vietnamese stock market. This
work targets volatility forecasting for Vietnamese equities in general and uses the VN30 constituents
(the most liquid stocks on Vietnam's Ho Chi Minh Stock Exchange) as the empirical case study. Two
structural signals a univariate Heterogeneous AutoRegressive (HAR) model discards are Vietnamese-language
news and cross-stock spillover. We propose a parallel
multi-branch model that fuses three branches for one ticker's multi-horizon Parkinson-variance forecast: a
per-node price LSTM over five node features, a real multi-head Graph Attention Network (GAT) over a
**directed volume-to-volatility (vol→PK) lead-lag edge**, and a PhoBERT news LSTM with a per-ticker
gate. The GAT branch consumes the raw node-feature vector at the forecast origin rather than the LSTM
hidden state, which matches the edge semantics (a source ticker's volume shock at $t$ leading a target
ticker's volatility at $t{+}1$) and keeps the graph branch an independent cross-sectional view for a
fair ablation. We evaluate the model with an **ablation**: from the full model we retrain
one variant per removed component (minus-graph removes the entire GAT branch, minus-gate removes the
per-ticker gate, minus-news removes the news branch), and we measure each component's contribution as
$\text{effect}(X)=\text{QLIKE}(\text{FULL})-\text{QLIKE}(\text{FULL}{-}X)$ on the held-out test set,
with a Diebold-Mariano test for significance. The classical HAR is the reference baseline. On a single-seed run across four horizons, the differences
among HAR, the full model, and the ablation variants are small on every metric. On MSE, RMSE, and $R^2$
the configurations lie within about 1% of one another and the best row alternates across horizons; on MAE
the differences are similarly small and the lowest value varies by horizon. On QLIKE, HAR has the lowest
value at h10 and h22, the full model is marginally lower at h1, and the two are comparable at h5.
Diebold-Mariano tests (single seed) on the QLIKE, squared-error, and absolute-error loss families show no
model favored across all three: relative to HAR the full model has significantly lower QLIKE at h1 and
significantly higher QLIKE at h10 and h22, significantly lower squared-error loss at h5, and significantly
lower absolute-error loss at h5 and h10, with the other cells not significant. The ablation removals are
significant on some loss families and horizons and not others. Across all five metrics no
configuration consistently outperforms HAR, and most differences are not statistically distinguishable.
This paper's contribution
is a leakage-safe ablation-based
attribution of a directed-spillover graph-attention news model against a strong HAR baseline on sparse
daily VN30 data.

**Keywords:** volatility forecasting; graph attention networks; directed spillover; financial news;
PhoBERT; emerging markets.

---

## 1. Introduction

Volatility forecasts drive the daily risk decisions of every desk that trades Vietnamese equities. This
work addresses stock-volatility forecasting for the Vietnamese market in general; the VN30 basket (the
roughly thirty most liquid stocks on Vietnam's Ho Chi Minh Stock Exchange) is the empirical case study
used for all experiments because it has the longest and cleanest daily histories. A margin engine sizes
collateral from forecast volatility, an option desk prices from it, and a risk officer sets position
limits against it. VN30 desks make these decisions inside an information environment that a price-only
forecaster discards: a steady flow of Vietnamese-language news that plausibly signals volatility shocks
before they reach the price range, and a web of cross-stock spillovers that a univariate model cannot
represent.

The classical HAR model of Corsi [1], the linear univariate workhorse that regresses future volatility
on its own daily, weekly, and monthly averages, admits neither signal, and it is hard to beat [14]. Two
structural extensions could close the gap: a news channel that reads text, and a cross-stock graph that
lets one stock's information reach another. Prior work on this project established, through a
leakage-safe exploratory analysis, that adding cross-sectional node features (a market factor and a
volume z-score) beats HAR on QLIKE, but that a *correlation-based* cross-stock edge adds no reliable
out-of-sample value because a single market factor dominates cross-stock volatility co-movement. This
motivates the present study's central design choice: rather than a symmetric correlation edge, we test a
**directed, lead-lag volume→volatility edge** — a source ticker's abnormal volume today linked to a
target ticker's range volatility tomorrow — which encodes a predictive, causal-direction relationship
that a contemporaneous correlation edge cannot.

We build this edge into a parallel multi-branch architecture (an LSTM temporal branch, a GAT graph
branch, and a gated news branch, concatenated at a shared head) and we quantify, per component and per
horizon, the marginal contribution of each part with an **ablation**: we build the
full model, then retrain a variant with exactly one component removed, and attribute each component's
contribution as the change in held-out QLIKE it causes. This paper makes three contributions.

1. **A directed vol→PK graph-attention news model on a parallel multi-branch backbone.** The model fuses a
   per-node price LSTM, a real multi-head GAT over a directed volume-to-volatility lead-lag edge, and a
   gated PhoBERT news branch. The GAT consumes the raw node-feature vector at the forecast origin, which
   matches the edge semantics and keeps the graph branch an independent cross-sectional view (Section 4).

2. **An ablation against a strong HAR baseline.** From the full model we retrain one
   variant per removed component — minus-graph (the whole GAT branch), minus-gate, minus-news — so every
   effect is measured on the same footing, and we report each component's marginal contribution
   $\text{effect}(X)=\text{QLIKE}(\text{FULL})-\text{QLIKE}(\text{FULL}{-}X)$ with a Diebold-Mariano test
   (Section 6).

3. **A leakage-safe multi-horizon evaluation.** All models are evaluated at horizons 1, 5, 10, and 22
   trading days on a chronological split with train-only scalers and a train-only frozen edge, with the
   held-out test set as the reported result (Sections 5 and 6).

---

## 2. Related Work

We group prior work into four families and state where the proposed model sits relative to each.

**Econometric volatility models.** The HAR model [1] and its range-based inputs [2] set the standard for
daily volatility forecasting and remain hard to beat. HAR regresses future volatility on daily, weekly,
and monthly moving averages, approximating volatility's long memory with a parsimonious linear fit. The
HARQ extension [15] shrinks the daily coefficient when the volatility estimate is noisy. A large-scale
controlled study finds that tuned machine-learning models fail to beat a carefully re-estimated HAR on
QLIKE and MSE when both use the same information set [14]. These models are univariate and linear: no
channel admits text, and no channel couples one stock to another. We keep the three HAR scales among the
node features and add the news and directed cross-stock channels HAR omits; classical HAR is the baseline
every component must beat.

**Deep and graph-based forecasters.** LSTM forecasters [4] capture nonlinear temporal structure, and
graph attention networks [5] model cross-asset coupling; hybrid LSTM-GNN designs combine both [9]. Chen
and Robert [19] forecast multivariate realized volatility with a graph over about 500 S&P names. Most
relevant, Zhang, Pu, Cucuringu and Dong [10] build a graph-neural-network HAR (GNNHAR) on DJIA-30 and,
under a Model Confidence Set, find that multi-hop cross-stock graph spillover gives no clear advantage:
the gains come from modeling nonlinearity and from switching the training loss from MSE to QLIKE. Their
controlled null on the graph component motivates our shift from a symmetric correlation edge to a
directed lead-lag edge, tested here under an ablation.

**Directed spillover and lead-lag graphs.** A symmetric correlation edge conflates contemporaneous
co-movement (largely a market factor) with predictive structure. Directed spillover measures — for
example volatility spillover indices [16] — encode who-leads-whom and are asymmetric by construction. We
instantiate a directed edge from a source ticker's abnormal volume to a target ticker's next-day range
volatility, motivated by the volume–volatility lead-lag relation, and freeze it on the training window.

**News- and text-augmented forecasting.** Text-augmented forecasters add sentiment scores or news
embeddings [7,8], usually fusing a single market-wide signal by concatenation on English- or
Chinese-language markets. We work in Vietnamese with PhoBERT [3] on VN30 and add a per-ticker gate that
admits a different amount of news per stock.

---

## 3. Data

**Universe (case study).** The method targets the Vietnamese stock market in general; VN30 is the
case-study universe for the experiments. We use 33 VN30 constituents with daily
open-high-low-close-volume (OHLCV) data. Series
lengths range from about 1,300 sessions (SSB, listed 2021) to about 4,900 (VNM, ACB, from 2006). The
universe is a fixed, point-in-time VN30-like set rather than the live index, a
limitation stated in Section 8.

**Forecast target.** We forecast the daily Parkinson range volatility at horizons 1, 5, 10, and 22
trading days ahead: the target is the single-day estimator value on day $t{+}h$ (i.e.
$\text{PK}(t{+}h)$), not an average over the next $h$ days. The Parkinson estimator uses the intraday high $H$
and low $L$:

$$\sigma^2_{\text{Park}} = \frac{(\ln(H/L))^2}{4\ln 2}.$$

The processed `parkinson_volatility` column is numerically the Parkinson **variance** estimator
($\sigma^2$, non-negative), and every model in this paper forecasts this same daily realized-variance
quantity.

**Node features.** Each ticker-day carries five node features in a fixed order, all with unified
22-trading-day windows to match the monthly HAR scale:

1. `pk_daily` — the Parkinson variance at $t$, clipped to $\pm 3\sigma$ using train statistics.
2. `har_weekly` — trailing 5-day mean of `pk_daily`.
3. `har_monthly` — trailing 22-day mean of `pk_daily`.
4. `market_pk` — cross-sectional median of $\sqrt{\text{PK}}$ across present tickers at $t$ (a
   contemporaneous market factor; uses column $t$ only).
5. `volume_zscore` — trailing rolling-22 z-score of $\log(1+\text{volume})$; tickers with no OHLCV
   volume series receive a neutral 0.0.

**News panel.** Each Vietnamese article (title and lead) passes once through PhoBERT [3], yielding a
768-dimensional embedding reduced by PCA (fit before the training cutoff) to a per-day per-stock vector
of 146 dimensions (embedding components, per-group norms, exponentially weighted averages with a
30-trading-day half-life, and topic counts). Missing news on a day is a zero vector with a mask bit set
to zero, and an all-missing window still produces a finite representation.

**Directed vol→PK edge.** The adjacency is directed: for each target ticker $j$ we connect the Top-5
source tickers $i$ ranked by the train-only lead-lag correlation
$\text{corr}(\text{volume\_shock}_i(t),\ \sqrt{\text{PK}_j}(t{+}1))$. Edges are estimated on training
rows only and frozen for validation and test (leakage-safe), and self-loops are kept on the diagonal.

**Temporal split and leakage control.** Each ticker's series is split chronologically into 70% train,
15% validation, and 15% test before generating features, fitting scalers, or building windows; the
earliest series begin in 2006. The split is applied per ticker on its own timeline, so the
train/validation/test calendar boundaries differ across tickers — a ticker listed in 2021 has later
boundaries than one trading since 2006 — which is precisely why a fixed-node synchronized panel would
collapse and why the pooled per-ticker split is used. Per-ticker
price and target scalers are fit on the training partition only and selected at evaluation by explicit
`ticker_id`. A news feature for a sample uses only information available by that sample's forecast
origin. Evaluation reads stored raw targets rather than inverse-transforming a clipped normalized target.
On the pooled masked manifest the evaluation sets hold, at the five-day horizon, 14,418 validation and
14,464 present-node test observations, shared identically across every rung; the counts vary slightly by
horizon (e.g. 14,550/14,596 at h1 and 14,253/14,299 at h10) because the target shift changes the number
of eligible windows.

---

## 4. Method

We describe the proposed full model first, then define the ablation variants as component removals
from it. The model forecasts one ticker's $h$-day-ahead Parkinson variance from four inputs: a 22-day
price window of the five node features, a 22-day news window of PhoBERT features with a mask, the ticker
identity, and the directed vol→PK adjacency over the tickers present on the same date.

![Model architecture: parallel price-LSTM, directed vol→PK GAT (on raw node features), and gated PhoBERT news branches, concatenated into a head with a softplus positivity floor.](diagrams/trackA_gat_architecture.svg)

**The proposed full model.** Three branches run in parallel and concatenate at a shared head.

- **(i) Price-LSTM branch (temporal, per node).** A two-layer LSTM (hidden size 64, dropout 0.2) reads
  each ticker's 22-day window of the five node features into a price representation $h_{\text{lstm}}$
  ($\mathbb{R}^{64}$ per node).

- **(ii) GAT graph branch (cross-sectional).** The node input is the **raw** node-feature vector at the
  last timestep, $\text{node\_raw}=\text{price}[:, :, {-}1, :]\in\mathbb{R}^{5}$, not the LSTM hidden
  state. Two self-written multi-head GAT layers (5→256, then 256→256; 4 heads) with attention masked by
  the adjacency ($\text{softmax}$ over source nodes, ELU output) produce $h_{\text{gnn}}\in\mathbb{R}^{256}$.
  Consuming the raw features matches the edge semantics (`volume_shock` is directly available to the
  attention) and keeps the graph branch independent of the LSTM branch, so removing it is a clean
  ablation. When the graph is switched off, the adjacency is the identity (self-loops only).

- **(iii) News branch with per-ticker gate.** The 146-dimensional daily news vector passes through a
  linear-plus-ReLU projection into a two-layer LSTM (hidden size 64) over the masked window, producing a
  news representation, scaled by a learned per-ticker sigmoid gate
  $\text{news}^{\text{gated}}_i = \sigma(g_i)\cdot\text{news}_i$.

The head concatenates $[h_{\text{lstm}}(64),\ h_{\text{gnn}}(256),\ \text{news}^{\text{gated}}(64)]$
($\mathbb{R}^{384}$) and maps it through a linear-ReLU-dropout-linear stack to the forecast. A softplus
**positivity floor** ($\varepsilon=10^{-6}$, applied in the denormalized space and identical across all
compared rungs) clamps predictions away from non-positive variance before QLIKE.

**The ablation variants.** We build the full model, then retrain one variant per removed component,
so every effect is measured on the same footing (each variant trains in the same graph on/off regime it
is evaluated in — no train/eval mismatch):

![Ablation: from FULL, one component is removed per variant (−graph, −gate, −news); HAR and a price-only LSTM-only model are reference baselines. effect(X) = QLIKE(FULL) − QLIKE(FULL−X).](diagrams/trackA_gat_ablation.svg)


- **FULL** — LSTM + directed vol→PK GAT graph + news + per-ticker gate.
- **minus_graph** — FULL with the **entire GAT branch removed** (no node/edge/GAT built; the head takes
  $[h_{\text{lstm}},\ \text{news}^{\text{gated}}]\in\mathbb{R}^{128}$). This isolates the whole graph
  subsystem, not merely the edges.
- **minus_gate** — FULL without the per-ticker gate.
- **minus_news** — FULL without the news branch (the gate is a no-op without news).
- **HAR** — a pooled per-ticker HAR linear regression on the three HAR scales; the floor every
  component must beat.

The contribution of component $X$ is $\text{effect}(X)=\text{QLIKE}(\text{FULL})-\text{QLIKE}(\text{FULL}{-}X)$
on the held-out test set; a negative value means removing $X$ raised QLIKE, i.e. $X$ helped.

**Training.** The deep models use Adam (learning rate $10^{-3}$), weight decay $10^{-5}$, dropout 0.2,
and gradient clipping at 1.0, with best-validation-checkpoint selection. Because pooled models converge
within a few epochs and then overfit, training uses early stopping (patience 3, minimum 6 epochs) under
a 12-epoch cap. The initial run reports a single seed (42); if the single-seed result is promising, the
study extends to three seeds (42, 123, 2026) with seed-ensembled Diebold-Mariano tests.
{{NEEDS CLARIFICATION: final seed count for the reported tables — 1 or 3}}.

---

## 5. Experimental Setup

**Metrics.** Every configuration reports five metrics on the raw variance scale: MSE, RMSE, MAE, $R^2$,
and QLIKE. QLIKE, the quasi-likelihood loss standard in the realized-volatility literature [patton],
penalizes under-prediction more than over-prediction and tolerates the noise in the volatility proxy:

$$\text{QLIKE} = \frac{1}{T}\sum_{t=1}^{T}\left(\frac{\hat{\sigma}^2_t}{\sigma^2_t} - \ln\frac{\hat{\sigma}^2_t}{\sigma^2_t} - 1\right).$$

Directional accuracy is not reported; the Limitations section notes why it is uninformative for this target.

**Training objective.** All deep configurations minimize the mean squared error between the model output
and the per-ticker normalized target; QLIKE and the other metrics are computed only at evaluation, after
inverting the normalization to the raw variance scale, so the proportional QLIKE loss never enters the
gradient.

**Reporting protocol.** Model selection (early-stopping, best-checkpoint) uses the validation split only;
all reported results and significance tests are computed on the held-out test set; validation is used
only for model selection, and no validation metrics are reported.

**Significance.** We complement metric comparisons with a Diebold-Mariano (DM) test on the
per-observation QLIKE (and squared-error) loss series (HAC truncation lag $h{-}1$,
Harvey-Leybourne-Newbold corrected), which tests forecast-accuracy equality directly on the held-out
observations. Where multiple seeds are used, DM runs on the seed-ensembled predictions.

**Implementation and compute.** All models are implemented in PyTorch (self-written GAT layer; no
external graph library) and use a CUDA GPU. Runs used an NVIDIA GeForce RTX 4060 Laptop GPU under PyTorch
2.6 with CUDA 12.4.

---

## 6. Results

### 6.1 Held-out test metrics across horizons

**Table 1. Held-out TEST metrics by horizon (single seed 42).** Lower is better for MSE, RMSE, MAE,
QLIKE; higher for $R^2$. Bold marks the best value per column within each horizon; the same test
observations are shared across all rows of a horizon.

*h = 1 trading day*

| Config | MSE (×10⁻⁶) ↓ | RMSE (×10⁻³) ↓ | MAE (×10⁻⁴) ↓ | $R^2$ ↑ | QLIKE ↓ |
|---|---|---|---|---|---|
| HAR | 4.05 | 2.014 | 5.41 | 0.8192 | 0.4813 |
| FULL | 3.99 | 1.998 | 5.39 | 0.8221 | 0.4780 |
| minus_graph | 3.99 | 1.998 | 5.35 | 0.8220 | 0.4741 |
| minus_gate | **3.94** | **1.984** | **5.33** | **0.8245** | 0.4731 |
| minus_news | 3.95 | 1.987 | 5.35 | 0.8239 | **0.4724** |
| LSTM-only | 3.96 | 1.991 | 5.36 | 0.8232 | 0.4729 |

*h = 5 trading days*

| Config | MSE (×10⁻⁶) ↓ | RMSE (×10⁻³) ↓ | MAE (×10⁻⁴) ↓ | $R^2$ ↑ | QLIKE ↓ |
|---|---|---|---|---|---|
| HAR | 5.23 | 2.287 | 6.05 | 0.7672 | 0.5735 |
| FULL | **5.09** | **2.257** | 5.99 | **0.7733** | 0.5724 |
| minus_graph | 5.13 | 2.264 | 6.02 | 0.7719 | **0.5692** |
| minus_gate | 5.13 | 2.265 | 5.99 | 0.7718 | 0.5741 |
| minus_news | 5.12 | 2.264 | **5.98** | 0.7720 | 0.5715 |
| LSTM-only | 5.11 | 2.261 | 6.02 | 0.7725 | 0.5696 |

*h = 10 trading days*

| Config | MSE (×10⁻⁶) ↓ | RMSE (×10⁻³) ↓ | MAE (×10⁻⁴) ↓ | $R^2$ ↑ | QLIKE ↓ |
|---|---|---|---|---|---|
| HAR | 5.56 | 2.358 | 6.30 | 0.7532 | **0.6139** |
| FULL | 5.73 | 2.393 | 6.15 | 0.7458 | 0.6924 |
| minus_graph | 5.61 | 2.368 | 6.31 | 0.7511 | 0.6222 |
| minus_gate | **5.53** | **2.352** | 6.22 | **0.7545** | 0.6261 |
| minus_news | 5.72 | 2.392 | **6.12** | 0.7461 | 0.7348 |
| LSTM-only | 5.56 | 2.357 | 6.28 | 0.7534 | 0.6245 |

*h = 22 trading days*

| Config | MSE (×10⁻⁶) ↓ | RMSE (×10⁻³) ↓ | MAE (×10⁻⁴) ↓ | $R^2$ ↑ | QLIKE ↓ |
|---|---|---|---|---|---|
| HAR | **6.02** | **2.453** | 6.56 | **0.7303** | **0.6742** |
| FULL | 6.11 | 2.473 | **6.53** | 0.7260 | 0.7012 |
| minus_graph | 6.16 | 2.482 | 6.53 | 0.7239 | 0.6890 |
| minus_gate | 6.05 | 2.459 | 6.55 | 0.7290 | 0.6962 |
| minus_news | 6.17 | 2.484 | 6.61 | 0.7235 | 0.7064 |
| LSTM-only | 6.11 | 2.473 | 6.58 | 0.7260 | 0.6985 |

Reading across horizons: on the squared-error metrics (MSE, RMSE, $R^2$) the configurations fall within
about 1% of one another and the best row alternates (FULL at h5, minus_gate at h1/h10, HAR at h22); on
MAE the differences are similarly small and the lowest value varies by horizon (minus_gate at h1,
minus_news at h5, minus_news at h10, FULL at h22); on QLIKE, HAR has the lowest value at h10 and h22
while FULL is marginally lower at h1 and h5. No single configuration dominates across the five metrics.
The remaining subsections isolate each component (Section 6.2), compare FULL against HAR (Section 6.3),
and test significance (Section 6.4).

### 6.2 Component contributions across horizons

**Table 2. Component contribution on held-out test QLIKE, $\text{effect}(X)=\text{QLIKE}(\text{FULL})-\text{QLIKE}(\text{FULL}{-}X)$.**
Negative = removing $X$ raised QLIKE, i.e. $X$ helped.

| Horizon | effect(graph) | effect(gate) | effect(news) |
|---|---|---|---|
| 1 | $+0.00390$ | $+0.00489$ | $+0.00565$ |
| 5 | $+0.00318$ | $-0.00168$ | $+0.00085$ |
| 10 | $+0.07020$ | $+0.06628$ | $-0.04237$ |
| 22 | $+0.01225$ | $+0.00506$ | $-0.00518$ |

Reading (single seed): `effect(graph)` is positive at all four horizons, so removing the GAT branch
lowers QLIKE, with the largest reductions at h10 and h22. `effect(gate)` is positive at three of four
horizons (a marginal $-0.0017$ at h5). `effect(news)` is positive at h1 and h5 and negative at h10
($-0.042$) and h22 ($-0.005$), so removing the news branch raises QLIKE only at the longer horizons.
These are QLIKE effects; the corresponding MSE, RMSE, and $R^2$ differences among the same
configurations are within about 1% (Table 1).

### 6.3 FULL versus HAR across horizons

**Table 3. FULL vs HAR, held-out test, all four horizons.**

| Horizon | HAR QLIKE | FULL QLIKE | HAR $R^2$ | FULL $R^2$ | HAR RMSE (×10⁻³) | FULL RMSE (×10⁻³) |
|---|---|---|---|---|---|---|
| 1 | 0.4813 | **0.4780** | 0.8192 | **0.8221** | 2.014 | **1.998** |
| 5 | 0.5735 | **0.5724** | 0.7672 | **0.7733** | 2.287 | **2.257** |
| 10 | **0.6139** | 0.6924 | **0.7532** | 0.7458 | **2.358** | 2.393 |
| 22 | **0.6742** | 0.7012 | **0.7303** | 0.7260 | **2.453** | 2.473 |

Reading (single seed): at the short horizons h1 and h5 the full model and HAR differ only at the
third-to-fourth significant figure on QLIKE, $R^2$, and RMSE, with the full model marginally ahead; at
h10 and h22 HAR has the lower QLIKE and RMSE and the higher $R^2$, with the gaps still small on the
squared-error metrics. The Diebold-Mariano test on QLIKE (Table 4) finds the full model significantly
better than HAR only at h1 and HAR significantly better at h10 and h22, with no significant difference at
h5. The three-seed replication (companion paper) is required before the single-seed reading is treated as
final.

### 6.4 Diebold-Mariano significance

**Table 4. Diebold-Mariano (HLN) on held-out test loss, single seed (42), per horizon and per loss
family.** DM is computed on three per-observation loss series — QLIKE, squared error (SE; the
MSE/RMSE/$R^2$ family), and absolute error (AE; MAE). Each cell is the HLN-corrected DM statistic with
its two-sided $p$-value in parentheses and a trailing `*` when $p<0.05$; a **negative** statistic means
FULL has the lower loss, a **positive** statistic means the comparator does. HAC truncation lag $h{-}1$;
$n$ per horizon is 14,596 (h1), 14,464 (h5), 14,299 (h10), 13,903 (h22) present-node test observations.

*h = 1 trading day*

| Comparison | QLIKE | SE (MSE/RMSE/$R^2$) | AE (MAE) |
|---|---|---|---|
| FULL vs HAR | $-2.01$ (0.04)* | $-1.35$ (0.18) | $-0.55$ (0.58) |
| FULL vs minus_graph | $+2.18$ (0.03)* | $-0.05$ (0.96) | $+2.32$ (0.02)* |
| FULL vs minus_gate | $+3.70$ (0.00)* | $+2.24$ (0.03)* | $+4.81$ (0.00)* |
| FULL vs minus_news | $+6.38$ (0.00)* | $+2.09$ (0.04)* | $+4.62$ (0.00)* |
| FULL vs LSTM-only | $+4.56$ (0.00)* | $+0.85$ (0.40) | $+2.00$ (0.05)* |

*h = 5 trading days*

| Comparison | QLIKE | SE (MSE/RMSE/$R^2$) | AE (MAE) |
|---|---|---|---|
| FULL vs HAR | $-0.52$ (0.60) | $-2.05$ (0.04)* | $-2.18$ (0.03)* |
| FULL vs minus_graph | $+2.93$ (0.00)* | $-0.67$ (0.51) | $-1.36$ (0.17) |
| FULL vs minus_gate | $-1.04$ (0.30) | $-1.10$ (0.27) | $+0.10$ (0.92) |
| FULL vs minus_news | $+0.77$ (0.44) | $-0.97$ (0.33) | $+0.90$ (0.37) |
| FULL vs LSTM-only | $+2.24$ (0.02)* | $-0.33$ (0.74) | $-1.29$ (0.20) |

*h = 10 trading days*

| Comparison | QLIKE | SE (MSE/RMSE/$R^2$) | AE (MAE) |
|---|---|---|---|
| FULL vs HAR | $+4.96$ (0.00)* | $+1.54$ (0.12) | $-3.59$ (0.00)* |
| FULL vs minus_graph | $+4.58$ (0.00)* | $+0.98$ (0.33) | $-3.45$ (0.00)* |
| FULL vs minus_gate | $+5.46$ (0.00)* | $+1.70$ (0.09) | $-1.69$ (0.09) |
| FULL vs minus_news | $-3.58$ (0.00)* | $+0.27$ (0.79) | $+2.88$ (0.00)* |
| FULL vs LSTM-only | $+4.67$ (0.00)* | $+1.59$ (0.11) | $-3.24$ (0.00)* |

*h = 22 trading days*

| Comparison | QLIKE | SE (MSE/RMSE/$R^2$) | AE (MAE) |
|---|---|---|---|
| FULL vs HAR | $+5.19$ (0.00)* | $+1.22$ (0.22) | $-1.32$ (0.19) |
| FULL vs minus_graph | $+3.39$ (0.00)* | $-0.95$ (0.34) | $-0.29$ (0.77) |
| FULL vs minus_gate | $+1.98$ (0.05)* | $+1.95$ (0.05) | $-2.42$ (0.02)* |
| FULL vs minus_news | $-2.45$ (0.01)* | $-1.57$ (0.12) | $-5.56$ (0.00)* |
| FULL vs LSTM-only | $+1.09$ (0.27) | $-0.01$ (0.99) | $-3.64$ (0.00)* |

Reading (single seed): weighing the three loss families even-handedly, no comparison is significant in
the same direction across all of them at a given horizon. FULL vs HAR: the full model has significantly
lower QLIKE at h1 and significantly higher QLIKE at h10 and h22 (no significant difference at h5); on SE
the two differ significantly only at h5 (full model lower); on AE the full model is significantly lower at
h5 and h10 and indistinguishable at h1 and h22 — so no single model is favored across all three losses.
FULL vs minus_graph is positive and significant on QLIKE at all four horizons, shows no significant SE
difference at any horizon, and on AE is significant with the comparator lower at h1 and the full model
lower at h10. FULL vs minus_gate, FULL vs minus_news, and FULL vs LSTM-only are each significant on some
loss/horizon cells and not others (for example minus_news is significant and positive on QLIKE at h1 and
negative at h10 and h22, while on AE it is positive at h10 and negative at h22); the significant cells do
not line up across QLIKE, SE, and AE, so the ablation verdicts depend on the loss family and horizon. All
statistics are single-seed; the three-seed seed-ensembled DM is reported in the companion three-seed
paper.

---

## 7. Discussion

*The reading below is single-seed; the Diebold-Mariano table (Table 4) provides significance, and the
three-seed replication (companion paper) is required before the reading is treated as final.*

**What each component contributes.** Table 2 reports each component's marginal QLIKE contribution, while
the MSE, RMSE, and $R^2$ differences among the same configurations are within about 1% (Table 1).
`effect(graph)` is positive at all four horizons, so removing the directed lead-lag GAT branch — the
design chosen specifically to avoid the market-factor redundancy of a correlation edge — lowers QLIKE,
with the largest reductions at h10 and h22; on the squared-error loss the graph ablation is not
significant at any horizon (Table 4). This is consistent with the earlier correlation-edge finding:
redirecting the edge to a predictive volume→volatility relation and giving the attention raw features
does not change the level-metric picture. `effect(news)` is positive at h1 and h5 and negative at h10
($-0.042$) and h22 ($-0.005$), so removing the news branch raises QLIKE only at the longer horizons.
`effect(gate)` is positive at three of four horizons. Across the five metrics the full model and HAR
differ marginally at h1 and h5, and HAR has the lower QLIKE and RMSE and the higher $R^2$ at h10 and h22
(Table 3), so no configuration consistently outperforms HAR at the single-seed level.

**The price-only backbone relative to the full stack.** The LSTM-only reference (a shared pooled price
LSTM with no news, no gate, no graph) reaches test QLIKE 0.4729 / 0.5696 / 0.6245 / 0.6985 at h1 / h5 /
h10 / h22, at or below the full model's QLIKE at every horizon (0.4780 / 0.5724 / 0.6924 / 0.7012), while
its MSE, RMSE, MAE, and $R^2$ are within about 1% of the full model (Table 1). Adding the news, gate, and
graph components to the price LSTM therefore does not lower QLIKE, and at the longer horizons the full
model has the higher QLIKE. On QLIKE the LSTM-only model and HAR are comparable at h1 and h5 and HAR has
the lower value at h10 and h22, mirroring the full model; on the squared-error metrics the two are within
about 1% at all horizons, so the deep temporal model matches HAR at the short horizons and does not
overtake the linear baseline at the long horizons.

**Relation to prior project findings (established earlier, not this study's new results).** A leakage-safe
exploratory analysis on this data previously found that the extra node features (`market_pk` and
`volume_zscore`) beat HAR on QLIKE under a Diebold-Mariano test, but that a symmetric correlation-based
cross-stock edge added no reliable out-of-sample value because a single market factor dominates
cross-stock volatility co-movement and the selected neighbourhoods reshuffle out of sample. The present
study tests whether replacing that correlation edge with a directed volume→volatility lead-lag edge, and
placing it in a parallel GAT branch on raw node features, changes that conclusion.

**Relation to the literature.** A null or negligible graph effect would align with the best-controlled
published GNN-vs-HAR study, GNNHAR on DJIA-30, where multi-hop graph spillover gave no clear advantage
under a Model Confidence Set and the gains came from nonlinearity and a QLIKE training loss rather than
the graph [10], and with the broader finding that a well-specified HAR is hard to beat on a limited
information set [14].

**Levers not exercised here.** The literature identifies a QLIKE training objective and intraday-derived
realized variance as the decisive levers for beating HAR [10]; this study trains on MSE and forecasts a
daily range-based variance, so the ablation isolates architecture rather than objective or estimator.
These remain future work.

---

## 8. Limitations

First, the 33-ticker universe is a fixed, point-in-time VN30-like set, so it is not the live index. Second, the study covers a single market at
daily frequency; the results may not transfer to higher frequencies or to markets with balanced listing
histories, and every rigorous published GNN-beats-HAR result relies on intraday-derived realized variance
this daily-OHLCV panel does not have. Third, the reported numbers are **single-seed** (seed 42) and a
Diebold-Mariano test is not yet applied, so no gap is established as significant; a two-additional-seed
extension (seeds 123, 2026) with seed-ensembled Diebold-Mariano tests, and a Model Confidence Set, are
required to confirm the single-seed reading. Fourth, training uses an MSE objective rather than
the QLIKE objective the literature credits for closing the HAR gap. Fifth, directional accuracy is not
reported: the day-to-day change in the daily Parkinson target is anti-persistent (lag-1 sign
autocorrelation about $-0.30$), so the metric has no skill ceiling above chance and does not discriminate
between models.

---

## 9. Conclusion

We propose a parallel multi-branch model for multi-horizon VN30 Parkinson-variance forecasting that
fuses a price LSTM, a real multi-head GAT over a directed volume→volatility lead-lag edge, and a gated
PhoBERT news branch, with the GAT consuming raw node features to match the edge semantics. We evaluate it
with an ablation that retrains one variant per removed component and attributes each
component's contribution as the change it causes in held-out QLIKE, against a strong HAR baseline and
with Diebold-Mariano significance, at horizons 1, 5, 10, and 22 trading days. On a single-seed run, the
configurations differ by about 1% or less on MSE, RMSE, MAE, and $R^2$, with the best value alternating
across horizons; on QLIKE, HAR is lowest at h10 and h22 and the full model is marginally lower at h1, and
a Diebold-Mariano test on QLIKE finds the full model significantly better than HAR only at h1 and
significantly worse at h10 and h22. Across all five metrics no configuration consistently or
significantly outperforms HAR — a provisional reading to be confirmed by a multi-seed extension.
Directional accuracy is omitted as uninformative for an anti-persistent daily target. For a Vietnamese
emerging market, the study shows how to build and honestly evaluate a directed-spillover graph-attention
news forecaster against the field-standard HAR.

---

## References

[1] Corsi, F. A simple approximate long-memory model of realized volatility. *Journal of Financial
Econometrics* 7(2), 174–196 (2009).

[2] Parkinson, M. The extreme value method for estimating the variance of the rate of return. *The
Journal of Business* 53(1), 61–65 (1980).

[3] Nguyen, D.Q., Nguyen, A.T. PhoBERT: Pre-trained language models for Vietnamese. In *Findings of
EMNLP 2020*, 1037–1042.

[patton] Patton, A.J. Volatility forecast comparison using imperfect volatility proxies. *Journal of
Econometrics* 160(1), 246–256 (2011).

[4] Hochreiter, S., Schmidhuber, J. Long short-term memory. *Neural Computation* 9(8), 1735–1780 (1997).

[5] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., Bengio, Y. Graph attention
networks. In *ICLR* (2018).

[7] Ouyang, Z.-S., Yang, X.-T., Lai, Y. Systemic financial risk early warning of financial market in
China using an Attention-LSTM model. *North American Journal of Economics and Finance* 56, 101383 (2021).

[8] Das, N., Sadhukhan, B., Chatterjee, R., Chakrabarti, S. Integrating sentiment analysis with graph
neural networks for enhanced stock prediction. *Decision Analytics Journal* 10, 100417 (2024).

[9] Sonani, M.S., Badii, A., Moin, A. Stock price prediction using a hybrid LSTM-GNN model. arXiv
preprint arXiv:2502.15813 (2025).

[10] Zhang, C., Pu, X., Cucuringu, M., Dong, X. Forecasting realized volatility with spillover effects:
perspectives from graph neural networks (GNNHAR). *International Journal of Forecasting* 41(1), 377–397
(2025). arXiv:2308.01419.

[14] Audrino, F., Chassot, J. HARd to beat: the overlooked impact of rolling windows in the era of
machine learning. *International Journal of Forecasting* (2025). arXiv:2406.08041.

[15] Bollerslev, T., Patton, A.J., Quaedvlieg, R. Exploiting the errors: a simple approach for improved
volatility forecasting (HARQ). *Journal of Econometrics* 192(1), 1–18 (2016).

[16] Diebold, F.X., Yilmaz, K. Better to give than to receive: predictive directional measurement of
volatility spillovers. *International Journal of Forecasting* 28(1), 57–66 (2012).

[19] Chen, Q., Robert, C.-Y. Multivariate realized volatility forecasting with graph neural network. In
*ACM ICAIF* (2022). arXiv:2112.09015.
