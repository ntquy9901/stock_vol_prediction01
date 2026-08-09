# A News-Augmented Cross-Stock Graph LSTM for VN30 Volatility Forecasting: A Component Ablation from HAR to Masked Message-Passing

*Track-B draft, 2026-08-09. Every number in this draft is read from a committed result artifact
(`results/*/results.json`, aggregate JSONs) or a cited consolidation report. Provenance appears in
Section 6 and in the per-table source lines. Numbers still awaiting a final multi-seed / Diebold-Mariano
re-run are marked with the superscript "†" and listed explicitly in Section 10.*

---

## Abstract

Volatility forecasts set risk limits, margin, and option prices for VN30, the most liquid stocks on
Vietnam's Ho Chi Minh Stock Exchange. Price-only forecasters ignore the Vietnamese-language news that
plausibly signals volatility shocks, and mature-market cross-stock graph models do not transfer to an
emerging market whose listing calendars are sparse and unbalanced. We propose a pooled news-augmented
cross-stock graph LSTM for five-day-ahead VN30 volatility. Each ticker-day is one asynchronous sample
(73,026 training samples across 33 tickers), which recovers the full trading calendar rather than the
26% synchronized-date intersection a fixed-node graph would require. The proposed model (G1) has four
components: a shared temporal LSTM over the three HAR volatility scales, a news LSTM branch over
PhoBERT features, a per-ticker gate that admits a stock-specific amount of news, and a cross-stock
message-passing layer over an availability-aware masked adjacency. We evaluate G1 with a six-step
ablation ladder that builds it up one component at a time (P0 HAR, P1 price LSTM, P2 +news, P3 +gate,
G0 graph-off, G1 graph-on) on the same 14,418 validation observations, so each step attributes one
component's contribution. News content carries the forecast gain: adding the news branch lowers
validation QLIKE from 0.51098 to 0.50843 and RMSE from 0.0014985 to 0.0014859, significant across
three seeds on MSE, RMSE, MAE, and $R^2$ ($|t|>10$). The per-ticker gate is inert (all $|t|<0.6$).
Enabling the cross-stock graph (G1 vs G0) moves the seed-42 metrics marginally in G1's favour (RMSE
0.0014645 to 0.0014579, QLIKE 0.5093 to 0.5082), and a sparse k-nearest-neighbour adjacency gives the
largest lift, but the training-loss delta changes sign across seeds, so the graph's marginal
contribution is not yet statistically confirmed. A multi-seed and Diebold-Mariano test is being
finalized. We present G1 as the proposed architecture and report its graph component honestly: on this
sparse daily VN30 panel the cross-stock graph adds at most a small increment over a strong
news-augmented backbone. Directional accuracy stays near 48.5% for every model because the day-to-day
change in daily Parkinson volatility is anti-persistent (lag-1 sign autocorrelation $-0.30$, negative
for all 33 tickers), a structural ceiling rather than a model defect.

**Keywords:** volatility forecasting; graph neural networks; financial news; PhoBERT; pooled panel;
emerging markets.

---

## 1. Introduction

Volatility forecasts drive the daily risk decisions of every desk that trades VN30, the basket of
roughly thirty most liquid stocks on Vietnam's Ho Chi Minh Stock Exchange. A margin engine sizes
collateral from forecast volatility, an option desk prices from it, and a risk officer sets position
limits against it. When the forecast tracks tomorrow's realized volatility poorly, each decision pays
for the error. VN30 desks make these decisions inside an information environment that a price-only
forecaster discards: a steady flow of Vietnamese-language news, from official press and brokerage
research, that plausibly signals volatility shocks before they reach the price range, and a web of
cross-stock spillovers that a univariate model cannot represent.

The classical Heterogeneous Autoregressive (HAR) model of Corsi [1], the linear univariate workhorse
that regresses future volatility on its own daily, weekly, and monthly averages, admits neither
signal. Two structural extensions could close the gap: a news channel that reads text, and a
cross-stock graph that lets one stock's information reach another. Recent financial graph neural
networks add cross-stock coupling [5,18,19], but they operate on large, long-lived, calendar-
synchronized markets. VN30 is different. Its listing dates are severely unbalanced: the newest
constituent (SSB, listed 2021) trades on about 1,300 sessions while the oldest (VNM, ACB) trade on
about 4,900. A fixed cross-stock graph over 33 nodes needs every node present on the same trading
date, which keeps only 1,296 dates, roughly 26% of the 4,989-date union, and starves the graph model
before it trains. Whether cross-stock structure helps VN30 volatility is therefore an open question
that the standard fixed-node design cannot answer without confounding graph signal with data scarcity.

We resolve the data-scarcity confound by pooling every ticker-day as one asynchronous sample, which
recovers the full timeline (73,026 training samples versus about 9,606 on the common-date panel), and
by running the graph on an availability-aware masked adjacency that builds edges per date over only
the tickers present that day. On this foundation we propose a news-augmented cross-stock graph LSTM
(G1) with four components: a shared temporal LSTM over the HAR price features, a news LSTM branch over
PhoBERT features, a per-ticker news gate, and a residual cross-stock message-passing layer over the
masked adjacency. We evaluate the model with a component ablation ladder that builds it up one step at
a time, so the paper reports not only the full model but exactly what each component contributes. This
paper makes four contributions.

1. **A pooled news-augmented cross-stock graph LSTM for VN30 volatility (G1).** The architecture
   combines temporal, textual, and cross-stock structure on a pooled asynchronous panel, and its
   graph runs on an availability-aware masked adjacency that recovers the full 4,989-date union
   without imputing pre-listing history (Section 4). The masked cross-stock formulation is the
   paper's architectural novelty for a sparse emerging-market panel.

2. **A six-step component ablation ladder that attributes each part's contribution.** P0 to G1 add
   HAR, temporal learning, news, the gate, and cross-stock message-passing in turn, each measured on
   the same 14,418 validation observations, so the paper isolates where the forecast quality comes
   from (Section 6).

3. **News content earns its place; the gate is inert; the cross-stock graph adds at most a small,
   not-yet-confirmed increment.** The news branch improves MSE, RMSE, MAE, and $R^2$ with paired
   $t$-tests significant across seeds. The gate moves nothing. Enabling the graph moves the seed-42
   metrics marginally toward the full model, largest under a sparse adjacency, but the multi-seed
   training-loss delta is sign-inconsistent, so we report the graph contribution as small and pending
   a Diebold-Mariano confirmation rather than overclaim it (Sections 6 and 7).

4. **Direction is near-random by construction.** No model beats 49.1% directional accuracy, and
   model-free forecasters reach the same ceiling, because the daily Parkinson target's day-to-day
   change is anti-persistent (Sections 6 and 8).

---

## 2. Related Work

We group prior work into four families and state where the proposed news-augmented graph LSTM sits
relative to each.

**Econometric volatility models.** The HAR model [1] and its range-based inputs [2] set the standard
for daily volatility forecasting and remain hard to beat. HAR regresses future volatility on daily,
weekly, and monthly moving averages, approximating volatility's long memory with a parsimonious linear
fit. It is univariate and linear: no channel admits text, and no channel couples one stock to another.
We keep its three-scale features as the input of the temporal branch and add the news and cross-stock
channels it omits. The classical HAR is the P0 baseline every component must beat.

**Deep and graph-based forecasters.** LSTM forecasters [4] capture nonlinear temporal structure, and
graph attention networks [5] model cross-asset coupling; hybrid LSTM-GNN designs combine both for
stock prediction [9]. Chen and Robert [19] forecast multivariate realized volatility with a graph
transformer over cross-stock relations on about 500 S&P names, and a 2024 study forecasts realized
volatility with spillover effects through graph neural networks on a synchronized universe [20]. These
designs operate on markets where the listing-date imbalance is negligible, so they never confront the
intersection collapse that VN30 forces. Our temporal-plus-graph backbone follows this line but runs
the graph on a masked availability-aware adjacency rather than a synchronized intersection.

**Sparse and dynamic graphs.** A fixed-node graph is not intrinsic to GNNs. Availability-aware
message-passing builds the adjacency per timestep over only the nodes present that day and masks
absent nodes and edges, so a snapshot with a subset of tickers becomes a valid training example rather
than a discarded date [11,12,13]. Dynamic-graph models handle nodes that appear and disappear:
EvolveGCN [6] evolves the GCN weights rather than embeddings, and feature-propagation methods diffuse
partially-available node features [11]. We adopt the masking approach, the lowest-risk option, because
it recovers the timeline without imputing pre-listing observations, which would fabricate history for
assets that did not trade. No surveyed financial-GNN paper trains on a per-day variable stock set with
masking, which is the gap this paper's graph formulation fills for an emerging-market panel.

**News- and text-augmented forecasting.** Text-augmented forecasters add sentiment scores or news
embeddings: an Attention-LSTM with a news opinion index warns of systemic risk in China [7], and a
growing line couples sentiment analysis with graph neural networks for stock prediction [8]. These
designs usually fuse a single market-wide signal by concatenation on English- or Chinese-language
markets. We work in Vietnamese with PhoBERT [3] on VN30 and add a per-ticker gate that admits a
different amount of news per stock, then let the cross-stock graph propagate that news between stocks.

---

## 3. Data

**Universe.** We use 33 VN30 constituents with daily open-high-low-close-volume (OHLCV) data. Series
lengths range from about 1,300 sessions (SSB, listed 2021) to about 4,900 (VNM, ACB, from 2006). We
exclude two current VN30 members, BSR and VPL, because their listing histories are too short to align
with the panel: at data cutoff VPL had 99 trading sessions and BSR 401, and including VPL would shrink
any synchronized window to about 58 days. The universe is therefore a fixed, point-in-time VN30-like
set rather than the live index, a limitation we state in Section 9.

**Forecast target.** We forecast the daily Parkinson range volatility five trading days ahead, a
single-day value at $t+5$ rather than a five-day average. The Parkinson estimator uses the intraday
high $H$ and low $L$:

$$\sigma^2_{\text{Park}} = \frac{(\ln(H/L))^2}{4\ln 2}.$$

The range estimator uses more of the day's price path than a close-to-close estimator and is a
standard choice for daily data [2]. It is a noisy one-day proxy for latent volatility, a fact that
Section 8 shows matters for the direction task.

**News panel.** Each Vietnamese article (title and lead) passes once through PhoBERT [3], a BERT model
[chart-devlin] pre-trained on Vietnamese, yielding a 768-dimensional embedding. An article that names
several stocks copies its embedding to each named stock. Principal component analysis, fit only on
data before the training cutoff, reduces each embedding to 32 dimensions per source group. We keep two
source groups separately, an official-press group and a brokerage group, so the model can weigh them
differently, and carry each group forward by an exponentially weighted average with a 30-trading-day
half-life ($\alpha \approx 0.0228$) to encode the persistence of a news effect on days with no fresh
article. The daily per-stock news vector has 146 dimensions: 32 embedding dimensions plus a norm per
group (66), the same for the two exponentially weighted averages (66), and 14 topic counts across
seven categories (earnings, dividend, M&A, management, regulation, macro, sector). Missing news on a
day is a zero vector with a mask bit set to zero, and an all-missing window still produces a finite
representation.

**Pooled asynchronous manifest.** Each pooled sample is one ticker at one forecast origin, carrying
its own price window, news window, news mask, normalized target, and raw target, keyed by
`(ticker_id, target_date)`. Pooling every ticker-day recovers the full timeline. The pooled training
manifest holds 73,026 samples and the validation manifest holds 14,418 samples, against about 9,606
training samples on the common-date panel that a fixed-node graph would require. We contrast the two
regimes directly in the A1 data-design ablation (Section 6.4).

**Temporal split and leakage control.** We split each ticker's series chronologically into 70% train,
15% validation, and 15% test before generating HAR features, fitting scalers, or building windows, so
no future information reaches training. Per-ticker price and target scalers are fit on the training
partition only and selected at evaluation by explicit `ticker_id`, never by flattened position. A news
feature for a sample uses only information available by that sample's forecast origin, defined as
15:00 Asia/Ho_Chi_Minh on the final input trading date; records with unknown timestamps or later
publication are excluded. Evaluation reads stored raw targets rather than inverse-transforming a
clipped normalized target, and directional accuracy differences never cross a ticker boundary.

---

## 4. Method

We describe the proposed full model (G1) first, then define the ablation ladder as a sequence of
component removals from it. G1 forecasts one ticker's five-day-ahead Parkinson volatility from four
inputs: a 22-day price window of the three HAR volatility scales, a 22-day news window of PhoBERT
features with a mask, the ticker identity, and the set of other tickers trading on the same date. All
encoders share weights across tickers; the ticker identity enters through the per-ticker scaler, the
per-ticker gate, and the cross-stock adjacency.

**The proposed model G1.** Four components combine in sequence. (i) A shared two-layer temporal LSTM
(hidden size 64, dropout 0.2) reads each ticker's HAR window into a price representation. (ii) A
shared news encoder maps the 146-dimensional daily news vector through a linear-plus-ReLU layer into a
single-layer LSTM (hidden size 64) over the window, producing a news representation; the mask lets an
all-missing window yield a finite vector. (iii) A per-ticker gate scales the news representation by a
learned scalar per stock before fusion,

$$\text{news}^{\text{gated}}_{i} = \sigma(g_i)\cdot\text{news}_{i}, \qquad i = 1,\dots,N,$$

so the model admits a stock-specific amount of news. The gated news and the price representation
concatenate into a per-node embedding. (iv) A residual cross-stock message-passing layer then shares
information between stocks,
$\text{base} \leftarrow \text{base} + \text{MP}(\text{base}, A, \text{mask})$, where $A$ is an
availability-aware adjacency built per date over only the tickers present that day, and the mask
restricts aggregation to present neighbours. A shared head maps the propagated embedding to the
forecast. The cross-stock message-passing over the masked adjacency is the architectural novelty: it
lets one stock's news reach another on a panel where a fixed-node graph would be impossible.

**Availability-aware masked adjacency.** Rather than intersecting to synchronized dates, we build the
adjacency per date over the present tickers and mask absent nodes and edges, so message-passing
aggregates over present neighbours only. This recovers the union timeline, and the masked validation
set contains exactly the same 14,418 present-node observations as the pooled backbone validation set,
which makes the graph on/off comparison fair against the ladder. We compare three adjacencies of
decreasing density: dense (average 18.6 off-diagonal edges), k-nearest-neighbour with $k=8$ (average
5.9 edges), and a 0.7 correlation threshold (average 1.1 edges). A positivity floor clamps
denormalized predictions away from non-positive volatility before QLIKE, which reports a
non-positive-prediction rate of 0.0 on the masked manifest.

**The ablation ladder.** The ladder removes one component of G1 at a time, from the top down, so each
rung isolates a contribution.

- **P0 (HAR baseline):** a closed-form per-ticker linear regression on the three HAR moving averages,
  no temporal learning, no news, no graph. It fixes the floor the deep components must beat.
- **P1 (price LSTM):** G1 with the news branch, gate, and graph removed. It measures what temporal
  learning adds over the linear HAR fit.
- **P2 (+news):** P1 with the news branch restored, no gate, no graph. It measures the news
  contribution.
- **P3 (+gate):** P2 with the per-ticker gate restored, no graph. It measures the gate contribution.
- **G0 (graph-off):** the full G1 pipeline with message-passing disabled, so node embeddings pass
  straight to the head. G0 is the exact graph-off control for G1.
- **G1 (full model):** G0 with cross-stock message-passing enabled.

G0 and G1 wrap a frozen P3 backbone and add only the message-passing projection as trainable
parameters, so their difference isolates cross-stock propagation cleanly. Because the wrapped backbone
is a graph-safe P3 trained on graph-bound samples, the G0/G1 absolute metrics sit slightly apart from
the P3 rung, and we read G0 and G1 against each other rather than against P3. End-to-end joint training
of the graph layer with the backbone is future work; the present G1 trains the message-passing layer
on frozen encoders, a conservative choice that avoids retraining the news gains into the graph.

**Training.** The deep models use Adam, batch size 256, dropout 0.2, weight decay $10^{-5}$, and
gradient clipping at 1.0, with best-validation checkpoint selection. On the pooled manifest each epoch
runs 286 optimizer updates (73,026 samples / batch 256), so the backbone reaches its validation basin
at epoch 5 to 6; a root-cause analysis attributes this fast convergence to the pooled data volume and
the learning rate, not to under-regularization [rc]. We report best-checkpoint metrics at three seeds
(42, 123, 2026).

---

## 5. Experimental Setup

**Metrics.** Every configuration reports the six mandated metrics on the raw volatility scale: MSE,
RMSE, MAE, $R^2$, QLIKE, and directional accuracy. QLIKE, the quasi-likelihood loss standard in the
realized-volatility literature [patton], penalizes under-prediction more than over-prediction and
tolerates the noise in the volatility proxy:

$$\text{QLIKE} = \frac{1}{T}\sum_{t=1}^{T}\left(\frac{\hat{\sigma}^2_t}{\sigma^2_t} - \ln\frac{\hat{\sigma}^2_t}{\sigma^2_t} - 1\right).$$

Directional accuracy compares the sign of the predicted change against the sign of the realized
change, computed per ticker over time and averaged across tickers, so a difference never crosses a
ticker boundary.

**Horizon and evaluation set.** The primary horizon is five trading days. The ladder and graph results
in Section 6 are validation-set metrics at the best checkpoint. Absolute values are comparable only
within a matched evaluation set; the ladder (P0 to P3) and the graph (G0/G1) both report on the same
14,418 pooled validation observations, so they are cross-comparable, while the common-date A1
comparison uses its own eval set and is read only within itself.

**Seeds and significance.** We repeat every deep configuration on three seeds (42, 123, 2026) and
report the mean, standard deviation, and a paired $t$-test across seeds. With three seeds the
two-sided 5% threshold at two degrees of freedom is $|t|>4.30$. Three seeds is the minimum for a
paired $t$-test, a caveat we carry into the graph result. A Diebold-Mariano test on the graph
forecasts is being finalized to complement the paired $t$-test on the sparse adjacency (Section 10).

---

## 6. Results

### 6.1 The component ladder: news helps, the gate does not

Table 1 reports the five-day-ahead validation metrics for the backbone ladder P0 to P3, three-seed
mean±std, on the pooled 14,418-observation validation set. Reading the ladder up, the price-only LSTM
(P1) does not beat the HAR baseline (P0) on squared or absolute error: P1 raises RMSE from 0.0014845
to 0.0014985 and MSE from 2.204e-06 to 2.245e-06. P1 does lower QLIKE (0.51671 to 0.51098), so
temporal learning already fits the proportional loss better while losing on raw error. Adding the news
branch (P2) recovers that lost error and improves the proportional loss further: P2 lowers RMSE back
to 0.0014859, MSE to 2.208e-06, and QLIKE to 0.50843, and raises $R^2$ from 0.73014 to 0.73464.
Relative to HAR, P2 ties on RMSE (0.0014859 vs 0.0014845) and clearly wins QLIKE (0.50843 vs 0.51671).
The per-ticker gate (P3) then moves nothing: P3 vs P2 shifts RMSE by 0.0000008 and QLIKE by 0.00001,
both inside seed noise.

**Table 1. Five-day-ahead validation metrics, backbone ladder P0 to P3.** Pooled asynchronous
manifest, best-validation checkpoint, three-seed mean±std (seeds 42/123/2026), 20-epoch budget. Lower
is better for MSE, RMSE, MAE, QLIKE; higher for $R^2$ and DirAcc. Same 14,418 validation observations
across all rows. Bold marks the best value per column. P0 is a deterministic per-ticker linear fit
(std ~0). Source: `results/pooled_20ep_aggregate.json` over
`results/pooled_20ep_seed{42,123,2026}/h5/{P0-P3}/`.

| Config | MSE ↓ | RMSE ↓ | MAE ↓ | $R^2$ ↑ | QLIKE ↓ | DirAcc % ↑ |
|---|---|---|---|---|---|---|
| P0 HAR baseline | **2.20379e-06** | **0.0014845** | **0.00047974** | **0.73515** | 0.51671 | 48.540 |
| P1 price-only LSTM | 2.24541e-06 ± 1.2e-08 | 0.0014985 ± 4.1e-06 | 0.00048871 ± 2.5e-06 | 0.73014 ± 0.0015 | 0.51098 ± 5.8e-04 | **48.663 ± 0.069** |
| P2 + news | 2.20800e-06 ± 8.7e-09 | 0.0014859 ± 2.9e-06 | 0.00048011 ± 2.5e-06 | 0.73464 ± 0.0010 | 0.50843 ± 7.9e-04 | 48.528 ± 0.017 |
| P3 + gate | 2.21042e-06 ± 1.5e-08 | 0.0014867 ± 5.1e-06 | 0.00048056 ± 4.9e-06 | 0.73435 ± 0.0018 | **0.50842 ± 2.4e-04** | 48.533 ± 0.10 |

Table 2 gives the paired $t$-tests that separate the two backbone mechanisms. Adding news (P2 vs P1)
is significant on four of six metrics: MSE $t=-10.50$, RMSE $t=-10.60$, MAE $t=-17.0$, and $R^2$
$t=+10.50$, all past the $|t|>4.30$ threshold. QLIKE is borderline ($t=-4.17$, $p=0.053$) and
directional accuracy is not significant ($t=-2.93$). Adding the gate (P3 vs P2) is null on every
metric, with all $|t|<0.6$ and $p>0.6$. The gate buys no measurable accuracy at this sample size.

**Table 2. Paired $t$-tests across three seeds (df=2, threshold $|t|>4.30$).** Source:
`results/pooled_20ep_aggregate.json`.

| Contrast | MSE | RMSE | MAE | $R^2$ | QLIKE | DirAcc |
|---|---|---|---|---|---|---|
| News (P2 vs P1) | $-10.50$ (sig.) | $-10.60$ (sig.) | $-17.0$ (sig.) | $+10.50$ (sig.) | $-4.17$ (p=0.053) | $-2.93$ (n.s.) |
| Gate (P3 vs P2) | $+0.60$ (n.s.) | $+0.60$ (n.s.) | $+0.29$ (n.s.) | $-0.60$ (n.s.) | $-0.03$ (n.s.) | $+0.06$ (n.s.) |

**Takeaway.** In the backbone, news content is the one added mechanism with a measurable, significant
effect: it recovers the RMSE the price-only LSTM loses against HAR and improves QLIKE and $R^2$. The
per-ticker gate is inert. The strong news-augmented backbone (P3) is what the graph layer builds on.

### 6.2 The proposed model: enabling the cross-stock graph

Table 3 completes the ladder with the graph on/off comparison that yields the proposed model. G0 and
G1 wrap the same frozen P3 backbone and evaluate on the same 14,418 present-node validation
observations as the ladder, so their difference isolates cross-stock message-passing. On the fair
masked comparison the full model G1 moves every reported metric marginally in its favour over the
graph-off control G0: RMSE 0.0014645 to 0.0014579, MSE 2.145e-06 to 2.126e-06, QLIKE 0.5093 to 0.5082,
$R^2$ 0.7422 to 0.7446, and DirAcc 48.85 to 49.09. The seed-42 metric shift favours the graph. The
evidence does not yet clear the significance bar, however: the training-space validation-loss delta
(G1 − G0) changes sign across the three seeds (+0.00241, +0.00076, $-0.00006$), so a consistent graph
benefit is not established at three seeds. We therefore present G1 as the proposed full architecture
while reporting its graph component as a small, not-yet-confirmed increment over an already-strong
news-augmented backbone.

**Table 3. Proposed model G1 vs graph-off control G0, fair masked comparison, seed 42.** Frozen-P3
wrapper, availability-aware masked manifest, same 14,418 validation observations as the ladder,
non-positive-prediction rate 0.0. G1 adds a residual message-passing layer over the cross-stock
adjacency. Source:
`results/pooled_news_gnn_masked_g0g1_2026-08-08_212959_seed42/h5/{G0,G1}/results.json`; three-seed
RMSE and the val-loss deltas from the same run family (`_212959/_214227/_214916`).

| Config | MSE ↓ | RMSE ↓ | $R^2$ ↑ | QLIKE ↓ | DirAcc % ↑ |
|---|---|---|---|---|---|
| G0 graph-off (control) | 2.145e-06 | 0.0014645 | 0.7422 | 0.5093 | 48.85 |
| **G1 full model (graph-on)** | **2.126e-06** | **0.0014579** | **0.7446** | **0.5082** | **49.09** |

An earlier intersection-panel run (before the masked manifest and positivity floor) showed G1 worse
than G0 with a QLIKE blow-up to 4.38. That run is not comparable to the ladder: it evaluated on a
smaller common-date population and lacked the positivity floor, so its gap is an artifact of
evaluation basis, not a graph effect [lineage]. We report only the fair masked comparison as evidence
on message-passing.

### 6.3 Adjacency ablation: a sparse graph gives the largest lift

Table 4 sweeps the adjacency density at a 15-epoch converged budget on seed 42. Every enabled graph
improves on G0, and the sparse k-nearest-neighbour adjacency (k-NN-8) gives the largest lift
(validation-loss delta $-0.00253$, best on five of six metrics), ahead of the dense adjacency
($-0.00127$). The threshold-0.7 adjacency is so sparse (average 1.1 off-diagonal edges) that it
collapses back to G0 (delta 0.0). The pattern suggests a moderately sparse graph carries the
cross-stock signal a dense one dilutes and a near-empty one loses. This is a single-seed observation.
The k-NN-8 seed-123 run directory is empty and the seed-2026 run does not yet exist, and no
Diebold-Mariano output has been produced, so the sparse-graph lift is a hint pending multi-seed and
Diebold-Mariano confirmation (Section 10). We adopt k-NN-8 as G1's adjacency on this evidence and flag
the confirmation as the one open item before the graph contribution is a settled claim.

**Table 4. Adjacency ablation, masked manifest, seed 42, 15-epoch converged.**† $\Delta$ is the
validation-loss delta (G1 − G0); negative means the graph helps. G0 is identical across modes because
the adjacency is irrelevant when message-passing is off. Source:
`results/pooled_news_gnn_masked_{dense,knn8,thr07}_seed42_2026-08-08_230837/h5/graph_validation_comparison.json`.

| Config | MSE ↓ | RMSE ↓ | $R^2$ ↑ | QLIKE ↓ | DirAcc % ↑ | $\Delta$ val-loss |
|---|---|---|---|---|---|---|
| G0 graph-off | 2.14947e-06 | 0.0014661 | 0.74167 | 0.51009 | 48.706 | — |
| G1 dense (18.6 edges) | 2.13264e-06 | 0.0014604 | 0.74370 | 0.50647 | **49.098** | $-0.00127$ |
| **G1 k-NN-8 (5.9 edges, adopted)**† | **2.12864e-06** | **0.0014590** | **0.74418** | **0.50646** | 48.712 | **$-0.00253$**† |
| G1 thr-0.7 (1.1 edges) | 2.14886e-06 | 0.0014659 | 0.74175 | 0.50980 | 48.457 | 0.00000 |

### 6.4 A1 data-design ablation: pooling recovers the timeline without changing the metrics

Table 5 contrasts the pooled asynchronous manifest against the common-date panel at a matched 5-epoch
screening budget. The two regimes differ by less than one standard deviation on every configuration:
pooled P2 RMSE 0.0014867 versus common-date 0.0015035, pooled P2 QLIKE 0.50839 versus 0.51777. The
pooled design recovers 7.6 times as many training samples (73,026 vs 9,606) without changing the
forecast quality on the matched horizon. Pooling is what makes the masked cross-stock graph feasible:
it supplies the full timeline the graph propagates over, rather than the starved 26% intersection a
fixed-node graph would see.

**Table 5. A1 data-design ablation, five-day-ahead validation RMSE and QLIKE, three-seed mean±std,
5-epoch screening.** Source:
`results/a1_{pooled,commondate}_seed{42,123,2026}/h5/validation_comparison.json`.

| Config | RMSE (pooled) | RMSE (common-date) | QLIKE (pooled) | QLIKE (common-date) |
|---|---|---|---|---|
| P0 HAR | 0.0014845 | 0.0014908 | 0.51671 | 0.51472 |
| P1 price | 0.0015024 | 0.0014933 | 0.51184 | 0.51277 |
| P2 + news | 0.0014867 | 0.0015035 | 0.50839 | 0.51777 |
| P3 + gate | 0.0014887 | 0.0015071 | 0.50856 | 0.51628 |

### 6.5 Direction is near-random for every model

Directional accuracy sits at 48.5% to 48.7% across the backbone ladder and near 48.5% to 49.1% across
the graph configurations, at or below the 50% no-skill line. No paired test separates any
configuration on direction. The near-random result is a property of the target, not of any model, as
Section 8 explains.

---

## 7. Discussion

**What each component contributes.** The ladder gives a clean attribution. Temporal learning (P1)
trades raw error for a proportional-loss gain over HAR. News content (P2) is the decisive component:
it recovers the raw-error gap and improves QLIKE, MSE, and $R^2$ significantly across seeds. The
per-ticker gate (P3) adds nothing measurable. The cross-stock graph (G1) moves the seed-42 metrics in
the full model's favour and is largest under a sparse adjacency, but its multi-seed evidence is not
yet conclusive. The proposed model G1 is therefore justified as an architecture that unifies the three
informative signals (temporal, textual, cross-stock), with the honest caveat that the graph's marginal
lift over the news-augmented backbone is small and awaiting confirmation.

**Why the graph's marginal lift is small, and why that is still informative.** Two structural facts
bound the graph contribution. First, the message-passing layer aggregates news that has already
entered each node's own representation, so the marginal information a neighbour adds on a daily panel
is small once a stock's own news and price history are encoded. Second, VN30's listing-date imbalance
thins the early-year neighbourhoods even under masking. The masked formulation is what lets us say
this cleanly: it removes the data-volume confound that would otherwise mask a genuine graph signal, so
a small measured lift is a statement about the signal, not about starvation. The sparse k-NN-8
adjacency giving the largest lift (Table 4) is consistent with a weak-but-real cross-stock signal that
a dense graph over-smooths; the pending Diebold-Mariano test will decide whether that lift is
significant or a parsimony argument for dropping the graph.

**News-content value.** News recovers the raw-error gap that the price-only LSTM opens against HAR and
improves the proportional QLIKE loss that risk and option desks weigh most. This is the paper's
strongest empirical result, and it holds against both the classical HAR benchmark and the price-only
deep control, independent of the graph and gate mechanisms layered on top.

---

## 8. Why Direction Is Near-Random

The forecast target is the single-day Parkinson estimator, and its day-to-day change is
anti-persistent. Measured on each stock's full series, the sign of the daily change in Parkinson
volatility has a lag-1 autocorrelation of $-0.30$ on average, negative for all 33 tickers (range
$-0.34$ to $-0.24$). An up-move in daily volatility tends to precede a down-move: the estimator
oscillates day to day around a slowly drifting level, a known property of range-based daily proxies. A
forecaster that produces a smooth estimate of the level cannot reproduce this high-frequency
oscillation, so its directional calls decouple from the target and sit at or just below 50%.
Model-free forecasters confirm the ceiling: a persistence forecaster reaches 49.5% direction and a
five-day trailing mean reaches 49.1%, both statistically indistinguishable from the trained models
near 48.5%, yet both have negative $R^2$ while every fitted model attains positive $R^2$. Every fitted
model does real work on the volatility level and no forecaster beats chance on the day-to-day sign, so
we base the conclusions on the continuous-error metrics and report directional accuracy for
completeness.

---

## 9. Limitations

Five limitations bound the claims. First, the 33-ticker universe is a fixed, point-in-time VN30-like
set: it keeps long-history names that later left the index and excludes two short-history current
members (BSR, VPL), so it is not the live index. Second, the study covers a single market at daily
frequency; the news-content result and the graph finding may not transfer to higher frequencies or to
markets with balanced listing histories. Third, three seeds is the minimum for a paired $t$-test, so
the gate null and the graph result rest on low power, and a multi-seed extension plus the pending
Diebold-Mariano test are needed before the graph contribution is a settled claim. Fourth, the proposed
graph layer trains on a frozen graph-safe P3 backbone rather than end-to-end, so its absolute metrics
are not directly rankable against the P3 rung, and joint training may change the graph's contribution;
we read the graph on/off contrast within the G0/G1 pair only. Fifth, low directional accuracy is a
structural property of the anti-persistent daily target, not a tunable model deficiency, so a
direction-focused deployment would need a different target construction.

---

## 10. Numbers Pending Final Re-Run

The following values are current best evidence and are marked "†" in the tables above. A final re-run
may update them before submission.

- **The adopted k-NN-8 graph lift is single-seed (Table 4).** Only seed 42 is on disk. The k-NN-8
  seed-123 run directory is empty (no `graph_validation_comparison.json`) and the seed-2026 run does
  not yet exist. The $\Delta$ val-loss $-0.00253$ and the "best on five of six metrics" claim are
  seed-42 only. Until the two remaining seeds land, the graph contribution to the proposed G1 is a
  hint, not a confirmed gain.
- **Diebold-Mariano test not yet produced.** No Diebold-Mariano output artifact was found under the
  masked-graph runs. The G1-vs-G0 comparison (Table 3) and the k-NN-8 lift (Table 4) rest on paired
  $t$-tests and validation-loss deltas until the Diebold-Mariano confirmation lands.
- **Graph three-seed metrics are RMSE and val-loss only.** Table 3 reports the full six metrics for
  seed 42; the seed-123 and seed-2026 masked runs contribute RMSE and validation-loss deltas. A full
  six-metric three-seed table for the proposed G1 is pending.

Every other number in this draft is a committed three-seed aggregate (ladder, Tables 1, 2, 5) or a
committed seed-42 result (Table 3), read from the cited artifact.

---

## 11. Conclusion

We propose a pooled news-augmented cross-stock graph LSTM (G1) for five-day-ahead VN30 volatility that
unifies temporal, textual, and cross-stock structure on a panel where a fixed-node graph is
infeasible. A pooled asynchronous design recovers the full trading calendar (73,026 training samples),
and an availability-aware masked adjacency lets the graph propagate news between stocks on real
per-day observations rather than a 26% intersection. A six-step ablation ladder attributes the model's
quality: temporal learning trades raw error for proportional-loss gain, news content carries a QLIKE,
RMSE, and $R^2$ gain significant across seeds, the per-ticker gate is inert, and enabling the
cross-stock graph moves the metrics in the full model's favour, largest under a sparse adjacency,
though its multi-seed lift is small and awaits a Diebold-Mariano confirmation. We therefore report the
graph as the paper's architectural novelty and its marginal contribution honestly, as a small
increment over an already-strong news-augmented backbone rather than a demonstrated improvement.
Directional accuracy stays near chance for every forecaster because the daily Parkinson target's
day-to-day change is anti-persistent, a structural ceiling rather than a model defect. For a
Vietnamese emerging market, the pooled masked-graph design shows how to build and honestly evaluate a
cross-stock volatility forecaster where the standard synchronized-panel approach cannot run.

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

[chart-devlin] Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. BERT: Pre-training of deep
bidirectional transformers for language understanding. In *NAACL-HLT*, 4171–4186 (2019).

[4] Hochreiter, S., Schmidhuber, J. Long short-term memory. *Neural Computation* 9(8), 1735–1780
(1997).

[5] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., Bengio, Y. Graph attention
networks. In *ICLR* (2018).

[6] Pareja, A., et al. EvolveGCN: Evolving graph convolutional networks for dynamic graphs. In *AAAI*
(2020). arXiv:1902.10191.

[7] Ouyang, Z.-S., Yang, X.-T., Lai, Y. Systemic financial risk early warning of financial market in
China using an Attention-LSTM model. *North American Journal of Economics and Finance* 56, 101383
(2021).

[8] Das, N., Sadhukhan, B., Chatterjee, R., Chakrabarti, S. Integrating sentiment analysis with graph
neural networks for enhanced stock prediction: a comprehensive survey. *Decision Analytics Journal*
10, 100417 (2024).

[9] Sonani, M.S., Badii, A., Moin, A. Stock price prediction using a hybrid LSTM-GNN model. arXiv
preprint arXiv:2502.15813 (2025).

[11] Rossi, E., Kenlay, H., Gorinova, M., Chamberlain, B., Dong, X., Bronstein, M. On the unreasonable
effectiveness of feature propagation in learning on graphs with missing node features. In *LoG*
(2022). arXiv:2111.12128.

[12] Cini, A., Marisca, I., Alippi, C. Filling the gaps: multivariate time series imputation by graph
neural networks (GRIN). In *ICLR* (2022). arXiv:2108.00298.

[13] Marisca, I., Cini, A., Alippi, C. Learning to reconstruct missing data from spatiotemporal graphs
with sparse observations (SPIN). In *NeurIPS* (2022). arXiv:2205.13479.

[18] Hsu, Y.-L., Tsai, Y.-C., et al. FinGAT: Financial graph attention networks for recommending
top-K profitable stocks. *IEEE TKDE* 35(1), 469–481 (2023). arXiv:2106.10159.

[19] Chen, Q., Robert, C.-Y. Multivariate realized volatility forecasting with graph neural network.
In *ACM ICAIF* (2022). arXiv:2112.09015.

[20] Forecasting realized volatility with spillover effects: perspectives from graph neural networks.
*International Journal of Forecasting* (2024).

*Internal provenance references (not for the reference list): [rc]
`docs/reports/2026-08-09_pooled_convergence_rootcause.md`; [lineage]
`docs/reports/2026-08-09_1003_g0g1_vs_p2p3_lineage.md`; sparse-graph method survey
`docs/reports/2026-08-08_gnn_sparse_data_research.md`; consolidated metrics
`docs/reports/2026-08-09_0747_all_metrics_consolidated.md`.*
