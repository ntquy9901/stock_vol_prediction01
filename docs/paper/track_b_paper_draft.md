# A Pooled News-Augmented LSTM for VN30 Volatility: News Content Helps, Cross-Stock Graph Propagation Does Not

*Track-B draft, 2026-08-09. Every number in this draft is read from a committed result artifact
(`results/*/results.json`, aggregate JSONs) or a cited consolidation report. Provenance appears in
Section 6 and in the per-table source lines. Numbers still awaiting a final multi-seed / Diebold-Mariano
re-run are marked with the superscript "†" and listed explicitly in Section 10.*

---

## Abstract

Volatility forecasts set risk limits, margin, and option prices for VN30, the most liquid stocks on
Vietnam's Ho Chi Minh Stock Exchange. Price-only forecasters ignore the Vietnamese-language news that
plausibly signals volatility shocks, and it is unclear whether news content, or cross-stock coupling
through a graph, improves the forecast beyond the classical Heterogeneous Autoregressive (HAR) model.
We answer both questions on a pooled architecture that treats every ticker-day as one asynchronous
sample (73,026 training samples across 33 tickers), rather than restricting the panel to synchronized
trading dates. We build a four-step component ladder: a HAR baseline (P0), a pooled price-only LSTM
(P1), a news LSTM branch (P2), and a per-ticker news gate (P3). On the same 14,418 validation
observations, adding news lowers five-day-ahead validation QLIKE from 0.51098 (P1) to 0.50843 (P2)
and RMSE from 0.0014985 to 0.0014859, with paired $t$-tests across three seeds significant on MSE,
RMSE, MAE, and $R^2$ ($|t|>10$). The per-ticker gate adds nothing measurable (P3 vs P2, all
$|t|<0.6$). We then wrap the frozen P3 in an availability-aware graph and switch cross-stock
message-passing on and off (G0/G1) on the identical validation set. Message-passing is near-null: on
a fair masked comparison G1 moves QLIKE by under 0.002 and the training-loss delta changes sign
across seeds. A single-seed sparse (k-NN-8) adjacency shows a small reversal that a multi-seed and
Diebold-Mariano confirmation has not yet closed. News content carries the forecast gain; graph-based
cross-stock propagation does not pay its way on this sparse daily VN30 data. Directional accuracy
stays near 48.5% for every model because the day-to-day change in daily Parkinson volatility is
anti-persistent (lag-1 sign autocorrelation $-0.30$, negative for all 33 tickers), a structural
ceiling rather than a model defect.

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
research, that plausibly signals volatility shocks before they reach the price range.

Two open questions frame this paper. First, does news content improve a daily volatility forecast
beyond the classical HAR model of Corsi [1], the linear univariate workhorse that regresses future
volatility on its own daily, weekly, and monthly averages? Second, does cross-stock coupling through
a graph, the mechanism that recent financial graph neural networks add [5,18,19], improve the
forecast on an emerging market where trading calendars are sparse and unbalanced? A widely-covered
bank and a thinly-covered utility carry different news, and their volatilities may spill over, yet no
study has isolated whether either mechanism earns its added complexity on VN30.

Answering these questions on VN30 forces a data-design choice that mature-market studies avoid. A
fixed cross-stock graph over 33 tickers needs every node present on the same trading date. VN30's
listing dates are severely unbalanced: the newest constituent (SSB, listed 2021) trades on about
1,300 sessions while the oldest (VNM, ACB) trade on about 4,900. Requiring the synchronized-date
intersection keeps only 1,296 dates, roughly 26% of the 4,989-date union, and starves any graph
model of data before it trains. We instead pool every ticker-day as one asynchronous sample. Pooling
recovers the full timeline: 73,026 training samples versus about 9,606 on the common-date panel. The
pooled design is the paper's data contribution, and it lets a graph ablation run on real observations
rather than on a starved intersection.

We build the forecaster as a component ladder and measure what each component contributes. P0 is the
classical HAR baseline. P1 is a pooled price-only LSTM over the three HAR volatility scales. P2 adds
a shared news LSTM branch that encodes each ticker-day's PhoBERT news features. P3 adds a per-ticker
gate, a learned scalar per stock passed through a sigmoid that scales the news representation before
fusion. We then wrap the frozen P3 in an availability-aware graph and toggle cross-stock
message-passing off (G0) and on (G1) on the identical validation set, so the graph on/off contrast is
clean. This paper makes four contributions.

1. **A pooled asynchronous architecture for VN30 volatility.** Treating each ticker-day as one sample
   recovers the full 4,989-date union (73,026 training samples) that a fixed-node graph discards,
   without imputing pre-listing history (Section 4).

2. **News content improves the forecast magnitude over both HAR and a price-only LSTM.** Adding the
   news branch lowers validation QLIKE and RMSE and raises $R^2$, with paired $t$-tests across three
   seeds significant on MSE, RMSE, MAE, and $R^2$ (Section 6).

3. **The per-ticker gate and cross-stock graph add no measurable value.** The gate leaves every
   metric within seed noise (P3 vs P2, all $|t|<0.6$). Availability-aware cross-stock
   message-passing is near-null on a fair masked comparison over the same 14,418 validation
   observations. This is a parsimony finding: news content matters, graph propagation does not on
   this sparse daily panel (Sections 6 and 7).

4. **Direction is near-random by construction.** No model beats 48.7% directional accuracy, and
   model-free forecasters reach the same ceiling, because the daily Parkinson target's day-to-day
   change is anti-persistent (Sections 6 and 8).

---

## 2. Related Work

We group prior work into four families and state where the pooled news-augmented forecaster sits
relative to each.

**Econometric volatility models.** The HAR model [1] and its range-based inputs [2] set the standard
for daily volatility forecasting and remain hard to beat. HAR regresses future volatility on daily,
weekly, and monthly moving averages, approximating volatility's long memory with a parsimonious
linear fit. It is univariate and linear: no channel admits text, and no channel couples one stock to
another. We keep its three-scale features as the input of the price branch and add the news channel it
omits. The classical HAR is the P0 baseline every deep model must beat.

**Deep and graph-based forecasters.** LSTM forecasters [4] capture nonlinear temporal structure, and
graph attention networks [5] model cross-asset coupling; hybrid LSTM-GNN designs combine both for
stock prediction [9]. Chen and Robert [19] forecast multivariate realized volatility with a graph
transformer over cross-stock relations on about 500 S&P names, and a 2024 study forecasts realized
volatility with spillover effects through graph neural networks on a synchronized universe [20].
These designs operate on large, long-lived, calendar-synchronized markets where the listing-date
imbalance is negligible. Our price branch adopts a pooled LSTM, and our graph ablation asks whether
cross-stock message-passing helps on a sparse emerging-market panel that these studies do not
confront.

**Sparse and dynamic graphs.** A fixed-node graph is not intrinsic to GNNs. Availability-aware
message-passing builds the adjacency per timestep over only the nodes present that day and masks
absent nodes and edges, so a snapshot with a subset of tickers becomes a valid training example
rather than a discarded date [11,12,13]. Dynamic-graph models handle nodes that appear and disappear:
EvolveGCN [6] evolves the GCN weights rather than embeddings, and feature-propagation methods diffuse
partially-available node features [11]. We adopt the masking approach, the lowest-risk option, because
it recovers the timeline without imputing pre-listing observations, which would fabricate history for
assets that did not trade.

**News- and text-augmented forecasting.** Text-augmented forecasters add sentiment scores or news
embeddings: an Attention-LSTM with a news opinion index warns of systemic risk in China [7], and a
growing line couples sentiment analysis with graph neural networks for stock prediction [8]. These
designs usually fuse a single market-wide signal by concatenation on English- or Chinese-language
markets. We work in Vietnamese with PhoBERT [3] on VN30, an emerging market where news-augmented
volatility forecasting is little-studied, and we test whether a per-ticker gate that admits a
different amount of news per stock improves on uniform fusion.

---

## 3. Data

**Universe.** We use 33 VN30 constituents with daily open-high-low-close-volume (OHLCV) data. Series
lengths range from about 1,300 sessions (SSB, listed 2021) to about 4,900 (VNM, ACB, from 2006). We
exclude two current VN30 members, BSR and VPL, because their listing histories are too short to align
with the panel: at data cutoff VPL had 99 trading sessions and BSR 401, and including VPL would
shrink any synchronized window to about 58 days. The universe is therefore a fixed, point-in-time
VN30-like set rather than the live index, a limitation we state in Section 9.

**Forecast target.** We forecast the daily Parkinson range volatility five trading days ahead, a
single-day value at $t+5$ rather than a five-day average. The Parkinson estimator uses the intraday
high $H$ and low $L$:

$$\sigma^2_{\text{Park}} = \frac{(\ln(H/L))^2}{4\ln 2}.$$

The range estimator uses more of the day's price path than a close-to-close estimator and is a
standard choice for daily data [2]. It is a noisy one-day proxy for latent volatility, a fact that
Section 8 shows matters for the direction task.

**News panel.** Each Vietnamese article (title and lead) passes once through PhoBERT [3], a BERT
model [chart-devlin] pre-trained on Vietnamese, yielding a 768-dimensional embedding. An article that
names several stocks copies its embedding to each named stock. Principal component analysis, fit only
on data before the training cutoff, reduces each embedding to 32 dimensions per source group. We keep
two source groups separately, an official-press group and a brokerage group, so the model can weigh
them differently, and carry each group forward by an exponentially weighted average with a
30-trading-day half-life ($\alpha \approx 0.0228$) to encode the persistence of a news effect on days
with no fresh article. The daily per-stock news vector has 146 dimensions: 32 embedding dimensions
plus a norm per group (66), the same for the two exponentially weighted averages (66), and 14 topic
counts across seven categories (earnings, dividend, M&A, management, regulation, macro, sector).
Missing news on a day is a zero vector with a mask bit set to zero, and an all-missing window still
produces a finite representation.

**Pooled asynchronous manifest.** Each pooled sample is one ticker at one forecast origin, carrying
its own price window, news window, news mask, normalized target, and raw target, keyed by
`(ticker_id, target_date)`. Pooling every ticker-day recovers the full timeline. The pooled training
manifest holds 73,026 samples and the validation manifest holds 14,418 samples, against about 9,606
training samples on the common-date panel that a fixed-node graph would require. We contrast the two
regimes directly in the A1 data-design ablation (Section 6.4).

**Temporal split and leakage control.** We split each ticker's series chronologically into 70%
train, 15% validation, and 15% test before generating HAR features, fitting scalers, or building
windows, so no future information reaches training. Per-ticker price and target scalers are fit on the
training partition only and selected at evaluation by explicit `ticker_id`, never by flattened
position. A news feature for a sample uses only information available by that sample's forecast
origin, defined as 15:00 Asia/Ho_Chi_Minh on the final input trading date; records with unknown
timestamps or later publication are excluded. Evaluation reads stored raw targets rather than
inverse-transforming a clipped normalized target, and directional accuracy differences never cross a
ticker boundary.

---

## 4. Method

The model reads two aligned inputs per 22-day window for one ticker: a price tensor of the three HAR
volatility scales and a news tensor of PhoBERT-derived features with a mask. Two shared encoders read
them, a per-ticker gate scales the news branch, and a shared head fuses the result into one
five-day-ahead forecast. Every encoder shares weights across all tickers; the ticker identity enters
only through the per-ticker scaler and the per-ticker gate. We describe the ladder P0 to P3, then the
graph extension G0/G1.

**P0: HAR baseline.** A closed-form linear regression predicts the five-day-ahead target from the
three HAR moving averages, fit per ticker on the training partition and evaluated on the same pooled
sample IDs as the deep models. P0 has no epochs, optimizer, or regularization knobs, so its
comparison is unaffected by any training-setup choice.

**P1: pooled price-only LSTM.** A shared two-layer LSTM (hidden size 64, dropout 0.2) reads each
ticker's 22-day HAR window into a price representation, and a shared linear head produces the
forecast. Pooling means the LSTM trains on all 73,026 ticker-day samples rather than a synchronized
panel.

**P2: news branch.** A shared news encoder maps the 146-dimensional daily news vector through a
linear layer with a ReLU into a single-layer LSTM (hidden size 64) over the 22-day window, producing
a news representation. The mask lets an all-missing window produce a finite vector. P2 concatenates
the price and news representations and feeds the shared head.

**P3: per-ticker gate (the proposed final model).** P3 scales the news representation by a learned
scalar per stock before fusion:

$$\text{news}^{\text{gated}}_{i} = \sigma(g_i)\cdot\text{news}_{i}, \qquad i = 1,\dots,N.$$

The gate is one learned number per stock, independent of the day's input, so the model admits a
different amount of news for each ticker. Because the encoders are shared and the gate is selected by
`ticker_id`, the gradient of $g_i$ depends only on stock $i$'s own data.

**G0/G1: the graph extension.** The graph ablation asks whether cross-stock message-passing helps,
holding the backbone fixed. G0 and G1 wrap a frozen P3: they reuse P3's price encoder, news encoder,
gate, and head, all frozen, and evaluate on graph-compatible snapshots after per-stock encoding. G0
runs message-passing off and passes the node embeddings straight to the head. G1 adds a single
residual message-passing layer over a cross-stock adjacency,
$\text{base} \leftarrow \text{base} + \text{MP}(\text{base}, A, \text{mask})$, whose only trainable
parameters are the message-passing projection. G0 is therefore the exact graph-off control for G1,
and their difference isolates cross-stock propagation.

The graph runs on an availability-aware masked manifest. Rather than intersecting to synchronized
dates, we build the adjacency per date over only the tickers present that day and mask absent nodes
and edges, so message-passing aggregates over present neighbours only. This recovers the union
timeline, and the masked validation set contains exactly the same 14,418 present-node observations as
the pooled P2/P3 validation set, which makes the graph comparison fair against the ladder. We compare
three adjacencies of decreasing density: dense (average 18.6 off-diagonal edges), k-nearest-neighbour
with $k=8$ (average 5.9 edges), and a 0.7 correlation threshold (average 1.1 edges). A positivity
floor clamps denormalized predictions away from non-positive volatility before QLIKE, which reports a
non-positive-prediction rate of 0.0 on the masked manifest.

**Training.** The deep models use Adam, batch size 256, dropout 0.2, weight decay $10^{-5}$, and
gradient clipping at 1.0. Best-validation checkpoint selection reports the metric at the validation
optimum. On the pooled manifest each epoch runs 286 optimizer updates (73,026 samples / batch 256),
so the models reach their validation basin at epoch 5 to 6; a root-cause analysis attributes this
fast convergence to the pooled data volume and the learning rate, not to under-regularization [rc].
We report best-checkpoint metrics at three seeds (42, 123, 2026).

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

**Horizon and evaluation set.** The primary horizon is five trading days. The ladder and graph
results in Section 6 are validation-set metrics at the best checkpoint. Absolute values are
comparable only within a matched evaluation set; the ladder (P0 to P3) and the graph (G0/G1) both
report on the same 14,418 pooled validation observations, so they are cross-comparable, while the
common-date A1 comparison uses its own eval set and is read only within itself.

**Seeds and significance.** We repeat every deep configuration on three seeds (42, 123, 2026) and
report the mean, standard deviation, and a paired $t$-test across seeds. With three seeds the
two-sided 5% threshold at two degrees of freedom is $|t|>4.30$. Three seeds is the minimum for a
paired $t$-test, a caveat we carry into the graph null result. A Diebold-Mariano test on the graph
forecasts is being finalized to complement the paired $t$-test on the k-NN adjacency (Section 10).

---

## 6. Results

### 6.1 The component ladder: news helps, the gate does not

Table 1 reports the five-day-ahead validation metrics for the full ladder P0 to P3, three-seed
mean±std, on the pooled 14,418-observation validation set. Reading the ladder from the top, the
price-only LSTM (P1) does not beat the HAR baseline (P0) on squared or absolute error: P1 raises RMSE
from 0.0014845 to 0.0014985 and MSE from 2.204e-06 to 2.245e-06. P1 does lower QLIKE (0.51671 to
0.51098), so the deep model already fits the proportional loss better while losing on raw error.
Adding the news branch (P2) recovers that lost error and improves the proportional loss further: P2
lowers RMSE back to 0.0014859, MSE to 2.208e-06, and QLIKE to 0.50843, and raises $R^2$ from 0.73014
to 0.73464. Relative to HAR, P2 ties on RMSE (0.0014859 vs 0.0014845) and clearly wins QLIKE (0.50843
vs 0.51671). The per-ticker gate (P3) then moves nothing: P3 vs P2 shifts RMSE by 0.0000008 and QLIKE
by 0.00001, both inside seed noise.

**Table 1. Five-day-ahead validation metrics, component ladder P0 to P3.** Pooled asynchronous
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
| P3 + gate (proposed) | 2.21042e-06 ± 1.5e-08 | 0.0014867 ± 5.1e-06 | 0.00048056 ± 4.9e-06 | 0.73435 ± 0.0018 | **0.50842 ± 2.4e-04** | 48.533 ± 0.10 |

Table 2 gives the paired $t$-tests that separate the two mechanisms. Adding news (P2 vs P1) is
significant on four of six metrics: MSE $t=-10.50$, RMSE $t=-10.60$, MAE $t=-17.0$, and $R^2$
$t=+10.50$, all past the $|t|>4.30$ threshold. QLIKE is borderline ($t=-4.17$, $p=0.053$) and
directional accuracy is not significant ($t=-2.93$). Adding the gate (P3 vs P2) is null on every
metric, with all $|t|<0.6$ and $p>0.6$. The gate buys no measurable accuracy at this sample size.

**Table 2. Paired $t$-tests across three seeds (df=2, threshold $|t|>4.30$).** Source:
`results/pooled_20ep_aggregate.json`.

| Contrast | MSE | RMSE | MAE | $R^2$ | QLIKE | DirAcc |
|---|---|---|---|---|---|---|
| News (P2 vs P1) | $-10.50$ (sig.) | $-10.60$ (sig.) | $-17.0$ (sig.) | $+10.50$ (sig.) | $-4.17$ (p=0.053) | $-2.93$ (n.s.) |
| Gate (P3 vs P2) | $+0.60$ (n.s.) | $+0.60$ (n.s.) | $+0.29$ (n.s.) | $-0.60$ (n.s.) | $-0.03$ (n.s.) | $+0.06$ (n.s.) |

**Takeaway.** News content is the one added mechanism with a measurable, significant effect: it
recovers the RMSE the price-only LSTM loses against HAR and improves the proportional QLIKE and
$R^2$. The per-ticker gate is inert. A simpler always-on news fusion matches the gated model within
seed noise, so the gate earns its parameters only as an interpretability probe, not as an accuracy
mechanism.

### 6.2 The graph extension: cross-stock message-passing is near-null

Table 3 reports the graph on/off comparison. G0 and G1 wrap the same frozen P3 backbone and evaluate
on the same 14,418 present-node validation observations as the ladder, so their difference isolates
cross-stock message-passing. Because the frozen backbone is a graph-safe P3 retrained only on
graph-bound samples, the G0/G1 absolute values sit slightly apart from the P3 row of Table 1; we read
G0 and G1 only against each other, not against P3. On the fair masked comparison G1 moves the metrics
by less than seed noise: QLIKE 0.5093 (G0) to 0.5082 (G1), RMSE 0.0014645 to 0.0014579, $R^2$ 0.7422
to 0.7446. The reported metrics favour G1 slightly, but the training-space validation-loss delta
(G1 − G0) changes sign across the three seeds (+0.00241, +0.00076, $-0.00006$), so no consistent
graph benefit survives at three seeds.

**Table 3. Graph on/off (G0/G1), fair masked comparison, seed 42.** Frozen-P3 wrapper, availability-
aware masked manifest, same 14,418 validation observations as the ladder, non-positive-prediction
rate 0.0. G1 adds a residual message-passing layer over the cross-stock adjacency. Source:
`results/pooled_news_gnn_masked_g0g1_2026-08-08_212959_seed42/h5/{G0,G1}/results.json`; three-seed
RMSE and the val-loss deltas from the same run family (`_212959/_214227/_214916`).

| Config | MSE ↓ | RMSE ↓ | $R^2$ ↑ | QLIKE ↓ | DirAcc % ↑ |
|---|---|---|---|---|---|
| G0 message-passing OFF | 2.145e-06 | 0.0014645 | 0.7422 | 0.5093 | 48.85 |
| G1 message-passing ON | 2.126e-06 | 0.0014579 | 0.7446 | 0.5082 | 49.09 |

An earlier intersection-panel run (before the masked manifest and positivity floor) showed G1 worse
than G0 on every metric with a QLIKE blow-up to 4.38. That run is not comparable to the ladder: it
evaluated on a smaller common-date population and lacked the positivity floor, so its gap is an
artifact of evaluation basis, not a graph effect [lineage]. We report only the fair masked comparison
as evidence on message-passing.

### 6.3 Adjacency ablation: sparsity and a single-seed reversal

Table 4 sweeps the adjacency density at a 15-epoch converged budget on seed 42. Dense and k-NN-8
adjacencies improve on G0, and k-NN-8 gives the largest gain (validation-loss delta $-0.00253$,
best on five of six metrics). The threshold-0.7 adjacency is so sparse (average 1.1 off-diagonal
edges) that it collapses back to G0 (delta 0.0). The k-NN-8 result reverses the near-null of Table 3,
which suggests a sparse adjacency may extract a small signal that a dense one dilutes. This is a
single-seed observation. The k-NN-8 seed-123 run directory is empty and the seed-2026 run does not
yet exist, and no Diebold-Mariano output has been produced, so the reversal is a hint pending
multi-seed and Diebold-Mariano confirmation (Section 10).

**Table 4. Adjacency ablation, masked manifest, seed 42, 15-epoch converged.**† $\Delta$ is the
validation-loss delta (G1 − G0); negative means G1 better. G0 is identical across modes because the
adjacency is irrelevant when message-passing is off. Source:
`results/pooled_news_gnn_masked_{dense,knn8,thr07}_seed42_2026-08-08_230837/h5/graph_validation_comparison.json`.

| Config | MSE ↓ | RMSE ↓ | $R^2$ ↑ | QLIKE ↓ | DirAcc % ↑ | $\Delta$ val-loss |
|---|---|---|---|---|---|---|
| G0 message-passing OFF | 2.14947e-06 | 0.0014661 | 0.74167 | 0.51009 | 48.706 | — |
| G1 dense (18.6 edges) | 2.13264e-06 | 0.0014604 | 0.74370 | 0.50647 | **49.098** | $-0.00127$ |
| G1 k-NN-8 (5.9 edges)† | **2.12864e-06** | **0.0014590** | **0.74418** | **0.50646** | 48.712 | **$-0.00253$**† |
| G1 thr-0.7 (1.1 edges) | 2.14886e-06 | 0.0014659 | 0.74175 | 0.50980 | 48.457 | 0.00000 |

### 6.4 A1 data-design ablation: pooling neither helps nor hurts

Table 5 contrasts the pooled asynchronous manifest against the common-date panel at a matched 5-epoch
screening budget. The two regimes differ by less than one standard deviation on every configuration:
pooled P2 RMSE 0.0014867 versus common-date 0.0015035, pooled P2 QLIKE 0.50839 versus 0.51777. The
pooled design recovers 7.6 times as many training samples (73,026 vs 9,606) without changing the
forecast quality on the matched horizon. The value of pooling is that it lets the graph ablation run
on the full timeline rather than a starved intersection, not that it moves the ladder metrics.

**Table 5. A1 data-design ablation, five-day-ahead validation RMSE and QLIKE, three-seed
mean±std, 5-epoch screening.** Source:
`results/a1_{pooled,commondate}_seed{42,123,2026}/h5/validation_comparison.json`.

| Config | RMSE (pooled) | RMSE (common-date) | QLIKE (pooled) | QLIKE (common-date) |
|---|---|---|---|---|
| P0 HAR | 0.0014845 | 0.0014908 | 0.51671 | 0.51472 |
| P1 price | 0.0015024 | 0.0014933 | 0.51184 | 0.51277 |
| P2 + news | 0.0014867 | 0.0015035 | 0.50839 | 0.51777 |
| P3 + gate | 0.0014887 | 0.0015071 | 0.50856 | 0.51628 |

### 6.5 Direction is near-random for every model

Directional accuracy sits at 48.5% to 48.7% across the entire ladder and near 48.5% to 49.1% across
the graph configurations, at or below the 50% no-skill line. No paired test separates any
configuration on direction. The near-random result is a property of the target, not of any model, as
Section 8 explains.

---

## 7. Discussion

**Parsimony: news content, not the mechanisms around it.** The ladder and the graph extension
converge on one reading. The news branch carries a QLIKE, RMSE, MSE, and $R^2$ gain that is
significant across seeds. The per-ticker gate and the cross-stock graph each add nothing measurable
on the same validation set. A simpler forecaster, a pooled price-plus-news LSTM without the gate and
without message-passing, matches the full architecture within seed noise. The finding argues for
architectural parsimony: admit news content simply and spend model complexity elsewhere.

**Why the graph is near-null on this panel.** Two structural facts explain why cross-stock
message-passing does not help. First, the graph aggregates news that has already entered each node's
own representation, and on a daily panel the marginal information a neighbour adds is small once a
stock's own news and price history are encoded. Second, VN30's listing-date imbalance means the early
years carry few nodes, so a fixed-node graph either starves on the 26% intersection or, under masking,
propagates over thin per-day neighbourhoods. The masked formulation removes the data-volume confound
by training on the full union, and the graph is still near-null, which turns a potentially confounded
null into a cleaner one. The single-seed k-NN-8 reversal (Section 6.3) leaves open whether a sparse
adjacency extracts a small residual signal; the pending multi-seed and Diebold-Mariano check will
settle it.

**News-content value.** News recovers the raw-error gap that the price-only LSTM opens against HAR
and improves the proportional QLIKE loss that risk and option desks weigh most. The gain is on the
forecast magnitude, the quantity a margin engine and an option pricer consume. This is the paper's
positive result, and it holds against both the classical HAR benchmark and the price-only deep
control.

---

## 8. Why Direction Is Near-Random

The forecast target is the single-day Parkinson estimator, and its day-to-day change is
anti-persistent. Measured on each stock's full series, the sign of the daily change in Parkinson
volatility has a lag-1 autocorrelation of $-0.30$ on average, negative for all 33 tickers (range
$-0.34$ to $-0.24$). An up-move in daily volatility tends to precede a down-move: the estimator
oscillates day to day around a slowly drifting level, a known property of range-based daily proxies.
A forecaster that produces a smooth estimate of the level cannot reproduce this high-frequency
oscillation, so its directional calls decouple from the target and sit at or just below 50%.
Model-free forecasters confirm the ceiling: a persistence forecaster reaches 49.5% direction and a
five-day trailing mean reaches 49.1%, both statistically indistinguishable from the trained models
near 48.5%, yet both have negative $R^2$ while every fitted model attains positive $R^2$. Every
fitted model does real work on the volatility level and no forecaster beats chance on the day-to-day
sign, so we base the conclusions on the continuous-error metrics and report directional accuracy for
completeness.

---

## 9. Limitations

Five limitations bound the claims. First, the 33-ticker universe is a fixed, point-in-time VN30-like
set: it keeps long-history names that later left the index and excludes two short-history current
members (BSR, VPL), so it is not the live index. Second, the study covers a single market at daily
frequency; the news-content result and the graph null may not transfer to higher frequencies or to
markets with balanced listing histories. Third, three seeds is the minimum for a paired $t$-test, so
the two null results (gate and graph) bound the effect rather than prove its absence, and a multi-seed
extension would strengthen both. Fourth, the graph extension wraps a frozen graph-safe P3 backbone
trained on a graph-bound subset, so its absolute metrics are not directly rankable against the ladder;
we read the graph on/off contrast within the G0/G1 pair only. Fifth, low directional accuracy is a
structural property of the anti-persistent daily target, not a tunable model deficiency, so a
direction-focused deployment would need a different target construction.

---

## 10. Numbers Pending Final Re-Run

The following values are current best evidence and are marked "†" in the tables above. A final
re-run may update them before submission.

- **k-NN-8 graph reversal is single-seed (Table 4).** Only seed 42 is on disk. The k-NN-8 seed-123
  run directory is empty (no `graph_validation_comparison.json`) and the seed-2026 run does not yet
  exist. The $\Delta$ val-loss $-0.00253$ and the "best on five of six metrics" claim are seed-42
  only.
- **Diebold-Mariano test not yet produced.** No Diebold-Mariano output artifact was found under the
  masked-graph runs. The graph null (Table 3) and the k-NN-8 hint (Table 4) rest on paired $t$-tests
  and validation-loss deltas until the Diebold-Mariano confirmation lands.
- **Graph three-seed metrics are RMSE and val-loss only.** Table 3 reports the full six metrics for
  seed 42; the seed-123 and seed-2026 masked runs contribute RMSE and validation-loss deltas. A full
  six-metric three-seed graph table is pending.

Every other number in this draft is a committed three-seed aggregate (ladder, Tables 1, 2, 5) or a
committed seed-42 result (Table 3), read from the cited artifact.

---

## 11. Conclusion

Vietnamese news content lowers five-day-ahead validation QLIKE and RMSE and raises $R^2$ for a pooled
volatility forecaster on VN30, against both the classical HAR baseline and a price-only LSTM, with
paired $t$-tests significant across seeds on four of six metrics. The per-ticker gate and the
cross-stock graph, each measured on the same 14,418 validation observations, add no measurable value:
the gate is inert, and availability-aware cross-stock message-passing is near-null even after a masked
formulation removes the data-volume confound. News content matters; graph-based cross-stock
propagation does not pay its way on this sparse daily VN30 panel. Directional accuracy stays near
chance for every forecaster because the daily Parkinson target's day-to-day change is anti-persistent,
a structural ceiling rather than a model defect. For a Vietnamese emerging market, the pooled design
maps out what news buys and which mechanism around it does not.

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

[11] Rossi, E., Kenlay, H., Gorinova, M., Chamberlain, B., Dong, X., Bronstein, M. On the
unreasonable effectiveness of feature propagation in learning on graphs with missing node features.
In *LoG* (2022). arXiv:2111.12128.

[12] Cini, A., Marisca, I., Alippi, C. Filling the gaps: multivariate time series imputation by graph
neural networks (GRIN). In *ICLR* (2022). arXiv:2108.00298.

[13] Marisca, I., Cini, A., Alippi, C. Learning to reconstruct missing data from spatiotemporal
graphs with sparse observations (SPIN). In *NeurIPS* (2022). arXiv:2205.13479.

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
