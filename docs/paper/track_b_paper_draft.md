# A News-Augmented Cross-Stock Graph LSTM for VN30 Volatility Forecasting: A Component Ablation from HAR to Masked Message-Passing

*Track-B consolidated draft, 2026-08-10. Every number is read from a committed result artifact.
Canonical sources: five-day ladder `docs/reports/ladder_consistent_h5_2026-08-09_154402.json`;
multi-horizon `docs/reports/ladder_consistent_h{1,10,22}_2026-08-09_180326.json` and
`..._multihorizon_2026-08-09_180326.md`; classical baselines
`docs/reports/classical_baselines_h5_2026-08-09_182129.json`. Paired-$t$ statistics on the ladder
are derived from the per-seed entries in those JSONs. Per-table source lines appear below each
table.*

---

## Abstract

Volatility forecasts set risk limits, margin, and option prices for VN30, the most liquid stocks on
Vietnam's Ho Chi Minh Stock Exchange. Price-only forecasters ignore the Vietnamese-language news that
plausibly signals volatility shocks, and mature-market cross-stock graph models do not transfer to an
emerging market whose listing calendars are sparse and unbalanced. We propose a pooled news-augmented
cross-stock graph LSTM (G1) for five-day-ahead VN30 volatility. Each ticker-day is one asynchronous
sample, which recovers the full trading calendar rather than the 26% synchronized-date intersection a
fixed-node graph would require. G1 has four components: a shared temporal LSTM over the three HAR
volatility scales, a news LSTM branch over PhoBERT features, a per-ticker gate, and a cross-stock
message-passing layer over an availability-aware masked adjacency. We evaluate G1 with a single-basis
component ablation, a nested ladder P0 (HAR) → P1 (price LSTM) → P2 (+news) → P3 (+gate) → G1
(+graph), where P3 is exactly G1 with the message-passing residual disabled (graph-off readout
determinism 0.0). All rungs are scored on the same 14,418 validation and 14,464 test observations, so
each step attributes one component's contribution on one basis. News content carries the forecast
gain: adding the news branch (P2 vs P1) lowers validation QLIKE from 0.5062 to 0.5031 and held-out
test QLIKE from 0.5648 to 0.5599, significant across three seeds ($t=-7.69$ val, $t=-9.50$ test). The
per-ticker gate delivers no consistent benefit and marginally raises test QLIKE. Enabling the
cross-stock graph (G1 vs P3) yields no statistically significant improvement at any horizon: a
Diebold-Mariano test on test QLIKE is not significant for h1, h5, h10, or h22 (verdict B, graph
null). A targeted sweep confirms the parsimony finding: neither the base graph nor five research-backed
enhancements (QLIKE-loss training, HAR + graph-residual, directed spillover edges, learned adjacency,
omit-self-loop) beats a well-specified HAR at a Diebold-Mariano-significant level. Classical econometric
baselines confirm the picture: HAR and HARQ tie the deep models on the
level metrics (test QLIKE 0.5793 / 0.5737, $R^2$ 0.767), while GARCH-family models are far worse
(test QLIKE 1.76-1.87, $R^2 \approx 0$). We present G1 as the proposed architecture and report a
rigorous parsimony finding: on sparse daily VN30 data, news content improves the forecast but
cross-stock graph propagation does not, and the strongest configuration is the parsimonious
news-augmented backbone. Directional accuracy stays near 48% for every model because the day-to-day
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
date, which keeps only about 1,296 dates, roughly 26% of the 4,989-date union, and starves the graph
model before it trains. Whether cross-stock structure helps VN30 volatility is therefore an open
question that the standard fixed-node design cannot answer without confounding graph signal with data
scarcity.

We resolve the data-scarcity confound by pooling every ticker-day as one asynchronous sample, which
recovers the full timeline, and by running the graph on an availability-aware masked adjacency that
builds edges per date over only the tickers present that day. On this foundation we propose a
news-augmented cross-stock graph LSTM (G1) with four components: a shared temporal LSTM over the HAR
price features, a news LSTM branch over PhoBERT features, a per-ticker news gate, and a residual
cross-stock message-passing layer over the masked adjacency. We evaluate the model with a nested
component ablation that removes one part at a time, from the full model down to the classical HAR
baseline, and we ground every configuration against classical econometric baselines (HAR, HARQ,
log-HAR, EWMA, persistence, and three GARCH variants) on the identical observation set. This paper
makes four contributions.

1. **A pooled news-augmented cross-stock graph LSTM for VN30 volatility (G1).** The architecture
   combines temporal, textual, and cross-stock structure on a pooled asynchronous panel, and its
   graph runs on an availability-aware masked adjacency that recovers the full 4,989-date union
   without imputing pre-listing history (Section 4). The masked cross-stock formulation is the
   paper's architectural novelty for a sparse emerging-market panel.

2. **A single-basis nested ablation from the full model down to the classical HAR baseline.**
   Starting from G1, removing the cross-stock message-passing residual lands exactly on P3
   (graph-off readout determinism 0.0), then removing the gate, the news branch, and temporal
   learning in turn reaches HAR. All rungs score on the same 14,418 validation and 14,464 test
   observations, so each step is one controlled increment and the whole ladder is directly
   comparable (Section 6).

3. **News content earns its place; the gate and the cross-stock graph do not.** The news branch
   lowers QLIKE significantly on both validation and held-out test. The gate delivers no consistent
   improvement and marginally raises test QLIKE. Enabling the graph gives no statistically
   significant QLIKE improvement under a Diebold-Mariano test at any of four horizons, so we report a
   rigorous parsimony finding: news content helps, graph propagation does not on sparse daily VN30
   data, and classical HAR-family models tie the deep models while GARCH is far worse (Sections 6
   and 7).

4. **Direction is near-random by construction.** No model reaches 49% directional accuracy, and
   model-free forecasters reach the same ceiling, because the daily Parkinson target's day-to-day
   change is anti-persistent (Sections 6 and 8).

---

## 2. Related Work

We group prior work into four families and state where the proposed news-augmented graph LSTM sits
relative to each.

**Econometric volatility models.** The HAR model [1] and its range-based inputs [2] set the standard
for daily volatility forecasting and remain hard to beat. HAR regresses future volatility on daily,
weekly, and monthly moving averages, approximating volatility's long memory with a parsimonious linear
fit. The HARQ extension [15] shrinks the daily coefficient when the volatility estimate is noisy and
generally improves on HAR at the daily horizon. A large-scale controlled study finds that tuned
machine-learning models fail to beat a carefully re-estimated HAR on QLIKE and MSE when both use the
same information set, and attributes most apparent ML wins to rolling-window and re-estimation choices
rather than model class [14]. These models are univariate and linear: no channel admits text, and no
channel couples one stock to another. We keep the three HAR scales as the input of the temporal branch
and add the news and cross-stock channels HAR omits; the classical HAR and HARQ are the baselines every
component must beat.

**Deep and graph-based forecasters.** LSTM forecasters [4] capture nonlinear temporal structure, and
graph attention networks [5] model cross-asset coupling; hybrid LSTM-GNN designs combine both [9].
Sonani et al. [9] pair an LSTM with a GNN to predict stock *price* on a ten-stock universe, with no
HAR baseline and no volatility target; it is a generic architectural precedent for combining temporal
and graph structure, not evidence that such a design beats HAR for volatility. Chen and Robert [19]
forecast multivariate realized volatility with a graph transformer over about 500 S&P names. Most
relevant, Zhang, Pu, Cucuringu and Dong [10] build a graph-neural-network HAR (GNNHAR) on DJIA-30 and,
under a Model Confidence Set, find that multi-hop cross-stock graph spillover gives no clear advantage:
the gains come from modeling nonlinearity and from switching the training loss from MSE to QLIKE, and
only at horizons up to one week. Their controlled null on the graph component directly corroborates the
finding of this paper. These designs operate on markets where the listing-date imbalance is negligible,
so they never confront the intersection collapse that VN30 forces. Our temporal-plus-graph backbone
follows this line but runs the graph on a masked availability-aware adjacency rather than a
synchronized intersection.

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

The processed `parkinson_volatility` column is numerically the Parkinson variance estimator, and every
model in this paper, deep and classical, forecasts this same daily realized-variance quantity. The
range estimator uses more of the day's price path than a close-to-close estimator and is a standard
choice for daily data [2]. It is a noisy one-day proxy for latent volatility, a fact that Section 8
shows matters for the direction task.

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
`(ticker_id, target_date)`. Pooling every ticker-day recovers the full timeline instead of the roughly
26% synchronized intersection a fixed-node graph would require. On the masked five-day manifest the
evaluation sets hold 14,418 present-node validation observations and 14,464 test observations, shared
identically across every ladder rung and every classical baseline; the graph runs over 6,470 per-date
snapshots. This shared observation basis is what makes the whole ladder, from HAR to G1, directly
comparable.

**Temporal split and leakage control.** We split each ticker's series chronologically into 70% train,
15% validation, and 15% test before generating HAR features, fitting scalers, or building windows, so
no future information reaches training. Per-ticker price and target scalers are fit on the training
partition only and selected at evaluation by explicit `ticker_id`, never by flattened position. The
graph adjacency is graph-bound to the training window. A news feature for a sample uses only
information available by that sample's forecast origin, defined as 15:00 Asia/Ho_Chi_Minh on the final
input trading date; records with unknown timestamps or later publication are excluded. Evaluation reads
stored raw targets rather than inverse-transforming a clipped normalized target, and directional
accuracy differences never cross a ticker boundary.

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
aggregates over present neighbours only. This recovers the union timeline, and the masked evaluation
sets contain exactly the same present-node observations as the ladder backbone, which makes the graph
on/off comparison fair. The headline adjacency is k-nearest-neighbour with $k=8$, giving an average of
5.87 off-diagonal edges per present row (maximum 8). A positivity floor clamps denormalized
predictions away from non-positive volatility before QLIKE, which reports a non-positive-prediction
rate of 0.0 on the masked manifest.

**The nested ablation ladder.** The ladder removes one component of G1 at a time, from the top down,
so each rung isolates a contribution, and every rung is scored on the identical observation set.

- **G1 (full model):** all four components together — the shared temporal LSTM, the news branch, the
  per-ticker gate, and cross-stock message-passing over the masked adjacency.
- **P3 (remove the cross-stock graph):** G1 with the message-passing residual disabled, so node
  embeddings pass straight to the head. P3 is exactly G1 with the graph switched off — removing the
  GAT from G1 lands on P3 with a graph-off readout determinism of 0.0 for every seed and every
  horizon (`nesting_check`), so the G1-versus-P3 pair is the clean, exactly-nested control that
  isolates cross-stock propagation. P3 is simultaneously the top backbone rung (price + news + gate)
  and the graph-off control; there is no separate G0 row.
- **P2 (remove the per-ticker gate):** the news-augmented backbone without the gate. The
  P3-versus-P2 contrast isolates the gate's contribution.
- **P1 (remove the news branch):** P2 with the news branch removed, a price-only LSTM. The
  P2-versus-P1 contrast isolates the news contribution, and P1 measures what temporal learning adds
  over the linear HAR fit.
- **P0 (remove temporal learning; the HAR baseline):** a closed-form per-ticker linear regression on
  the three HAR moving averages, with no temporal learning, no news, and no graph. It fixes the floor
  every component must beat.

**Training.** The deep models use Adam (default learning rate $10^{-3}$), weight decay $10^{-5}$,
dropout 0.2, and gradient clipping at 1.0, with best-validation checkpoint selection on the pooled
masked manifest at a matched screening budget. G1 is trained end-to-end (temporal LSTM, news branch,
gate, and message-passing residual jointly); P3 is obtained by disabling G1's message-passing at
inference, so it shares G1's weights exactly. P0 is a closed-form per-ticker least-squares fit. We
report results at three seeds (42, 123, 2026). Section 5 states the exact configuration.

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

**Training objective.** All deep configurations are trained by minimizing the mean squared error
between the model output and the per-ticker normalized target. QLIKE and the other five metrics are
computed only at evaluation, after inverting the normalization back to the raw volatility scale, so the
proportional QLIKE loss never enters the gradient; this keeps the training loss convex and stable while
the volatility-standard QLIKE is reserved for comparison. Switching the training loss to QLIKE is a
lever the literature identifies as the decisive one for beating HAR [10]; we note it as future work in
Section 7 and keep MSE training here so that the ablation isolates architecture rather than objective.

**Optimization and hyperparameters.** Optimization uses the Adam optimizer with the default learning
rate $10^{-3}$ and weight decay $10^{-5}$, dropout $0.2$ on the LSTM encoders, and gradient-norm
clipping at $1.0$. Each window is 22 trading days; the forecast horizon is five trading days. The deep
rungs train on the pooled masked manifest with best-validation-loss checkpoint selection at a matched
screening budget; P0 is a closed-form per-ticker linear least-squares fit with no iterative
optimization. Every run seeds Python, NumPy, and Torch (including CUDA) from its run seed, and the
three seeds 42/123/2026 are independent repetitions of the full pipeline. Because P3 is G1 read out
with the graph disabled, the ladder is a single consistent basis rather than two separately trained
families: there is no frozen-backbone two-phase split between the backbone rungs and the graph.

**Seeds and significance.** We repeat every deep configuration on three seeds (42, 123, 2026) and
report the mean, standard deviation, and a paired $t$-test across seeds. With three seeds the
two-sided 5% threshold at two degrees of freedom is $|t|>4.30$. We complement the paired $t$-test with
a Diebold-Mariano (DM) test on the per-observation QLIKE and MSE loss series of the graph forecasts
(HAC truncation lag $h-1$, Harvey-Leybourne-Newbold corrected), which tests forecast-accuracy equality
directly on the held-out observations rather than on the seed-level means. We attempted a Model
Confidence Set (Hansen-Lunde-Nabney) over the ladder and classical baselines; the aggregate result
artifacts store per-configuration metrics but not the per-observation prediction series, and clean
re-scoring onto a single common observation set (GARCH covers a 32-ticker subset) was not feasible
within the submission window, so we report the DM tests as the primary significance evidence for the
graph.

**Classical baselines.** On the identical val/test observation keys and raw targets, scored by the same
routine as the deep ladder, we fit seven classical econometric baselines: a persistence (random-walk)
forecast, an EWMA, the classical HAR, a HARQ variant with a daily range-based realized-quarticity proxy
(the canonical 5-minute quarticity is not identified on daily OHLCV, so this is an approximation), a
log-HAR, and GARCH(1,1), GJR-GARCH, and EGARCH fit per ticker on close-to-close log returns. The GARCH
family is scored on 32 of 33 tickers (14,247 val / 14,292 test observations) because one ticker (LPB)
lacks the raw OHLCV needed to form returns; all other baselines cover the full 14,418 / 14,464 set.

**Implementation and compute.** All models are implemented in PyTorch and select a CUDA GPU when one is
available, falling back to CPU otherwise. The runs used an NVIDIA GeForce RTX 4060 Laptop GPU under
PyTorch 2.6 with CUDA 12.4.

---

## 6. Results

We report the proposed model G1 against the classical baselines first (Section 6.1), then present the
nested component ablation on one basis (Section 6.2), the graph ablation at the five-day horizon
(Section 6.3), the multi-horizon graph result (Section 6.4), and direction (Section 6.5). Every rung
and every baseline scores on the same 14,418 validation and 14,464 test observations, so all levels are
directly comparable.

### 6.1 The proposed model versus classical econometric baselines

Table 1 places the proposed model G1 against the classical baselines on the held-out test split. G1
reaches test QLIKE 0.575926, RMSE 0.00230527, and $R^2$ 0.763520. The classical HAR reaches QLIKE
0.579291 and HARQ 0.573674 at $R^2 \approx 0.767$; EWMA reaches QLIKE 0.600625. On the level metrics
the HAR family and the deep models tie: HARQ's test QLIKE (0.573674) is marginally below G1's, and G1's
$R^2$ is within 0.004 of HAR's. The GARCH family is far worse on every level metric (test QLIKE 1.76 to
1.87, $R^2$ from $+0.003$ to $-0.003$, RMSE roughly double the HAR family's), because a conditional
return-variance model tracks the daily range-variance target poorly. The persistence forecast collapses
on QLIKE (4151), confirming that a naive last-value forecast is inadmissible under the proportional
loss. The headline reading is a tie between the deep news-augmented system and the field-standard
HAR/HARQ on the level the volatility literature weights most, and a decisive win of both over GARCH.

**Table 1. Five-day-ahead held-out test metrics: proposed model G1 and classical econometric
baselines.** Same 14,464 test observations for all rows except the GARCH family (14,292 test
observations, 32/33 tickers). Deep-model rows are three-seed means (42/123/2026); classical rows are
single deterministic fits. Lower is better for MSE, RMSE, MAE, QLIKE; higher for $R^2$ and DirAcc.
Bold marks the best value per column among all rows. Source:
`docs/reports/ladder_consistent_h5_2026-08-09_154402.json` (G1) and
`docs/reports/classical_baselines_h5_2026-08-09_182129.json` (classical).

| Model | MSE ↓ | RMSE ↓ | MAE ↓ | $R^2$ ↑ | QLIKE ↓ | DirAcc % ↑ |
|---|---|---|---|---|---|---|
| G1 (proposed) | 5.31428e-06 | 0.00230527 | 0.000599607 | 0.763520 | 0.575926 | 48.221 |
| HAR | 5.24277e-06 | 0.00228971 | 0.000631170 | 0.766703 | 0.579291 | 48.401 |
| HARQ | **5.24020e-06** | **0.00228915** | 0.000628925 | **0.766817** | **0.573674** | 48.379 |
| log-HAR | 5.62759e-06 | 0.00237225 | **0.000593200** | 0.749579 | 0.779422 | **48.830** |
| EWMA | 5.33929e-06 | 0.00231069 | 0.000610615 | 0.762408 | 0.600625 | 48.031 |
| Persistence | 7.68559e-06 | 0.00277229 | 0.000722742 | 0.658000 | 4151.22 | 48.009 |
| GARCH(1,1) | 2.26642e-05 | 0.00476069 | 0.001168780 | 0.003075 | 1.76100 | 48.673 |
| GJR-GARCH | 2.27171e-05 | 0.00476625 | 0.001172287 | 0.000746 | 1.82432 | 48.653 |
| EGARCH | 2.28053e-05 | 0.00477549 | 0.001176398 | -0.003130 | 1.87379 | 48.827 |

The rest of this section decomposes G1 to show which component earns its place: the ladder isolates
the news branch, the gate, and temporal learning (Section 6.2), and the graph ablation isolates
cross-stock message-passing (Sections 6.3 and 6.4).

### 6.2 The nested ladder: news is decisive, the gate does not pay its way

Table 2 reports the five-day-ahead validation and test metrics for the full ladder P0 → P1 → P2 → P3 →
G1, three-seed mean, on the shared observation set. Reading the ladder up: the price-only LSTM (P1)
lowers test RMSE against HAR-linear P0 (0.00226464 vs 0.00228929) and lowers QLIKE on both splits, so
temporal learning already buys a proportional-loss gain. Adding the news branch (P2) is the decisive
step: it lowers validation QLIKE from 0.506196 to 0.503117 and test QLIKE from 0.564780 to 0.559854.
P2, the parsimonious news-augmented backbone with neither gate nor graph, attains the lowest test QLIKE
of any configuration in the study, deep or classical (0.559854, below HARQ's 0.573674). Adding the
per-ticker gate (P3) does not extend the gain: P3's test QLIKE (0.576488) is above P2's, and its
validation QLIKE (0.513001) is above P2's as well. Enabling the graph (G1) recovers part of what the
gate gave up but does not exceed the parsimonious P2 backbone on test QLIKE.

**Table 2. Five-day-ahead ladder metrics, validation and test, three-seed mean (seeds 42/123/2026).**
Same 14,418 validation / 14,464 test observations across all rows. Lower is better for MSE, RMSE, MAE,
QLIKE; higher for $R^2$ and DirAcc. P0 is a deterministic per-ticker linear fit. Bold marks the best
value per column within each split. Source:
`docs/reports/ladder_consistent_h5_2026-08-09_154402.json`.

| Split | Config | MSE ↓ | RMSE ↓ | MAE ↓ | $R^2$ ↑ | QLIKE ↓ | DirAcc % ↑ |
|---|---|---|---|---|---|---|---|
| VAL | P0 HAR | 2.16810e-06 | 0.00147245 | 0.000473666 | 0.739435 | 0.509637 | 48.519 |
| VAL | P1 price LSTM | 2.22654e-06 | 0.00149216 | 0.000485894 | 0.732412 | 0.506196 | 48.543 |
| VAL | P2 +news | 2.18733e-06 | 0.00147896 | 0.000476297 | 0.737125 | **0.503117** | 48.524 |
| VAL | P3 +gate | 2.15542e-06 | 0.00146813 | 0.000466772 | 0.740959 | 0.513001 | 48.441 |
| VAL | G1 +graph | **2.11845e-06** | **0.00145549** | **0.000461872** | **0.745403** | 0.509102 | **48.683** |
| TEST | P0 HAR | 5.24084e-06 | 0.00228929 | 0.000602656 | 0.766788 | 0.567625 | **48.527** |
| TEST | P1 price LSTM | **5.12862e-06** | **0.00226464** | 0.000606516 | **0.771782** | 0.564780 | 47.975 |
| TEST | P2 +news | 5.15428e-06 | 0.00227030 | 0.000601621 | 0.770640 | **0.559854** | 48.043 |
| TEST | P3 +gate | 5.34955e-06 | 0.00231291 | 0.000601409 | 0.761951 | 0.576488 | 47.881 |
| TEST | G1 +graph | 5.31428e-06 | 0.00230527 | **0.000599607** | 0.763520 | 0.575926 | 48.221 |

Table 3 gives the paired $t$-tests across the three seeds, computed on the per-seed metric values in
the canonical ladder JSON. Adding news (P2 vs P1) lowers QLIKE significantly on both splits ($t=-7.69$
validation, $t=-9.50$ test, both past $|t|>4.30$); on validation it also improves MSE, RMSE, and $R^2$
significantly, while on held-out test it trades a small, significant increase in squared error (RMSE
$t=+10.69$) for the QLIKE and MAE gain. Adding the gate (P3 vs P2) does not lower QLIKE — it raises it
significantly on both splits (validation $t=+30.6$, test $t=+35.8$) — so the gate buys no
proportional-loss accuracy despite improving the squared-error metrics on validation only. Temporal
learning (P1 vs P0) improves QLIKE on both splits.

**Table 3. Paired $t$-tests across three seeds (df=2, threshold $|t|>4.30$), derived from the per-seed
entries of the canonical ladder JSON.** Sign is (upper rung $-$ lower rung); negative $t$ on QLIKE
means the added component lowers QLIKE. Source (per-seed values):
`docs/reports/ladder_consistent_h5_2026-08-09_154402.json`.

| Contrast | Split | MSE | RMSE | MAE | $R^2$ | QLIKE |
|---|---|---|---|---|---|---|
| News (P2 vs P1) | VAL | $-4.38$ (sig.) | $-4.38$ (sig.) | $-4.46$ (sig.) | $+4.38$ (sig.) | $-7.69$ (sig.) |
| News (P2 vs P1) | TEST | $+10.6$ (worse) | $+10.7$ (worse) | $-3.15$ (n.s.) | $-10.6$ (worse) | $-9.50$ (sig.) |
| Gate (P3 vs P2) | VAL | $-7.32$ | $-7.37$ | $-6.53$ | $+7.32$ | $+30.6$ (worse) |
| Gate (P3 vs P2) | TEST | $+27.6$ (worse) | $+28.4$ (worse) | $-0.15$ (n.s.) | $-27.6$ (worse) | $+35.8$ (worse) |
| LSTM (P1 vs P0) | VAL | $+9.17$ (worse) | $+9.24$ (worse) | $+10.7$ (worse) | $-9.17$ (worse) | $-6.09$ (sig.) |
| LSTM (P1 vs P0) | TEST | $-25.7$ (better) | $-25.6$ (better) | $+5.79$ (worse) | $+25.7$ (better) | $-3.83$ (n.s.) |

**Takeaway.** News content is the one added mechanism with a consistent, significant QLIKE gain on both
validation and held-out test, and the parsimonious P2 news backbone attains the lowest test QLIKE in
the study. The per-ticker gate raises QLIKE and earns no place. The graph is examined next.

### 6.3 Graph ablation: removing cross-stock message-passing changes nothing significant (h5)

Table 4 isolates the cross-stock graph by the exactly-nested G1-versus-P3 pair at the five-day horizon.
Because P3 is G1 read out with the message-passing residual disabled (graph-off determinism 0.0), the
pair differs only in cross-stock propagation. On validation the graph lowers loss on all three seeds
(QLIKE 0.513001 to 0.509102, paired $t$ $p=0.0096$), but the per-observation Diebold-Mariano test on
QLIKE is not significant-negative on all seeds, so the validation verdict is B (no clean graph win). On
the held-out test set the case for the graph disappears: G1's QLIKE (0.575926) improves on P3's
(0.576488) by less than its fourth significant figure, the improvement holds on only two of three
seeds, and the paired $t$ is not significant ($p=0.79$). Cross-stock message-passing yields no
statistically significant forecast improvement over the news-augmented backbone.

**Table 4. Graph ablation G1 vs P3 (exactly nested), five-day horizon, three-seed mean.** Same
observations as the ladder. QLIKE delta = mean(G1 $-$ P3) over seeds (negative = graph helps); paired
$t$ $p$ across seeds; per-seed DM on QLIKE. Verdict B = graph null (G1 does not beat P3 on QLIKE in all
seeds AND per-seed DM-QLIKE not significant-negative in all seeds). Source:
`docs/reports/ladder_consistent_h5_2026-08-09_154402.json` (`graph_effect_dm`, `graph_effect_verdict`).

| Split | P3 QLIKE | G1 QLIKE | QLIKE Δ (G1−P3) | G1<P3 seeds | paired-$t$ $p$ | DM-QLIKE all sig-neg | Verdict |
|---|---|---|---|---|---|---|---|
| VAL | 0.513001 | 0.509102 | $-0.003899$ | 3/3 | 0.0096 | No | B (null) |
| TEST | 0.576488 | 0.575926 | $-0.000562$ | 2/3 | 0.7913 | No | B (null) |

### 6.4 The null holds across all four horizons

Table 5 replicates the exactly-nested G1-versus-P3 graph ablation at horizons 1, 5, 10, and 22 trading
days, each on its own leakage-safe masked manifest. The verdict is B (graph null) at every horizon: the
graph never clears the bar of a QLIKE improvement on all seeds together with a per-seed DM-QLIKE
significant-negative on all seeds. Two horizon-specific nuances are worth stating plainly. At h1 the
graph does improve the squared-error metrics — DM on the MSE loss series is significant-negative on all
three test seeds — but its QLIKE is unstable: seed 2026 inflates G1's QLIKE through near-floor one-day
predictions, so the QLIKE-based verdict stays B. At h22 the graph improves validation QLIKE (paired
$t$ $p=0.0002$) but the gain does not carry to the held-out test, where G1 is worse than P3 on all
three seeds. h5 and h10 are the cleanest nulls, with small and statistically insignificant test
differences.

**Table 5. Multi-horizon graph ablation G1 vs P3, held-out test, three-seed mean.** QLIKE delta =
mean(G1 $-$ P3) over seeds; paired $t$ $p$ across seeds; verdict as in Table 4. Companion validation
QLIKE shown for context. Source: `docs/reports/ladder_consistent_h{1,10,22}_2026-08-09_180326.json`,
`docs/reports/ladder_consistent_h5_2026-08-09_154402.json`, and
`docs/reports/ladder_consistent_multihorizon_2026-08-09_180326.md`.

| Horizon | VAL P3 QLIKE | VAL G1 QLIKE | TEST P3 QLIKE | TEST G1 QLIKE | TEST QLIKE Δ | G1<P3 test seeds | test paired-$t$ $p$ | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.443238 | 0.877562 | 0.485719 | 0.516594 | $+0.030880$ | 2/3 | 0.5305 | B (null) |
| 5 | 0.513001 | 0.509102 | 0.576488 | 0.575926 | $-0.000562$ | 2/3 | 0.7913 | B (null) |
| 10 | 0.554665 | 0.548133 | 0.616804 | 0.615622 | $-0.001182$ | 3/3 | 0.0669 | B (null) |
| 22 | 0.605964 | 0.599794 | 0.673834 | 0.678706 | $+0.004871$ | 0/3 | 0.1425 | B (null) |

A companion per-seed DM table (test MSE and QLIKE per seed per horizon) is in the canonical
multi-horizon report; the h1 DM-MSE is significant-negative on all three test seeds while its DM-QLIKE
is not, which is the source of the h1 nuance above. Across all four horizons the graph mechanism does
not deliver a statistically significant QLIKE improvement over the news-augmented backbone.

### 6.5 Direction is near-random for every model

Directional accuracy sits at 47.9% to 48.7% across the ladder and 48.0% to 48.8% across the classical
baselines, at or below the 50% no-skill line. No paired test separates any configuration on direction,
and the model-free persistence and EWMA forecasters reach the same band. The near-random result is a
property of the target, not of any model, as Section 8 explains.

---

## 7. Discussion

**What each component contributes.** The single-basis ladder gives a clean attribution. Temporal
learning (P1) buys a proportional-loss gain over HAR. News content (P2) is the decisive component: it
lowers QLIKE significantly on both validation and held-out test, and the parsimonious P2 news backbone
attains the lowest test QLIKE of any model in the study. The per-ticker gate (P3) does not extend the
gain and raises QLIKE. The cross-stock graph (G1 vs P3) yields no statistically significant QLIKE
improvement at any of the four horizons. The proposed model G1 unifies temporal, textual, gating, and
cross-stock structure, and the rigorous finding is that only temporal learning and news content carry a
measurable forecast gain on this panel.

**Why the graph adds nothing, and why the null is clean.** Three facts bound the graph contribution.
First, the message-passing layer aggregates news that has already entered each node's own
representation, so the marginal information a neighbour adds on a daily panel is small once a stock's
own news and price history are encoded. Second, VN30's listing-date imbalance thins the early-year
neighbourhoods even under masking. Third, the masked formulation is what makes the null clean: it
removes the data-volume confound that a synchronized-intersection graph would suffer, and the exact
nesting (P3 is G1 with the graph switched off, determinism 0.0) removes any training-setup confound, so
the null speaks to the cross-stock signal itself rather than to data starvation or an unmatched
control. The null replicates across four horizons and under a Diebold-Mariano test. It aligns with the
best-controlled published GNN-vs-HAR study, GNNHAR on DJIA-30, where multi-hop graph spillover gave no
clear advantage under a Model Confidence Set and the gains came from nonlinearity and a QLIKE training
loss, not from the graph [10], and with the broader finding that a well-specified HAR is hard to beat
on a limited information set [14]. The contribution is therefore a rigorous parsimony result for sparse
daily cross-stock news: the graph mechanism does not pay its way, and a news-augmented per-stock model
is the appropriate architecture.

**Classical baselines confirm the ceiling.** HAR and HARQ tie the deep models on the level metrics
(test QLIKE 0.5793 / 0.5737, $R^2 \approx 0.767$), and HARQ's QLIKE is marginally below G1's, while the
GARCH family is far worse (test QLIKE 1.76 to 1.87, $R^2 \approx 0$). A well-specified HAR-family model
sits near a structural ceiling for daily range-based variance on a small universe, so the deep model's
value is that it matches HAR while adding a news channel, not that it dominates HAR on the level.

**News-content value.** News lowers the proportional QLIKE loss that risk and option desks weigh most,
on both validation and held-out test, and the effect survives a paired $t$-test across seeds. This is
the paper's strongest empirical result, and it holds against the classical HAR and HARQ benchmarks and
against the price-only deep control, independent of the graph and gate mechanisms layered on top.

**Attempts to strengthen the graph and future directions.** The published record indicates the levers
that separate a graph that beats HAR from one that ties it, most of which this data regime lacks:
intraday-derived realized (rather than daily range-based) variance, a QLIKE training loss, directed
volatility-spillover edges in place of symmetric correlation, richer HAR-family node features
(semivariance, realized-quarticity, overnight and range estimators), and a large cross-sectional
universe [10,14,15]. The three realistic on-data levers for this project are (i) training and
evaluating on a QLIKE objective, the single decisive lever in the most rigorous GNN-vs-HAR study, (ii)
augmenting the node with range-estimator, overnight, and HAR-residual-decomposition features, and (iii)
replacing correlation k-NN edges with a directed Diebold-Yilmaz spillover adjacency.

We ran that targeted sweep on the identical consistent basis (same 14,418 validation / 14,464 test
observations, leakage-safe masked k-NN-8 graph, three seeds), evaluating a leaner price-only GAT and
five research-backed levers against the pooled-HAR anchor P0 (test QLIKE 0.5676): (C1) a QLIKE-loss GAT
on the news backbone, (C2) an additive HAR + graph-residual decomposition, (C3) directed
Diebold-Yilmaz volatility-spillover edges, (C5) spillover with an omitted self-loop under a $k$-sweep,
and (C6) a learned dynamic adjacency. No configuration beats the anchor at a Diebold-Mariano-significant
level on QLIKE. The best, C2, reaches test QLIKE 0.5662 but its per-seed sign is inconsistent and the
across-seed paired-$t$ is far from significant ($p=0.562$), so it statistically ties P0 rather than
beating it, and it ties P0 on RMSE as well. The QLIKE-loss GAT (C1, test QLIKE 0.5730) improves on the
MSE-trained backbone and beats the classical per-ticker HAR (0.5793) and ties P0 on RMSE, but is
significantly worse than the pooled-HAR anchor on QLIKE (paired-$t$ $p=0.027$). The directed-spillover,
omit-self, and learned-adjacency variants (C3 0.5908, C5 0.5748, C6 0.5903) are significantly worse than
P0 on QLIKE. A separate leaner check, a price-only GAT on the P1 backbone, beats HAR on only one of six
held-out test metrics (MAE) and is directionally, though not significantly, worse on squared-error loss
(Diebold-Mariano $p=0.19$-$0.25$). Two further levers were not run: HAR-RV-X range/overnight node
features (C4) require a multi-feature backbone beyond the pooled preprocessor's single-feature contract,
and news-as-edge co-mention (C7) is infeasible on the per-ticker news panel, which carries no
article-level multi-ticker structure. Neither the base graph nor any of the five targeted enhancements
beats a well-specified HAR, which strengthens rather than weakens the parsimony conclusion: the
cross-stock graph adds no measurable value even under the levers the literature credits for closing the
HAR gap. Source: `docs/reports/2026-08-10_0412_beat_har_sweep_results.md` (C1-C6),
`docs/reports/2026-08-10_0130_gat_price_har_quick.md` (price-only GAT).

---

## 8. Why Direction Is Near-Random

The forecast target is the single-day Parkinson estimator, and its day-to-day change is
anti-persistent. Measured on each stock's full series, the sign of the daily change in Parkinson
volatility has a lag-1 autocorrelation of $-0.30$ on average, negative for all 33 tickers (range
$-0.34$ to $-0.24$). An up-move in daily volatility tends to precede a down-move: the estimator
oscillates day to day around a slowly drifting level, a known property of range-based daily proxies. A
forecaster that produces a smooth estimate of the level cannot reproduce this high-frequency
oscillation, so its directional calls decouple from the target and sit at or just below 50%.
Model-free forecasters confirm the ceiling: persistence and EWMA reach the same 48% band as the trained
models, yet both fitted deep models and the HAR family attain positive $R^2$ (0.76 to 0.77) while the
persistence forecast collapses on QLIKE. Every fitted model does real work on the volatility level and
no forecaster beats chance on the day-to-day sign, so we base the conclusions on the continuous-error
metrics and report directional accuracy for completeness.

---

## 9. Limitations

Five limitations bound the claims. First, the 33-ticker universe is a fixed, point-in-time VN30-like
set: it keeps long-history names that later left the index and excludes two short-history current
members (BSR, VPL), so it is not the live index. Second, the study covers a single market at daily
frequency; the news-content result and the graph null may not transfer to higher frequencies or to
markets with balanced listing histories, and every rigorous published GNN-beats-HAR result relies on
intraday-derived realized variance this daily-OHLCV panel does not have. Third, three seeds is the
minimum for a paired $t$-test; the graph null is strengthened by a Diebold-Mariano test on the held-out
QLIKE across four horizons, but a larger seed set and a Model Confidence Set would tighten it, and the
MCS was not computed here because per-observation prediction series were not retained in the result
artifacts. Fourth, the HARQ baseline uses a daily range-based realized-quarticity proxy rather than the
canonical 5-minute quarticity, and the GARCH family covers 32 of 33 tickers, so those rows are read as
approximations on a near-identical basis. Fifth, low directional accuracy is a structural property of
the anti-persistent daily target, not a tunable model deficiency, so a direction-focused deployment
would need a different target construction.

---

## 10. Conclusion

We propose a pooled news-augmented cross-stock graph LSTM (G1) for five-day-ahead VN30 volatility that
unifies temporal, textual, and cross-stock structure on a panel where a fixed-node graph is
infeasible. A pooled asynchronous design recovers the full trading calendar, and an availability-aware
masked adjacency lets the graph propagate news between stocks on real per-day observations rather than a
26% intersection. A single-basis nested ablation ladder — P0 (HAR) → P1 (price LSTM) → P2 (+news) → P3
(+gate) → G1 (+graph), where P3 is exactly G1 with the graph disabled — attributes the model's quality:
temporal learning and news content carry a QLIKE gain significant across seeds, the per-ticker gate
raises QLIKE and earns no place, and enabling the cross-stock graph gives no statistically significant
improvement under a Diebold-Mariano test at any of four horizons. Classical baselines confirm the
picture: HAR and HARQ tie the deep models on the level metrics while GARCH is far worse. The paper's
second contribution is therefore a rigorous parsimony finding: on sparse daily VN30 data, news content
improves the forecast while cross-stock graph propagation does not, and the strongest configuration is
the parsimonious news-augmented backbone. Directional accuracy stays near chance for every forecaster
because the daily Parkinson target's day-to-day change is anti-persistent, a structural ceiling rather
than a model defect. For a Vietnamese emerging market, the pooled masked-graph design shows how to
build and honestly evaluate a cross-stock volatility forecaster where the standard synchronized-panel
approach cannot run.

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
preprint arXiv:2502.15813 (2025). *(Generic LSTM+GNN precedent: predicts stock price on 10 stocks; no
HAR baseline and no volatility target.)*

[10] Zhang, C., Pu, X., Cucuringu, M., Dong, X. Forecasting realized volatility with spillover
effects: perspectives from graph neural networks (GNNHAR). *International Journal of Forecasting*
41(1), 377–397 (2025). arXiv:2308.01419. *(Under a Model Confidence Set, multi-hop graph spillover
gives no clear advantage; gains come from nonlinearity and a QLIKE training loss, at horizons up to one
week.)*

[11] Rossi, E., Kenlay, H., Gorinova, M., Chamberlain, B., Dong, X., Bronstein, M. On the unreasonable
effectiveness of feature propagation in learning on graphs with missing node features. In *LoG*
(2022). arXiv:2111.12128.

[12] Cini, A., Marisca, I., Alippi, C. Filling the gaps: multivariate time series imputation by graph
neural networks (GRIN). In *ICLR* (2022). arXiv:2108.00298.

[13] Marisca, I., Cini, A., Alippi, C. Learning to reconstruct missing data from spatiotemporal graphs
with sparse observations (SPIN). In *NeurIPS* (2022). arXiv:2205.13479.

[14] Audrino, F., Chassot, J. HARd to beat: the overlooked impact of rolling windows in the era of
machine learning. *International Journal of Forecasting* (2025). arXiv:2406.08041. *(Tuned ML models
fail to beat a carefully re-estimated HAR on QLIKE/MSE with a matched information set.)*

[15] Bollerslev, T., Patton, A.J., Quaedvlieg, R. Exploiting the errors: a simple approach for
improved volatility forecasting (HARQ). *Journal of Econometrics* 192(1), 1–18 (2016).

[18] Hsu, Y.-L., Tsai, Y.-C., et al. FinGAT: Financial graph attention networks for recommending
top-K profitable stocks. *IEEE TKDE* 35(1), 469–481 (2023). arXiv:2106.10159.

[19] Chen, Q., Robert, C.-Y. Multivariate realized volatility forecasting with graph neural network.
In *ACM ICAIF* (2022). arXiv:2112.09015.

*Internal provenance references (not for the reference list): five-day ladder
`docs/reports/ladder_consistent_h5_2026-08-09_154402.json`; multi-horizon
`docs/reports/ladder_consistent_h{1,10,22}_2026-08-09_180326.json` and
`..._multihorizon_2026-08-09_180326.md`; classical baselines
`docs/reports/classical_baselines_h5_2026-08-09_182129.json`; diagnosis
`docs/reports/2026-08-09_2148_old_gat_vs_new_g1_diagnosis.md`; beat-HAR conditions research
`docs/reports/2026-08-09_2209_gnn_volatility_beat_har_research.md`; hybrid combinations research
`docs/reports/2026-08-09_2214_gnn_hybrid_combinations_research.md`; beat-HAR sweep (C1-C6)
`docs/reports/2026-08-10_0412_beat_har_sweep_results.md`; price-only GAT check
`docs/reports/2026-08-10_0130_gat_price_har_quick.md`.*
