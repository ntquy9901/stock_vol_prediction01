# EDA and Dynamic GNN Graph Construction Plan for Daily Stock Parkinson Volatility

## 1. Objective

This document defines a complete exploratory data analysis (EDA) and graph-construction experiment for a stock-volatility forecasting system using **daily OHLCV data only**.

The purpose is to determine, empirically, whether cross-stock relationships are strong, stable, and predictive enough to justify using a Graph Neural Network (GNN), especially a Dynamic GAT or edge-aware GAT.

The experiments must answer the following questions:

1. Do stocks exhibit meaningful cross-sectional relationships in daily returns?
2. Do stocks exhibit meaningful relationships in daily Parkinson volatility?
3. Do abnormal trading-volume changes co-move across stocks?
4. Are there directional lead-lag relationships between stocks?
5. Are Parkinson-volatility relationships stronger or more useful than return relationships for volatility forecasting?
6. Are graph relationships stable through time?
7. Does sector information explain graph structure?
8. Is a dynamic graph justified compared with a static graph?
9. Which edge attributes should be retained?
10. Does graph structure contain predictive information beyond a stock-only model and a market-factor baseline?

The final goal is **not** to force a GNN into the system. The goal is to determine whether the data provides sufficient evidence to justify a GNN.

---

# 2. Important Data Constraint

The dataset contains **daily OHLCV data only**.

Available fields are expected to include:

- Date
- Ticker
- Open
- High
- Low
- Close
- Volume

There is **no intraday data**.

Therefore:

- Do NOT calculate realized volatility from intraday returns.
- Do NOT refer to intraday volatility.
- Do NOT use realized variance based on high-frequency observations.
- Do NOT describe effects as intraday volatility spillovers.

The volatility measure used throughout this experiment is **daily Parkinson volatility**.

---

# 3. Daily Parkinson Volatility

For stock \(i\) on day \(t\), define Parkinson variance as:

\[
PKVar_{i,t}
=
\frac{1}{4\ln(2)}
\left[
\ln\left(\frac{High_{i,t}}{Low_{i,t}}\right)
\right]^2
\]

Daily Parkinson volatility is:

\[
PK_{i,t}
=
\sqrt{
\frac{1}{4\ln(2)}
\left[
\ln\left(\frac{High_{i,t}}{Low_{i,t}}\right)
\right]^2
}
\]

The implementation should explicitly state whether the forecasting target is:

- Parkinson variance (`PKVar`), or
- Parkinson volatility (`PK`).

Do not silently mix the two.

Recommended default for this EDA:

```text
pk_vol = sqrt((log(high / low) ** 2) / (4 * log(2)))
```

Also retain:

```text
pk_var = (log(high / low) ** 2) / (4 * log(2))
```

for sensitivity analysis.

---

# 4. Core Daily Features

For every stock \(i\) and date \(t\), calculate the following.

## 4.1 Daily log return

\[
r_{i,t}
=
\ln\left(
\frac{Close_{i,t}}
{Close_{i,t-1}}
\right)
\]

Feature:

```text
log_return_1d
```

Also create optional cumulative returns:

```text
return_5d
return_10d
return_20d
```

using past data only.

---

## 4.2 Parkinson volatility

Create:

```text
pk_vol
pk_var
```

Optional lagged/rolling features:

```text
pk_vol_lag1
pk_vol_lag2
pk_vol_lag5

pk_mean_5
pk_mean_10
pk_mean_20

pk_std_5
pk_std_20
```

These must use information available at or before date \(t\).

---

## 4.3 Volume transformations

Do not compare raw trading volume across different stocks directly because the scales can differ substantially.

Create:

### Log volume

\[
LV_{i,t} = \ln(1 + Volume_{i,t})
\]

Feature:

```text
log_volume
```

### Volume change

\[
\Delta LV_{i,t}
=
LV_{i,t} - LV_{i,t-1}
\]

Feature:

```text
log_volume_change
```

### Rolling volume z-score

For a 20-day historical window:

\[
VolumeShock_{i,t}
=
\frac{
LV_{i,t} - \mu_{i,t}^{20}
}{
\sigma_{i,t}^{20}
}
\]

Feature:

```text
volume_zscore_20
```

The rolling mean and standard deviation must be constructed without future observations.

Optional:

```text
volume_zscore_5
volume_zscore_60
```

---

# 5. Initial Data Quality EDA

Before graph analysis, perform data-quality checks.

For every ticker report:

```text
start_date
end_date
number_of_rows
missing_dates
missing_open
missing_high
missing_low
missing_close
missing_volume
zero_volume_count
duplicate_date_count
high_below_low_count
nonpositive_price_count
```

Check:

```text
High >= max(Open, Close, Low)
Low <= min(Open, Close, High)
High > 0
Low > 0
Close > 0
Volume >= 0
```

Report suspicious observations.

Do not automatically delete outliers without showing evidence.

---

# 6. Alignment Across Stocks

Graph construction requires observations for multiple stocks on the same trading date.

Create a common trading-date panel.

Recommended representation:

```text
date × ticker × features
```

or a wide matrix for each feature:

```text
date × ticker
```

Example:

```text
date        VCB      BID      CTG      FPT
2025-01-02  ...
2025-01-03  ...
```

Report:

- number of common dates across all stocks;
- percentage of missing ticker-date cells;
- stocks with unusual listing gaps;
- whether missing values are filled or observations are excluded.

Preferred default:

Do not forward-fill prices through non-trading observations unless there is a documented reason.

---

# 7. Cross-Stock EDA

Assume there are \(N\) stocks.

For 30 VN30 stocks, the number of unique undirected stock pairs is:

\[
\frac{30 \times 29}{2} = 435
\]

Analyze all available pairs.

---

# 8. Return Correlation

For every stock pair \(i,j\), calculate:

\[
\rho^r_{ij}
=
Corr(r_i, r_j)
\]

Calculate both:

```text
Pearson correlation
Spearman correlation
```

Produce:

```text
return_corr_pearson
return_corr_spearman
```

Generate:

1. Full-period correlation matrix.
2. Heatmap.
3. Distribution of pairwise correlations.
4. Top 20 strongest positive pairs.
5. Top 20 strongest negative pairs.
6. Weakest absolute-correlation pairs.
7. Correlation grouped by same-sector vs different-sector pairs.

Do not interpret correlation as causation.

---

# 9. Parkinson-Volatility Correlation

This is a primary EDA for the volatility-forecasting problem.

For every pair \(i,j\):

\[
\rho^{PK}_{ij}
=
Corr(PK_i, PK_j)
\]

Calculate:

```text
pk_corr_pearson
pk_corr_spearman
```

Produce:

1. Full-period Parkinson-volatility correlation matrix.
2. Heatmap.
3. Distribution across all stock pairs.
4. Strongest Parkinson-volatility pairs.
5. Weakest pairs.
6. Same-sector versus cross-sector comparison.
7. Comparison with return correlation.

Important question:

```text
Is PK-volatility correlation systematically stronger than return correlation?
```

Compare:

\[
|\rho^{PK}_{ij}|
\]

against:

\[
|\rho^{r}_{ij}|
\]

for every pair.

Produce a scatter plot:

```text
x = abs(return correlation)
y = abs(Parkinson-volatility correlation)
```

Calculate summary statistics of the difference:

\[
D_{ij}
=
|\rho^{PK}_{ij}|
-
|\rho^r_{ij}|
\]

---

# 10. Volume Relationship EDA

For every pair calculate correlations using transformed volume variables.

Recommended:

\[
Corr(VolumeShock_i, VolumeShock_j)
\]

and:

\[
Corr(\Delta LV_i,\Delta LV_j)
\]

Create:

```text
volume_zscore_corr
volume_change_corr
```

Analyze:

- same-sector relationships;
- cross-sector relationships;
- whether high Parkinson-volatility-correlation pairs also have high volume correlation.

---

# 11. Contemporaneous Cross-Feature Relationships

Analyze relationships beyond same-feature correlation.

For every ordered pair \(i \rightarrow j\), calculate:

\[
Corr(VolumeShock_{i,t}, PK_{j,t})
\]

\[
Corr(r_{i,t}, PK_{j,t})
\]

These relationships should mainly be treated as descriptive EDA.

They are not necessarily valid directional predictive edges because the variables occur on the same day.

---

# 12. Lead-Lag Analysis

Lead-lag analysis is critical for deciding whether edges should be directed.

For every ordered pair \(i \rightarrow j\), calculate lagged relationships.

## 12.1 Parkinson-volatility lead-lag

For lag \(k\):

\[
L^{PK}_{ij}(k)
=
Corr(PK_{i,t}, PK_{j,t+k})
\]

Test:

```text
k = 1
k = 2
k = 5
k = 10
```

Primary focus:

```text
PK_i(t) -> PK_j(t+1)
PK_i(t) -> PK_j(t+5)
```

These relationships are directly relevant to T+1 and T+5 volatility forecasting.

---

# 13. Return Lead-Lag

Calculate:

\[
L^{r}_{ij}(k)
=
Corr(r_{i,t}, r_{j,t+k})
\]

for:

```text
k = 1
k = 2
k = 5
```

Use this mainly as supplementary edge information.

---

# 14. Volume-to-Future-Volatility Lead-Lag

Test whether abnormal activity in stock \(i\) contains information about future volatility of stock \(j\):

\[
L^{V \rightarrow PK}_{ij}(k)
=
Corr(
VolumeShock_{i,t},
PK_{j,t+k}
)
\]

Evaluate:

```text
k = 1
k = 2
k = 5
```

This can become an edge feature such as:

```text
volume_to_pk_lag1
volume_to_pk_lag5
```

---

# 15. Reverse-Direction Comparison

For each directional relationship compare:

\[
i \rightarrow j
\]

with:

\[
j \rightarrow i
\]

Example:

```text
Corr(PK_VCB(t), PK_BID(t+1))
```

versus:

```text
Corr(PK_BID(t), PK_VCB(t+1))
```

Define directional asymmetry:

\[
A_{ij}
=
L_{ij}(1) - L_{ji}(1)
\]

Large stable asymmetry supports the use of directed edges.

---

# 16. Statistical Significance

For pairwise correlation analysis, provide:

```text
correlation
p-value
number_of_observations
```

However, because hundreds of stock pairs are tested, raw p-values are insufficient.

Apply a multiple-testing correction.

Recommended:

```text
Benjamini-Hochberg False Discovery Rate
```

Report:

```text
raw_p
fdr_q
significant_at_0.05
```

Do not select edges merely because raw `p < 0.05`.

---

# 17. Permutation / Null Test

Correlation thresholds such as `0.3` or `0.5` should not be accepted blindly.

Construct a null distribution.

For example:

1. Randomly permute the dates of stock \(j\).
2. Recalculate pairwise correlation.
3. Repeat 500-1000 times.
4. Compare observed correlation to the null distribution.

Perform this especially for:

```text
Parkinson-volatility correlation
PK lead-lag correlation
volume -> PK lead-lag correlation
```

This provides evidence that graph relationships are stronger than random temporal alignment.

---

# 18. Rolling Correlation and Dynamic Graph EDA

Financial relationships change through time.

Therefore calculate rolling correlations.

Recommended windows:

```text
20 trading days
60 trading days
120 trading days
```

For every date \(t\):

\[
\rho^{PK}_{ij,t,W}
=
Corr(
PK_{i,t-W+1:t},
PK_{j,t-W+1:t}
)
\]

Similarly calculate:

```text
return_corr_20
return_corr_60
return_corr_120

pk_corr_20
pk_corr_60
pk_corr_120

volume_corr_20
volume_corr_60
volume_corr_120
```

---

# 19. Edge Stability

A useful edge should not exist only because of one short accidental period.

For every pair define stability metrics.

## 19.1 Mean rolling correlation

\[
MeanCorr_{ij}
=
mean_t(\rho_{ij,t})
\]

## 19.2 Standard deviation

\[
StdCorr_{ij}
=
std_t(\rho_{ij,t})
\]

## 19.3 Sign consistency

\[
SignConsistency_{ij}
=
\frac{
\#\{t : sign(\rho_{ij,t}) = dominant\ sign\}
}{
T
}
\]

## 19.4 Threshold persistence

For threshold \(\tau\):

\[
Persistence_{ij}
=
\frac{
\#\{t : |\rho_{ij,t}| > \tau\}
}{
T
}
\]

Evaluate thresholds such as:

```text
0.2
0.3
0.4
0.5
```

But treat them as diagnostic thresholds, not universal truths.

---

# 20. Dynamic vs Static Graph Decision

Compare graph stability.

If edge weights change materially across time, prefer:

```text
Dynamic Graph
```

If relationships are highly stable, a static graph may be sufficient.

Produce plots for representative pairs showing:

```text
rolling PK correlation through time
rolling return correlation through time
rolling volume correlation through time
```

Include:

- strongest stable pair;
- unstable pair;
- same-sector pair;
- cross-sector pair.

---

# 21. Market-Regime Analysis

Relationships can become stronger during high-volatility market regimes.

Create a market-wide daily Parkinson-volatility indicator.

For example:

\[
MarketPK_t
=
median_i(PK_{i,t})
\]

or:

\[
MarketPK_t
=
mean_i(PK_{i,t})
\]

Define regimes based only on historical or training-period thresholds.

Example:

```text
Low volatility
Normal volatility
High volatility
```

Possible thresholds:

```text
bottom 33%
middle 34%
top 33%
```

or train-period quantiles.

Compare pairwise PK correlations by regime.

Question:

```text
Do cross-stock PK-volatility correlations become stronger in high-volatility regimes?
```

If yes, a dynamic graph is more strongly justified.

---

# 22. Sector Analysis

If sector metadata is available, create:

```text
same_sector = 1
different_sector = 0
```

For each relationship type compare:

```text
same-sector mean correlation
cross-sector mean correlation
same-sector median correlation
cross-sector median correlation
```

Perform tests such as:

```text
Mann-Whitney U
```

if distributional assumptions for a t-test are questionable.

Produce box plots for:

```text
return correlation by sector relationship
PK-volatility correlation by sector relationship
volume correlation by sector relationship
```

---

# 23. Graph Clustering EDA

Build preliminary graphs using Parkinson-volatility correlation.

For example:

```text
nodes = stocks
edge weight = abs(pk_corr)
```

Try:

```text
threshold graph
top-K graph
```

Analyze whether known economic groups emerge.

Possible metrics:

```text
community structure
modularity
degree distribution
connected components
clustering coefficient
```

Do not claim sector discovery unless it is visible and measurable.

---

# 24. Candidate Graph A: Return Graph

Define:

```text
nodes = stocks
edge weight = rolling return correlation
```

Example:

\[
w^A_{ij,t}
=
\rho^r_{ij,t,60}
\]

Possible construction:

```text
Top-K neighbors per node
```

Recommended K search:

```text
K = 3
K = 5
K = 8
K = 10
```

This graph should mostly serve as a comparison or ablation for a volatility forecasting task.

---

# 25. Candidate Graph B: Parkinson-Volatility Graph

This is the primary graph candidate.

Define:

\[
w^B_{ij,t}
=
\rho^{PK}_{ij,t,60}
\]

or:

\[
w^B_{ij,t}
=
|\rho^{PK}_{ij,t,60}|
\]

Do not choose signed versus absolute correlation without testing both.

Experiments:

```text
signed PK correlation
absolute PK correlation
positive-only PK correlation
```

Recommended top-K:

```text
K = 3, 5, 8, 10
```

---

# 26. Candidate Graph C: Directed Lead-Lag PK Graph

Create a directed edge:

```text
i -> j
```

when historical Parkinson volatility of stock \(i\) is associated with future Parkinson volatility of stock \(j\).

Example edge score:

\[
w^{lead}_{ij,t}
=
Corr(
PK_{i,\tau},
PK_{j,\tau+1}
)
\]

calculated using only historical observations inside a rolling window ending before or at prediction time.

This graph is directed:

```text
edge_index:
i -> j
```

is distinct from:

```text
j -> i
```

---

# 27. Candidate Graph D: Multi-Attribute Graph

Recommended edge feature vector:

\[
e_{ij,t}
=
[
\rho^{PK}_{ij,20},
\rho^{PK}_{ij,60},
\rho^r_{ij,20},
\rho^r_{ij,60},
\rho^V_{ij,20},
L^{PK}_{ij}(1),
L^{PK}_{ij}(5),
L^{V\rightarrow PK}_{ij}(1),
sameSector_{ij}
]
\]

Suggested implementation:

```text
edge_attr = [
    pk_corr_20,
    pk_corr_60,
    return_corr_20,
    return_corr_60,
    volume_corr_20,
    pk_leadlag_1,
    pk_leadlag_5,
    volume_to_pk_lag1,
    same_sector
]
```

Optional additional features:

```text
edge_stability
edge_persistence
directional_asymmetry
```

---

# 28. Node Features

A graph edge should describe relationships between stocks.

A node should describe the current state of one stock.

Possible node features:

```text
log_return_1d
return_5d
return_10d
return_20d

pk_vol
pk_vol_lag1
pk_mean_5
pk_mean_20
pk_std_20

log_volume
log_volume_change
volume_zscore_20

high_low_log_range
open_close_return
```

If HAR-style Parkinson features are already used, include:

```text
PK_daily
PK_weekly
PK_monthly
```

where weekly/monthly values are trailing historical aggregates.

---

# 29. Avoid Raw Price Correlation as the Main Edge

Do not use:

```text
Corr(Close_i, Close_j)
```

as the main graph relationship.

Price levels are often non-stationary.

High correlation between price levels can occur without meaningful predictive dependence.

Preferred quantities are:

```text
returns
Parkinson volatility
normalized volume changes
lagged relationships
```

Raw-price correlation may be shown only as an educational or diagnostic comparison.

---

# 30. Graph Sparsification

A complete graph with 30 nodes contains:

```text
435 undirected pairs
870 directed edges
```

This is computationally manageable, but a fully connected graph may inject noise.

Compare the following strategies.

## 30.1 Fully connected graph

Use all possible edges.

Purpose:

```text
baseline
```

## 30.2 Threshold graph

Keep edge if:

\[
|w_{ij}| > \tau
\]

Try:

```text
tau = 0.2
0.3
0.4
0.5
```

## 30.3 Top-K graph

For every node keep K strongest neighbors.

Try:

```text
K = 3
5
8
10
```

This is recommended as a primary experiment.

## 30.4 Statistically filtered graph

Keep edges satisfying:

```text
FDR q < 0.05
```

plus a minimum effect size.

## 30.5 Stable-edge graph

Keep edges satisfying both:

```text
minimum average relationship strength
minimum persistence/stability
```

---

# 31. Preventing Look-Ahead Leakage

This is mandatory.

Suppose the model predicts at date \(t\).

Every graph feature must use information available at or before \(t\).

Never calculate a full-dataset correlation matrix and use it for historical training observations.

Incorrect:

```text
calculate correlation using 2025-2026 data
use same correlation matrix for samples in early 2025
```

This leaks future information.

Correct:

For graph snapshot \(G_t\):

```text
use observations from t-W+1 through t
```

only.

For directed lead-lag edges intended for a prediction sample at \(t\), the historical estimate must be learned from previously completed pairs.

Example for lag 1:

```text
X dates: <= t-1
Y dates: <= t
```

so no observation after prediction time enters edge estimation.

Every implementation must explicitly validate the date boundaries.

---

# 32. Train/Validation/Test Leakage

Split data chronologically.

Example:

```text
train -> earliest period
validation -> later period
test -> latest period
```

Do not randomly shuffle dates.

Any normalization must be fitted on training data only.

This includes:

```text
means
standard deviations
quantile thresholds
sector statistics
edge-score scaling
market-regime thresholds
```

Validation and test data must not influence training transformations.

---

# 33. EDA for Predictive Value

Correlation alone does not prove that a GNN will improve forecasting.

Perform simple predictive tests before implementing a complex GNN.

For each stock \(j\), compare:

### Baseline A

Use only own-stock historical features:

```text
PK_j(t)
returns_j(t)
volume_j(t)
```

### Baseline B

Add market-wide factor:

```text
market Parkinson volatility
market return
market volume shock
```

### Baseline C

Add top correlated neighbor features:

```text
PK_neighbor_1
PK_neighbor_2
...
```

If neighbor information does not improve a simple model out of sample, graph modeling may not be justified.

---

# 34. Required Model-Level Baselines

Before accepting GNN value, eventually compare:

```text
1. HAR / linear baseline
2. Stock-only LSTM
3. Stock-only model + market factor
4. Fully connected attention baseline
5. LSTM + Return GAT
6. LSTM + PK-Vol GAT
7. LSTM + Directed PK Lead-Lag GAT
8. LSTM + Multi-Edge GAT
```

The EDA should prepare the graph candidates for these comparisons.

---

# 35. Recommended Quality Gates

These are experimental decision gates rather than universal statistical laws.

## Gate 1 — Cross-stock dependence exists

Observed pairwise Parkinson-volatility relationships should be stronger than a temporal-shuffle null baseline.

If not:

```text
GNN justification is weak.
```

---

## Gate 2 — Relationships survive out of sample

Relationships discovered in the training period should remain meaningfully present in validation/test periods.

Measure:

```text
rank stability
sign stability
top-K neighbor overlap
```

---

## Gate 3 — Dynamic structure exists

If rolling correlations change substantially across time or regimes, dynamic graph construction is justified.

If the graph is nearly constant:

```text
static graph may be enough.
```

---

## Gate 4 — Directionality exists

If:

\[
L_{ij}(1)
\]

and:

\[
L_{ji}(1)
\]

are materially different and stable, directed edges may be justified.

Otherwise use symmetric relationships.

---

## Gate 5 — PK graph is relevant to the target

Parkinson-volatility graph features should demonstrate stronger relevance to future Parkinson volatility than arbitrary or random graphs.

---

## Gate 6 — Neighbor information adds value

A simple model with neighbor features should outperform a stock-only baseline out of sample.

If not, a full GNN is unlikely to help much.

---

## Gate 7 — GNN beats market-factor baseline

Eventually require:

```text
Stock-only model + market factor
```

versus:

```text
Stock-only model + graph
```

This prevents falsely attributing a common market factor to GNN intelligence.

---

# 36. Important Ablation Experiments

Run at least the following graph ablations.

```text
A0: No graph
A1: Fully connected graph
A2: Return-correlation graph
A3: Parkinson-volatility correlation graph
A4: Directed PK lead-lag graph
A5: Volume graph
A6: PK + return
A7: PK + volume
A8: PK + lead-lag
A9: PK + return + volume + lead-lag + sector
A10: Random graph control
```

The random graph control is important.

If a random graph performs similarly to the proposed graph, graph construction is not adding meaningful structure.

---

# 37. Random Graph Controls

Create random graphs preserving approximately the same:

```text
number of nodes
number of edges
degree distribution, if possible
```

Compare downstream validation performance.

This helps answer:

```text
Does economic/data-driven topology matter,
or does the model simply benefit from extra parameters?
```

---

# 38. Edge Attribute Scaling

Edge attributes have different numerical ranges.

For example:

```text
correlation: [-1, 1]
same_sector: {0, 1}
stability: [0, 1]
```

If edge attributes are used directly in a neural model, standardize continuous features using training data only.

Possible handling:

```text
continuous edge features -> standard scaler
binary features -> unchanged
```

Do not normalize using the entire dataset.

---

# 39. Signed Correlation

Test whether negative correlation should be preserved.

Variants:

```text
signed_corr = corr
absolute_corr = abs(corr)
positive_corr = max(corr, 0)
```

For volatility, PK values are nonnegative and correlations may frequently be positive, but negative relationships should not be discarded without EDA.

Compare graph topology under all three transformations.

---

# 40. Correlation Alternatives

Pearson correlation is not the only possible relationship measure.

EDA should include:

```text
Pearson
Spearman
```

Optional later experiments:

```text
partial correlation
mutual information
distance correlation
```

Do not add advanced metrics unless basic analyses show a reason.

Start simple and interpretable.

---

# 41. Partial Correlation / Market-Factor Removal

A high pairwise correlation may simply reflect the entire market moving together.

Construct a market Parkinson-volatility factor:

\[
MarketPK_t = median_i(PK_{i,t})
\]

Then regress each stock's PK volatility on the market factor using training-period information.

Example:

\[
PK_{i,t}
=
\alpha_i
+
\beta_i MarketPK_t
+
\epsilon_{i,t}
\]

Analyze residual correlation:

\[
Corr(\epsilon_i,\epsilon_j)
\]

Compare:

```text
raw PK correlation
market-adjusted PK residual correlation
```

This is an important EDA.

If most graph structure disappears after controlling for the market factor, a complex GNN may not be necessary.

---

# 42. Sector-Adjusted Relationships

Optionally compare:

```text
raw PK correlation
market-adjusted PK correlation
market + sector-adjusted relationship
```

This helps determine whether edges capture:

```text
general market factor
sector factor
pair-specific relationship
```

This decomposition is valuable for interpreting a GNN.

---

# 43. Neighbor Consistency

For each stock and each rolling date, identify its Top-K PK-volatility neighbors.

Example:

```text
VCB:
t1 -> BID, CTG, MBB, TCB, VPB
t2 -> BID, CTG, TCB, ACB, MBB
...
```

Calculate Jaccard similarity between neighbor sets at consecutive snapshots:

\[
J(A,B)
=
\frac{|A \cap B|}{|A \cup B|}
\]

Report average neighbor-set stability.

Low stability supports a dynamic graph.

Extremely low random-like stability may indicate noisy edges.

---

# 44. Graph Density Through Time

For threshold graphs, calculate daily or rolling:

```text
number_of_edges
graph_density
average_degree
largest_component_size
```

Plot these through time.

Question:

```text
Does the market graph become denser during high-volatility periods?
```

This can be a meaningful market-regime characteristic.

---

# 45. Edge Turnover

For dynamic Top-K graphs calculate:

\[
EdgeTurnover_t
=
1 -
\frac{
|E_t \cap E_{t-1}|
}{
|E_t \cup E_{t-1}|
}
\]

Plot edge turnover through time.

High but structured turnover supports dynamic topology.

Near-random turnover may signal unstable/noisy estimation.

---

# 46. Recommended EDA Outputs

The AI executing this document should generate the following.

## Data quality outputs

```text
data_quality_summary.csv
ticker_coverage.csv
```

## Feature datasets

```text
daily_stock_features.parquet
```

## Full-period relationship matrices

```text
return_corr_pearson.csv
return_corr_spearman.csv
pk_corr_pearson.csv
pk_corr_spearman.csv
volume_corr.csv
```

## Lead-lag matrices

```text
pk_leadlag_1.csv
pk_leadlag_2.csv
pk_leadlag_5.csv
return_leadlag_1.csv
volume_to_pk_lag1.csv
volume_to_pk_lag5.csv
```

## Rolling edge data

Prefer long format:

```text
date
source
target
pk_corr_20
pk_corr_60
return_corr_20
return_corr_60
volume_corr_20
pk_leadlag_1
pk_leadlag_5
same_sector
```

Output:

```text
dynamic_edge_features.parquet
```

## Stability outputs

```text
edge_stability.csv
neighbor_stability.csv
edge_turnover.csv
```

---

# 47. Required Visualizations

Generate at least:

```text
01_data_coverage.png
02_return_corr_heatmap.png
03_pk_corr_heatmap.png
04_volume_corr_heatmap.png
05_pk_vs_return_corr_scatter.png
06_pairwise_pk_corr_distribution.png
07_same_vs_cross_sector_pk_corr.png
08_pk_leadlag_heatmap.png
09_volume_to_pk_leadlag_heatmap.png
10_edge_stability_distribution.png
11_neighbor_jaccard_over_time.png
12_graph_density_over_time.png
13_market_pk_over_time.png
14_pk_corr_by_market_regime.png
15_example_rolling_correlations.png
16_raw_vs_market_adjusted_pk_corr.png
```

---

# 48. Graph Visualizations

For selected dates generate graph plots.

Choose at least:

```text
normal-volatility date
high-volatility date
low-volatility date
```

Visualize:

```text
PK correlation Top-5 graph
PK lead-lag directed graph
multi-edge graph using a scalar edge-selection score
```

Node labels should be ticker symbols.

If sector metadata exists, use it only for grouping or annotation.

Avoid overly dense unreadable graph plots.

---

# 49. Recommended Edge Construction Experiments

Run the following topology configurations:

| ID | Graph | Directed | Edge Selection | Edge Attributes |
|---|---|---:|---|---|
| G0 | Fully connected | No | all pairs | none/basic |
| G1 | Return | No | Top-K | return corr |
| G2 | PK correlation | No | Top-K | PK corr |
| G3 | PK correlation | No | threshold | PK corr |
| G4 | PK lead-lag | Yes | Top-K outgoing | PK lag |
| G5 | PK + volume | No | Top-K PK | PK + volume |
| G6 | Multi-edge | Yes/No variants | Top-K | PK + return + volume + lag + sector |
| GR | Random control | matched | random | random/matched |

Recommended primary candidates:

```text
G2
G4
G6
```

---

# 50. Recommended Top-K Search

Do not decide K based only on intuition.

Evaluate:

```text
K = 3
K = 5
K = 8
K = 10
```

For every K report:

```text
average graph density
neighbor stability
sector purity
edge strength
validation predictive usefulness
```

Recommended default starting point:

```text
K = 5
```

but do not treat it as final until tested.

---

# 51. Edge Selection Score for Multi-Attribute Graph

Do not manually create an arbitrary weighted sum as the final method unless validated.

For EDA only, a ranking score can be used.

Example:

\[
Score_{ij}
=
z(|PKCorr_{ij}|)
+
z(|PKLeadLag_{ij}|)
+
0.5z(|VolumeCorr_{ij}|)
\]

This score is only for exploratory neighbor ranking.

For the final edge-aware GAT, prefer feeding individual edge attributes to the model and allowing the network to learn their usefulness.

---

# 52. Edge-Aware GAT Recommendation

If the chosen GNN architecture supports edge features, the attention calculation should conceptually depend on:

```text
source node embedding
target node embedding
edge attributes
```

Conceptually:

\[
\alpha_{ij}
=
Attention(
h_i,
h_j,
e_{ij}
)
\]

where:

```text
h_i = source stock representation
h_j = target stock representation
e_ij = relationship attributes
```

Do not collapse all edge information into one scalar if the architecture can learn from a vector.

---

# 53. Suggested Final Architecture After EDA

Potential architecture:

```text
Daily OHLCV
    |
    +--> stock historical features
    |
    v
22-day sequence per stock
    |
    v
LSTM / temporal encoder
    |
    v
node embedding h_i(t)
    |
    +-----------------------------+
    | Dynamic graph G_t           |
    |                             |
    | edge features:              |
    | - PK corr                   |
    | - return corr               |
    | - volume corr               |
    | - PK lead-lag               |
    | - volume -> PK lag          |
    | - sector relation           |
    +-----------------------------+
    |
    v
Edge-aware Dynamic GAT
    |
    v
graph-enhanced stock embedding
    |
    v
prediction head
    |
    v
Parkinson volatility forecast

T+1 / T+5 / T+10 / T+22
```

---

# 54. Prediction Target Alignment

Be explicit about the target.

Examples:

## T+1

\[
y_{i,t}
=
PK_{i,t+1}
\]

## T+5 point target

\[
y_{i,t}
=
PK_{i,t+5}
\]

or alternatively a forward aggregate:

\[
y_{i,t}^{(5)}
=
mean(
PK_{i,t+1:t+5}
)
\]

These are different targets.

Do not mix:

```text
point forecast at T+5
```

with:

```text
average volatility over next 5 days
```

The implementation must document the exact target definition.

---

# 55. Leakage Check for Forecast Horizons

For prediction date \(t\):

```text
features <= t
graph snapshot <= t
target > t
```

For T+5:

```text
X uses data <= t
y uses future definition beginning after t
```

Never allow:

```text
rolling graph correlation
```

to include target-period observations.

Write automated assertions checking this.

---

# 56. Recommended Statistical Reporting

For major EDA claims provide:

```text
effect size
confidence interval where appropriate
p-value
FDR-adjusted q-value
sample size
```

Avoid conclusions based only on colorful heatmaps.

---

# 57. Minimum Evidence Required to Recommend a GNN

Recommend proceeding to a GNN only if several of these are observed:

1. Parkinson-volatility cross-stock dependence is clearly stronger than shuffled-null relationships.
2. Relationships remain meaningful in validation/test periods.
3. Neighbor structure has reasonable stability.
4. Dynamic changes correspond to interpretable regimes rather than pure noise.
5. Directed lag relationships exist for at least some stocks.
6. Cross-stock information improves a simple predictive baseline.
7. Graph structure retains information after controlling for a broad market factor.
8. Data-driven graphs outperform random graph controls.

If these conditions fail, recommend a simpler model.

---

# 58. Possible Conclusions

The EDA must be allowed to conclude any of the following.

## Conclusion A — Strong GNN evidence

```text
Cross-stock PK relationships are strong,
stable enough,
dynamic across regimes,
and predictive out of sample.
```

Recommendation:

```text
Dynamic Edge-Aware GAT
```

---

## Conclusion B — Static graph sufficient

```text
Relationships are strong but highly stable.
```

Recommendation:

```text
Static GAT or fixed sector/correlation graph
```

---

## Conclusion C — Market factor dominates

```text
Pairwise relationships disappear after controlling for market PK.
```

Recommendation:

```text
LSTM/HAR + market factor
```

instead of a complex GNN.

---

## Conclusion D — Weak graph evidence

```text
Relationships are unstable,
random-like,
or non-predictive out of sample.
```

Recommendation:

```text
Do not use GNN.
```

---

# 59. Deliverable: EDA Summary Report

Generate:

```text
EDA_GRAPH_REPORT.md
```

The report must include:

## Executive Summary

Answer directly:

```text
Should a GNN be used: YES / MAYBE / NO?
```

Then explain why.

## Data Summary

Include:

```text
number of tickers
number of trading days
date range
missingness
```

## Parkinson Volatility Summary

Show:

```text
distribution
cross-stock correlation
regime behavior
```

## Return Relationships

Summarize return correlation structure.

## Volume Relationships

Summarize normalized-volume relationships.

## Lead-Lag Findings

Identify strongest stable directional edges.

## Sector Findings

Explain whether economic sectors correspond to graph communities.

## Dynamic Graph Evidence

Explain rolling-correlation and edge-turnover behavior.

## Market-Factor Adjustment

Explain how much pairwise PK structure remains after controlling for market volatility.

## Recommended Graph

Specify:

```text
directed / undirected
static / dynamic
Top-K / threshold / fully connected
K value
edge features
```

## Rejected Edge Features

List features tested but not recommended.

## Leakage Audit

Document how future information was prevented.

## Next Model Experiments

Recommend exact downstream model ablations.

---

# 60. Machine-Readable Final Recommendation

Also create:

```text
graph_recommendation.json
```

Example structure:

```json
{
  "use_gnn": true,
  "confidence": "medium",
  "graph_type": "dynamic_directed",
  "edge_selection": "top_k",
  "top_k": 5,
  "primary_relationship": "parkinson_volatility",
  "edge_features": [
    "pk_corr_20",
    "pk_corr_60",
    "return_corr_20",
    "volume_corr_20",
    "pk_leadlag_1",
    "pk_leadlag_5",
    "volume_to_pk_lag1",
    "same_sector"
  ],
  "rejected_features": [],
  "main_evidence": [],
  "main_risks": []
}
```

The actual values must come from the experiment and must not be hard-coded.

---

# 61. Recommended Project Structure

```text
graph_eda/
|
|-- config/
|   `-- config.yaml
|
|-- data/
|   |-- raw/
|   `-- processed/
|
|-- src/
|   |-- data_quality.py
|   |-- features.py
|   |-- parkinson.py
|   |-- correlation.py
|   |-- lead_lag.py
|   |-- rolling_graph.py
|   |-- stability.py
|   |-- regime.py
|   |-- market_adjustment.py
|   |-- graph_builders.py
|   |-- statistical_tests.py
|   `-- visualization.py
|
|-- outputs/
|   |-- tables/
|   |-- figures/
|   |-- graphs/
|   `-- reports/
|
|-- notebooks/
|   `-- graph_eda.ipynb
|
|-- tests/
|   |-- test_parkinson.py
|   |-- test_leakage.py
|   |-- test_alignment.py
|   `-- test_graph_builder.py
|
`-- README.md
```

---

# 62. Suggested Configuration

Example:

```yaml
data:
  date_column: date
  ticker_column: ticker
  open_column: open
  high_column: high
  low_column: low
  close_column: close
  volume_column: volume

volatility:
  measure: parkinson
  retain_variance: true

rolling_windows:
  - 20
  - 60
  - 120

lead_lags:
  - 1
  - 2
  - 5
  - 10

graph:
  top_k_values:
    - 3
    - 5
    - 8
    - 10
  thresholds:
    - 0.2
    - 0.3
    - 0.4
    - 0.5

statistics:
  multiple_testing: benjamini_hochberg
  alpha: 0.05
  permutation_iterations: 1000

outputs:
  save_csv: true
  save_parquet: true
  save_png: true
```

---

# 63. Implementation Requirements for the AI Agent

The AI running this experiment should:

1. Inspect the actual dataset schema before coding.
2. Map real column names to the configuration.
3. Validate OHLC consistency.
4. Calculate Parkinson volatility exactly.
5. Create chronological features without future leakage.
6. Produce all core relationship matrices.
7. Run rolling and lagged analyses.
8. Apply FDR correction.
9. Perform permutation/null tests.
10. Analyze edge stability.
11. Analyze Top-K neighbor stability.
12. Compare market-adjusted versus raw relationships.
13. Compare static versus dynamic graph evidence.
14. Compare directed versus undirected graph evidence.
15. Generate graph candidates.
16. Produce figures and machine-readable outputs.
17. Write a final evidence-based recommendation.
18. Never claim a GNN is useful without measurable evidence.

---

# 64. Coding Requirements

Preferred stack:

```text
Python 3.10+
pandas
numpy
scipy
statsmodels
scikit-learn
matplotlib
networkx
pyarrow
```

Optional later for graph modeling:

```text
PyTorch
PyTorch Geometric
```

Use vectorized computations when practical.

Avoid nested Python loops over dates and stock pairs when a matrix operation is available.

Keep graph feature generation deterministic with a fixed random seed.

---

# 65. Validation Tests

At minimum implement automated checks.

## Parkinson formula test

For known High and Low values verify the formula numerically.

## No negative PK volatility

Assert:

```text
pk_vol >= 0
```

## Date order

Assert all ticker series are sorted chronologically.

## Leakage test

For each graph snapshot:

```text
max(edge_source_date) <= prediction_date
```

## Target test

Assert target date is later than the feature cutoff date.

## Correlation matrix test

Assert:

```text
diagonal approximately 1
matrix symmetric
```

for contemporaneous undirected correlation matrices.

## Directed lag test

Do not enforce symmetry for lead-lag matrices.

---

# 66. Interpretation Rules

The final analysis must follow these rules.

Do not say:

```text
Stock A causes Stock B.
```

Prefer:

```text
Stock A's historical Parkinson volatility contains a stable lagged association with Stock B's subsequent Parkinson volatility.
```

Do not say:

```text
The GNN is necessary because stocks are correlated.
```

Prefer:

```text
The graph hypothesis is supported only if cross-stock information is stable and improves out-of-sample prediction beyond simpler baselines.
```

Do not interpret same-day correlation as evidence of directional transmission.

---

# 67. Research Questions to Answer Explicitly

The final report should contain explicit answers to:

### RQ1

Are daily Parkinson-volatility relationships between stocks statistically and economically meaningful?

### RQ2

Are Parkinson-volatility relationships stronger than return relationships for this forecasting problem?

### RQ3

Do normalized volume relationships provide complementary information?

### RQ4

Are there stable directional PK-volatility lead-lag relationships?

### RQ5

Does relationship strength change across market regimes?

### RQ6

Does graph structure remain after controlling for a market-wide Parkinson-volatility factor?

### RQ7

Are sector-linked stocks more strongly connected?

### RQ8

Is a dynamic graph preferable to a static graph?

### RQ9

What graph sparsification method is most defensible?

### RQ10

Which edge attributes should be passed into a GNN?

---

# 68. Recommended Final Decision Matrix

Use a table like:

| Evidence | Weak | Moderate | Strong |
|---|---:|---:|---:|
| PK cross-stock dependence | | | |
| Null-test separation | | | |
| Out-of-sample stability | | | |
| Lead-lag directionality | | | |
| Regime dependence | | | |
| Sector structure | | | |
| Market-adjusted dependence | | | |
| Neighbor predictive gain | | | |
| Dynamic graph justification | | | |

Then produce:

```text
Final recommendation:
NO GNN / STATIC GNN / DYNAMIC GNN / DYNAMIC DIRECTED EDGE-AWARE GNN
```

---

# 69. Priority Order

If compute or time is limited, execute in this order.

## Priority 1

```text
data validation
Parkinson volatility
return correlation
PK correlation
volume-shock correlation
```

## Priority 2

```text
rolling PK correlation
PK lead-lag
volume -> future PK
edge stability
```

## Priority 3

```text
market-factor adjustment
sector analysis
Top-K stability
regime analysis
```

## Priority 4

```text
graph topology experiments
random graph controls
simple predictive neighbor tests
```

## Priority 5

```text
GNN architecture experiments
```

---

# 70. Primary Recommendation Before Running the EDA

The strongest graph hypothesis for this project is:

```text
Dynamic Parkinson-Volatility Graph
```

rather than a graph based on raw stock prices.

A strong multi-edge candidate is:

```text
edge_attr(i -> j, t) = [
    pk_corr_20,
    pk_corr_60,
    return_corr_20,
    return_corr_60,
    volume_corr_20,
    pk_leadlag_1,
    pk_leadlag_5,
    volume_to_pk_lag1,
    same_sector
]
```

However, this is only a hypothesis.

The EDA must be allowed to reject some or all of these features.

The final graph should be chosen based on out-of-sample evidence, stability, leakage-safe computation, and downstream predictive usefulness.

---

# 71. Final Instruction to the AI Executing This Plan

Do not start by assuming that a GNN is the correct solution.

Start from the data.

First establish:

```text
whether cross-stock structure exists,
whether it is stable,
whether it is directional,
whether it changes over time,
whether it survives market-factor adjustment,
and whether it adds predictive information.
```

Only after those conditions are evaluated should the system recommend:

```text
no graph,
static graph,
dynamic graph,
or dynamic directed edge-aware graph.
```

All conclusions must be backed by generated tables, plots, statistical tests, and leakage-safe out-of-sample evidence.
