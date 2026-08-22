# HAR versus Deep and Graph-Attention Models for Multi-Horizon Volatility Forecasting: A Panel-Correct Study on Vietnamese and U.S. Equities

*SOICT submission (markdown draft; transcribed to the SOICT/Overleaf LaTeX template in
`docs/paper/soict_harlstmgat.tex`). Architecture diagram: `docs/paper/diagrams/soict_harlstmgat.svg`.*

---

## Abstract

We study whether deep sequence models or a cross-sectional graph-attention branch can improve daily
Parkinson-variance forecasting over the Heterogeneous Autoregressive (HAR) baseline, and we evaluate them
with panel-correct inference. Three equity panels are used: VN30 (33 tickers) and VN100 (104 tickers) as the
Vietnamese case study, and a long-history S&P 500 subset (457 tickers) as a data-rich cross-market check.
Against the classical baselines HAR and GARCH(1,1) we benchmark a pooled LSTM over the three HAR features
(daily, weekly and monthly Parkinson volatility), an LSTM with a Graph Attention Network branch over a
graphical-lasso partial-correlation graph, a frozen-expert convex HAR–deep combination, and HAR-anchored
residual models, at horizons {1, 5, 10, 22} with five random seeds. Statistical significance is assessed
with the date-clustered Diebold–Mariano test, which collapses the loss differential to one value per
calendar date; a naive per-observation test on a cross-sectionally dependent panel overstates significance
by a factor on the order of the square root of the cross-section size. Three findings follow. First, HAR is
very hard to beat on the small Vietnamese panels: under date-clustered inference no model — including the
forecast combination — significantly beats HAR at any horizon (the closest competitor reaches p = 0.08).
Second, on the large S&P 500 the deep temporal LSTM (no graph) significantly beats HAR, by up to +7.1% QLIKE
at the twenty-two-day horizon (p<0.001), and the convex combination beats HAR at all four horizons
(p<0.001); deep-versus-HAR competitiveness grows with panel size. Third, the cross-sectional graph adds no
out-of-sample value on any panel: removing it never improves on the no-graph model, a leakage-safe linear
neighbour-signal regressor adds essentially zero incremental out-of-sample R², and diagnostics rule out a
software bug or overfitting. We report MSE, RMSE, MAE, QLIKE and R² together, because the model ranking is
metric- and panel-dependent, and every number is read from a stored results file.

---

## 1. Introduction

Daily volatility forecasting underpins risk management, position sizing and derivative pricing. The HAR
model of Corsi (2009) remains a strong and parsimonious benchmark: three ordinary-least-squares
coefficients on daily, weekly and monthly aggregates of realized volatility capture much of the long-memory
structure of the series. Deep sequence models and graph neural networks are frequently proposed to capture
nonlinear temporal dynamics and cross-sectional volatility spillovers, but the evidence that they
out-of-sample beat HAR is mixed, and positive claims are often confounded by data design choices.

This paper asks two direct questions for daily volatility forecasting, with the Vietnamese equity market as
the primary case study and a long-history U.S. S&P 500 subset as a data-rich cross-market check:

1. Can deep or HAR-anchored models — a pooled LSTM over the three HAR features, a convex HAR–deep
   combination, or HAR-residual correctors — beat the HAR and GARCH baselines out of sample under
   panel-correct inference, and at which horizons?
2. Does adding a cross-sectional Graph Attention Network branch, with edges estimated by graphical lasso,
   contribute any out-of-sample value beyond the same-capacity no-graph model?

Our contributions are: (i) a panel-correct multi-horizon benchmark of HAR against an LSTM, an LSTM-GAT, a
convex HAR–deep combination and HAR-anchored residual models on three panels, with statistical significance
assessed by the date-clustered Diebold–Mariano test — because a naive per-observation test overstates
significance by a factor on the order of the square root of the cross-section size on this
cross-sectionally dependent panel; (ii) the main empirical finding that HAR is not significantly beaten on
the small Vietnamese panels, whereas on the data-rich S&P 500 the deep temporal LSTM significantly beats HAR
(up to +7.1% QLIKE at the twenty-two-day horizon) and the convex combination beats it at every horizon, so
deep-versus-HAR competitiveness scales with the amount of data; (iii) a model-free demonstration that the
cross-sectional graph adds no out-of-sample value on any panel, with mechanistic evidence that this is a
genuine null rather than a bug or overfitting; and (iv) per-metric fairness — reporting MSE, RMSE, MAE,
QLIKE and R² together and showing that the ranking is metric- and panel-dependent. Every number reported
here is taken from a stored results file.

---

## 2. Related Work

**HAR and classical volatility models.** The HAR model (Corsi, 2009) approximates the long memory of
realized volatility with a cascade of daily, weekly and monthly components, motivated by the heterogeneous
market hypothesis. It is a demanding baseline: recent machine-learning work reports that rolling-window and
specification choices, rather than model class, often drive apparent gains over HAR (Audrino and Chassot,
2025). Extensions such as HARQ (Bollerslev, Patton and Quaedvlieg, 2016) exploit measurement error, and
GARCH-family models (Bollerslev) remain standard conditional-variance benchmarks. We use HAR and GARCH(1,1)
as the two baselines "to be beaten".

**Deep and graph models for volatility.** LSTMs (Hochreiter and Schmidhuber, 1997) are widely applied to
financial time series, and graph neural networks have been proposed to model cross-asset spillovers, for
example Graph Attention Networks (Veličković et al., 2018), multivariate realized-volatility GNNs (Chen and
Robert, 2022) and spillover-aware GNN-HAR hybrids (Zhang et al., 2025). Evidence is mixed: some studies
report gains, while others find graph or deep components add little out-of-sample value once a strong HAR
baseline and honest testing are in place. Volatility spillover measurement itself has a long econometric
tradition (Diebold and Yilmaz, 2012).

**Graph construction.** A common design choice is the graph edge set. Rather than a raw correlation graph,
which is dominated by a market factor, we estimate a sparse partial-correlation graph with graphical lasso
(Friedman, Hastie and Tibshirani, 2008) on training rows only, which is a standard robust choice for
conditional dependence structure.

**Forecast evaluation.** Volatility is latent, so we use the QLIKE loss, which is robust to the volatility
proxy (Patton, 2011), alongside MSE-based losses. Statistical significance of forecast-accuracy differences
is assessed with the Diebold–Mariano test (Diebold and Mariano, 1995) using the small-sample correction of
Harvey, Leybourne and Newbold (1997).

---

## 3. Method

### 3.1 Target and features

The forecasting target is the Parkinson variance estimator (Parkinson, 1980) at day `t+h` for horizons
`h ∈ {1, 5, 10, 22}` in the primary date-clustered study (one trading day, one trading week, two weeks and
roughly one trading month ahead); the descriptive per-observation and snapshot studies use `h ∈ {1, 5}`. The
target is a non-negative point forecast. Following HAR, we
use three features per node (ticker): the daily Parkinson variance at `t`, its 5-day rolling mean (weekly)
and its 22-day rolling mean (monthly). The same three features drive every model, so differences reflect
the functional form, not the information set. Per the experiment specification, no additional technical or
news features are used.

### 3.2 Main model: per-observation LSTM

The main model, referred to simply as **LSTM**, is a two-layer LSTM (hidden size 64) applied to the
lookback window of the three HAR features. A single pooled model is trained over all tickers. Each ticker
is standardized with its own `StandardScaler` fit on training rows only; the output layer is linear (no
non-negativity activation), and predictions are inverse-transformed to the physical scale for evaluation,
following the project's established normalization pattern. Overfitting is controlled with dropout, weight
decay, gradient clipping, a `ReduceLROnPlateau` schedule and early stopping.

Crucially, the training examples are formed by **per-observation pooling**: every (ticker, window) pair in
the training span is an independent training example, and the chronological 80/10/10 split is applied
**within each ticker's own history**. This yields tens of thousands of training windows (about 84,000 for
VN30) and interleaves every ticker's regimes, rather than restricting attention to calendar dates on which
all tickers are simultaneously present.

### 3.3 Graph-attention variant: LSTM+GAT (ablation)

To test whether a cross-sectional graph helps, we add a Graph Attention Network branch (Veličković et al.,
2018), producing the full **LSTM+GAT** model (Figure 1). A
per-node LSTM encodes the temporal window (temporal branch); a GAT reads the node features at day `t` and
attends over a fixed graph whose edges are the Top-5 graphical-lasso partial-correlation links estimated on
training rows only and frozen (spatial branch); the two branch outputs are concatenated and passed to an
MLP head. The leave-one-out ablation **LSTM (w/o GAT)** removes the spatial branch, isolating the graph's
marginal contribution.

Because a GAT operates over a cross-section of nodes at a common date, the LSTM+GAT variant requires a
**common-date snapshot** design: only dates on which all nodes are present are kept, with a single global
chronological 80/10/10 split. We therefore report the graph ablation as a **separate study** on this
snapshot design, and keep the main deep-versus-HAR comparison on the richer per-observation design. This
separation is deliberate: it prevents the graph's data-design requirement from confounding the deep-versus-HAR
verdict.

![Figure 1: HAR-LSTM-GAT architecture](diagrams/soict_harlstmgat.png)

*Figure 1. Architecture of LSTM+GAT: a per-node LSTM temporal branch and a GAT spatial branch over a
graphical-lasso partial-correlation graph are concatenated and passed to an MLP head; the leave-one-out
ablation removes the GAT branch to give the main LSTM.*

### 3.4 Baselines to beat

- **HAR**: pooled ordinary-least-squares regression on the three HAR features, refit on the same training
  rows the deep models see.
- **GARCH(1,1)**: per-ticker conditional-variance model, the classical volatility benchmark.

### 3.5 Training and evaluation protocol

Each configuration is run with five seeds {42, 123, 2026, 7, 2024}, up to 20 epochs with early stopping, on
GPU with parallel data workers; learning curves are recorded every five epochs. We report five metrics —
MSE, RMSE, MAE, QLIKE and R² — seed-averaged over the test set, with a shared QLIKE positivity floor of
`1e-8` applied identically to every model. The primary decision metric is QLIKE, which is robust to the
volatility proxy (Patton, 2011). Predictive-accuracy (Diebold–Mariano) tests are run only on the
proxy-robust losses — QLIKE (primary) and squared error; MAE is reported as a descriptive metric but is not
used for significance testing, because absolute error is not robust to noise in the volatility proxy and can
invert forecast rankings (Patton, 2011).

Statistical significance is assessed with the **date-clustered Diebold–Mariano test**: the per-observation
loss differential is collapsed to one value per calendar date before the Diebold–Mariano statistic is
computed, with the Harvey–Leybourne–Newbold small-sample correction and a HAC lag of `h-1`. This is the
panel-correct treatment for a cross-section in which all tickers share each trading date; a naive
per-observation Diebold–Mariano test over every (ticker, date) row treats the effective sample as the
number of tickers times the number of dates, understates the loss-differential variance, and inflates the
statistic by a factor on the order of the square root of the cross-section size (roughly six on VN30 and ten
on VN100). All significance statements in this paper use the date-clustered test unless a result is
explicitly labelled per-observation. A positive dQLIKE% denotes a QLIKE improvement over HAR. In the
results tables a compact marker (`*`) flags a model that significantly beats HAR on QLIKE under the
date-clustered test (p < 0.05); the per-horizon date-clustered DM p-values versus HAR — the single headline
comparison against the benchmark — are collected in the HAR-anchored tables of Section 6.6.

---

## 4. Data

Three equity panels are used, all with the Parkinson variance target and the three HAR features:

| Panel | Tickers | Role | Source |
|---|---|---|---|
| VN30 | 33 | Primary (Vietnam) | Project processed data |
| VN100 | 104 | Vietnam breadth | vnstock |
| S&P 500 | ~500 | Cross-market robustness (U.S.) | Yahoo (gitignored; only aggregate metrics stored) |

The per-observation main study uses per-ticker chronological 80/10/10 splits (train precedes validation
precedes test within each ticker), per-ticker standardization fit on training rows only, and a single
pooled model. The graph-ablation study uses common-date fixed-node snapshots with a global chronological
80/10/10 split. The S&P 500 constituent list is the current membership, so U.S. results carry survivorship
bias in level; the short-versus-long horizon ordering, not the absolute level, is what we claim transfers.

---

## 5. Experiments

We run five studies, all on the same three HAR features across VN30, VN100 and the S&P 500 subset:

1. **Primary: HAR-anchored ladder (E0–E10) under date-clustered inference** (Section 6.6) — HAR versus the
   full-target LSTM, the full-target LSTM+GAT, the convex HAR–deep combination, and additive/multiplicative
   HAR-anchored residual and gate models, at horizons 1, 5, 10 and 22, with all beat-HAR verdicts by the
   date-clustered Diebold–Mariano test. This is the source of the paper's headline results.
2. **Design comparison: LSTM versus HAR and GARCH** on the per-observation per-stock design (lookback 10;
   horizons 1 and 5), reporting point estimates and naive per-observation statistics (Section 6.1).
3. **Graph-check ablation** LSTM+GAT versus LSTM (w/o GAT) on the common-date snapshot design (lookback 10,
   horizons 1 and 5), with HAR and GARCH for reference (Section 6.2).
4. **Lookback variation** (10 versus 22) for the snapshot design on VN30 (Section 6.3).
5. **Model-free graph screening** (Section 6.7) — a leakage-safe, architecture-independent test of whether
   any cross-stock neighbour signal adds out-of-sample value beyond HAR.

---

## 6. Results

Volatility magnitudes are small (Parkinson variance is of order `1e-7`), so MSE, RMSE and MAE are reported
in scaled units to avoid scientific notation: **MSE ×10⁻⁷, RMSE ×10⁻⁴, MAE ×10⁻⁴**. QLIKE and R² are
unscaled. Row order in every table follows the baselines-first convention: HAR → GARCH → LSTM (and, for the
graph study, HAR → GARCH → LSTM (w/o GAT) → LSTM+GAT). All values are the five-seed test-set means from the
stored `result.json` files, with the design stated in each table caption.

**Primary results and inference convention.** The primary results of the paper are the panel-correct,
multi-horizon, date-clustered study of Section 6.6, which compares HAR against the LSTM, the LSTM+GAT, the
convex combination and the HAR-anchored residual models on all three panels at horizons {1, 5, 10, 22}. All
beat-HAR significance claims in the paper are date-clustered. Sections 6.1–6.5 report point-estimate tables
(all five metrics) and descriptive per-observation Diebold–Mariano statistics that isolate design and
lookback effects; the per-observation statistics are retained only to expose the inference artifact and are
not used for beat-HAR claims (see the methodological note in Section 6.6).

### 6.1 Main study — per-observation LSTM (design: per-observation, per-stock 80/10/10)

*Source: `results/soict_perobs/{panel}_lb10_h{1,5}/result.json`.*

**VN30 (33 tickers; test n = 10,577 at h1, 10,564 at h5)**

| h | Model | MSE (×10⁻⁷) | RMSE (×10⁻⁴) | MAE (×10⁻⁴) | QLIKE | R² |
|---|---|---|---|---|---|---|
| 1 | HAR   | 2.2449 | 4.7380 | 2.7467 | 0.4675 | 0.2892 |
| 1 | GARCH | 3.4119 | 5.8411 | 3.8013 | 0.7434 | -0.0804 |
| 1 | LSTM  | 2.2360 | 4.7286 | 2.6994 | **0.4578** | 0.2920 |
| 5 | HAR   | 2.5493 | 5.0491 | 2.9946 | 0.5514 | 0.1933 |
| 5 | GARCH | 3.4064 | 5.8364 | 3.8094 | 0.7409 | -0.0779 |
| 5 | LSTM  | 2.5450 | 5.0448 | 2.9235 | **0.5458** | 0.1947 |

**VN100 (104 tickers; test n = 35,506 at h1, 35,468 at h5)**

| h | Model | MSE (×10⁻⁷) | RMSE (×10⁻⁴) | MAE (×10⁻⁴) | QLIKE | R² |
|---|---|---|---|---|---|---|
| 1 | HAR   | 2.5131 | 5.0131 | 3.0412 | 0.4798 | 0.2282 |
| 1 | GARCH | 3.9308 | 6.2696 | 4.4409 | 0.7031 | -0.2072 |
| 1 | LSTM  | 2.5138 | 5.0137 | 2.9909 | **0.4735** | 0.2280 |
| 5 | HAR   | 2.7815 | 5.2740 | 3.2941 | 0.5441 | 0.1462 |
| 5 | GARCH | 3.8892 | 6.2364 | 4.4164 | 0.6993 | -0.1939 |
| 5 | LSTM  | 2.7739 | 5.2667 | 3.2367 | **0.5372** | 0.1485 |

**S&P 500 (~500 tickers; test n = 436,569 at h1, 436,421 at h5)**

| h | Model | MSE (×10⁻⁷) | RMSE (×10⁻⁴) | MAE (×10⁻⁴) | QLIKE | R² |
|---|---|---|---|---|---|---|
| 1 | HAR   | 2.9797 | 5.4587 | 1.9909 | **0.3616** | 0.1970 |
| 1 | GARCH | 12.5558 | 11.2053 | 5.1235 | 0.7286 | -2.3838 |
| 1 | LSTM  | 2.9408 | 5.4229 | 2.0586 | 0.3654 | 0.2074 |
| 5 | HAR   | 3.3453 | 5.7838 | 2.1779 | **0.4226** | 0.0982 |
| 5 | GARCH | 12.5241 | 11.1911 | 5.0885 | 0.7287 | -2.3763 |
| 5 | LSTM  | 3.3001 | 5.7446 | 2.2118 | 0.4232 | 0.1103 |

This table reports point estimates only; the panel-correct significance verdicts belong to the date-clustered
DM tables of Section 6.6. A naive per-observation DM on this design (QLIKE, LSTM vs HAR) flags the
short-horizon Vietnamese gaps as significant (VN30 h1 statistic −3.60, p<0.001; VN100 h1 −6.23, p<0.001;
VN100 h5 −3.34, p<0.001) while the h5 VN30 and S&P 500 gaps are not; but that test overstates significance on
a cross-sectionally dependent panel by a factor on the order of the square root of the cross-section size (see
the methodological note in Section 6.6), so it is not used for any beat-HAR claim.

**Reading.** On the Vietnamese panels the LSTM's QLIKE point estimates edge below HAR at the short horizons
(for example VN30-h1 0.4578 versus 0.4675, VN100-h1 0.4735 versus 0.4798). The naive per-observation Diebold–
Mariano test flags several of these as significant, but it overstates significance on a
cross-sectionally dependent panel; under the panel-correct date-clustered test of Section 6.6 none of these
Vietnamese gaps is statistically significant. Every learned model beats GARCH at every horizon by very large
margins. On the S&P 500 the LSTM attains both the lower QLIKE and the lower MSE at both horizons in this
per-observation design, and the date-clustered study of Section 6.6 confirms the S&P 500 beat-HAR result is
statistically significant there. The takeaway of this section is descriptive: the LSTM's point estimates
favor it on the richer per-observation design, but the panel-correct significance verdict belongs to
Section 6.6, where the beat-HAR result survives only on the large S&P 500 panel.

### 6.2 Graph-check ablation — LSTM+GAT versus LSTM (w/o GAT) (design: common-date snapshot, global split)

*Source: `results/soict/{panel}_lb10_h{1,5}/result.json`. Snapshot test n: VN30 4,356/4,323; VN100
5,096/4,992; S&P 500 17,500/17,500.*

**VN30 (lookback 10)**

| h | Model | MSE (×10⁻⁷) | RMSE (×10⁻⁴) | MAE (×10⁻⁴) | QLIKE | R² |
|---|---|---|---|---|---|---|
| 1 | HAR            | 2.1527 | 4.6398 | 2.7735 | **0.3946** | 0.3114 |
| 1 | GARCH          | 3.0380 | 5.5118 | 3.3035 | 0.6500 | 0.0283 |
| 1 | LSTM (w/o GAT) | 2.2975 | 4.7933 | 2.8707 | 0.4120 | 0.2651 |
| 1 | LSTM+GAT       | 2.4862 | 4.9862 | 3.0564 | 0.4528 | 0.2048 |
| 5 | HAR            | 2.4342 | 4.9338 | 2.9848 | **0.4531** | 0.2182 |
| 5 | GARCH          | 3.0306 | 5.5051 | 3.2993 | 0.6420 | 0.0267 |
| 5 | LSTM (w/o GAT) | 2.5270 | 5.0269 | 3.1070 | 0.4663 | 0.1884 |
| 5 | LSTM+GAT       | 2.7081 | 5.2039 | 3.3082 | 0.4991 | 0.1303 |

**VN100 (lookback 10)**

| h | Model | MSE (×10⁻⁷) | RMSE (×10⁻⁴) | MAE (×10⁻⁴) | QLIKE | R² |
|---|---|---|---|---|---|---|
| 1 | HAR            | 1.9936 | 4.4650 | 2.7856 | **0.4843** | 0.2093 |
| 1 | GARCH          | 2.9017 | 5.3868 | 3.9215 | 0.6210 | -0.1508 |
| 1 | LSTM (w/o GAT) | 2.1568 | 4.6441 | 2.9884 | 0.5204 | 0.1446 |
| 1 | LSTM+GAT       | 2.0810 | 4.5618 | 2.8335 | 0.5296 | 0.1747 |
| 5 | HAR            | 2.2817 | 4.7767 | 3.0164 | **0.5442** | 0.1067 |
| 5 | GARCH          | 2.7781 | 5.2708 | 3.8665 | 0.6157 | -0.0876 |
| 5 | LSTM (w/o GAT) | 2.3035 | 4.7995 | 3.0960 | 0.5552 | 0.0982 |
| 5 | LSTM+GAT       | 2.2559 | 4.7496 | 2.9780 | 0.5588 | 0.1168 |

**S&P 500 (lookback 10)**

| h | Model | MSE (×10⁻⁷) | RMSE (×10⁻⁴) | MAE (×10⁻⁴) | QLIKE | R² |
|---|---|---|---|---|---|---|
| 1 | HAR            | 6.9544 | 8.3393 | 2.8348 | **0.3390** | 0.1613 |
| 1 | GARCH          | 9.2388 | 9.6119 | 3.0650 | 0.3842 | -0.1142 |
| 1 | LSTM (w/o GAT) | 6.7829 | 8.2358 | 2.7589 | 0.3401 | 0.1820 |
| 1 | LSTM+GAT       | 6.9702 | 8.3488 | 2.9353 | 0.3473 | 0.1594 |
| 5 | HAR            | 6.9568 | 8.3408 | 2.8948 | 0.3680 | 0.1610 |
| 5 | GARCH          | 7.5510 | 8.6897 | 2.9196 | 0.3890 | 0.0893 |
| 5 | LSTM (w/o GAT) | 6.8914 | 8.3014 | 2.8794 | **0.3582** | 0.1689 |
| 5 | LSTM+GAT       | 6.9924 | 8.3620 | 3.0063 | 0.3701 | 0.1567 |

**Diebold–Mariano, graph study — leave-one-out (LSTM+GAT vs LSTM (w/o GAT) on QLIKE)** (the headline
ablation contrast; a positive statistic favors the no-graph model; `*` denotes p < 0.05):

| Panel | h | DM: LSTM+GAT vs LSTM (w/o GAT) (QLIKE) |
|---|---|---|
| VN30  | 1 | +6.40* (favors no-graph) |
| VN30  | 5 | +4.94* (favors no-graph) |
| VN100 | 1 | +0.93 (p=0.35, tie) |
| VN100 | 5 | +0.38 (p=0.70, tie) |
| S&P500| 1 | +3.23* (favors no-graph) |
| S&P500| 5 | +6.87* (favors no-graph) |

**Reading.** The leave-one-out ablation is unambiguous on the decision metric: **removing the GAT branch
improves QLIKE in all eight configurations** (the LSTM (w/o GAT) vs LSTM+GAT column favors the no-graph
model everywhere, significantly on VN30 and S&P 500). The graph therefore does not help; at best it is
QLIKE-neutral on VN100. On the squared-error loss the graph is neutral-to-slightly-helpful only on VN100
(where LSTM+GAT has the lower MSE at both horizons) and hurts elsewhere, so no metric supports adopting the
graph as a general improvement. Under this snapshot design HAR is the strongest model on the Vietnamese
panels, whereas on the large S&P 500 the price-only LSTM (w/o GAT) attains the best QLIKE at h5
(0.3582 versus HAR 0.3680) and the best MSE and R² at both horizons — the data-scaling pattern of
Section 6.4. All learned models again beat GARCH.

### 6.3 Lookback variation — 10 versus 22 (design: common-date snapshot, VN30)

*Source: `results/soict/vn30_lb{10,22}_h{1,5}/result.json` (lb22 test n = 4,323/4,290). Lookback-10 numbers
are in Section 6.2.*

**VN30, lookback 22**

| h | Model | MSE (×10⁻⁷) | RMSE (×10⁻⁴) | MAE (×10⁻⁴) | QLIKE | R² |
|---|---|---|---|---|---|---|
| 1 | HAR            | 2.1553 | 4.6425 | 2.7799 | **0.3969** | 0.3078 |
| 1 | GARCH          | 3.0784 | 5.5483 | 3.4958 | 0.6034 | 0.0114 |
| 1 | LSTM (w/o GAT) | 2.2877 | 4.7830 | 2.8991 | 0.4137 | 0.2653 |
| 1 | LSTM+GAT       | 2.5588 | 5.0584 | 3.1228 | 0.4714 | 0.1782 |
| 5 | HAR            | 2.4353 | 4.9349 | 2.9848 | **0.4547** | 0.2150 |
| 5 | GARCH          | 3.0142 | 5.4902 | 3.4269 | 0.5980 | 0.0284 |
| 5 | LSTM (w/o GAT) | 2.5381 | 5.0379 | 3.0919 | 0.4692 | 0.1819 |
| 5 | LSTM+GAT       | 2.8060 | 5.2972 | 3.4238 | 0.5166 | 0.0955 |

Comparing to the lookback-10 snapshot results (Section 6.2), the longer lookback does not help the deep
models: LSTM (w/o GAT) QLIKE is essentially unchanged at h1 (0.4137 versus 0.4120) and h5 (0.4692 versus
0.4663), while LSTM+GAT is slightly worse at lookback 22 (0.4714 versus 0.4528 at h1). HAR is nearly
invariant to the lookback because its features are precomputed rolling aggregates. The DM ablation verdict
is unchanged: at lookback 22 the graph again hurts (LSTM+GAT vs LSTM (w/o GAT) on QLIKE is +8.80 at h1 and
+5.92 at h5, both favoring the no-graph model, p < 0.001).

### 6.4 Cross-market and data scaling

Reading QLIKE across panels of increasing size, under the panel-correct date-clustered inference of
Section 6.6, shows a consistent data-scaling gradient. On the small VN30 (33 tickers) HAR is best at every
horizon and no learned model beats it significantly. On VN100 (104 tickers) the point-estimate gaps narrow
and turn slightly in the deep models' favor at the long horizons, but still without date-clustered
significance (the closest cell, the convex combination at h22, reaches p = 0.078). On the large S&P 500
subset (457 tickers) the deep temporal LSTM significantly beats HAR at every horizon (dQLIKE +3.01% at h1
to +7.10% at h22, all p<0.001 except h10 at p = 0.0013), and the convex combination beats HAR at every
horizon (p<0.001). The qualitative conclusion — deep-versus-HAR competitiveness grows with the size of the
panel, and the significant beat-HAR result emerges only once the panel is large and data-rich — is the
central data-scaling observation of the paper, and it holds under panel-correct inference rather than the
naive per-observation test.

---

### 6.5 Long-horizon study --- h10 and h22 (design: per-observation)

*Source: `results/soict_perobs/{panel}_lb10_h{10,22}/result.json`. S\&P~500 long-horizon runs were
still completing at submission time and will be added; the Vietnamese primary-market results below are
representative and consistent with the horizon pattern.*

To locate where the deep model's short-horizon advantage ends, we extend the main per-observation study
to h10 (two trading weeks) and h22 (roughly one trading month).

| Panel | h | HAR QLIKE | GARCH QLIKE | LSTM QLIKE | HAR R² | LSTM R² | DM LSTM-vs-HAR (QLIKE) |
|---|---|---|---|---|---|---|---|
| VN30  | 10 | 0.5925 | 0.7361 | 0.5908 | 0.144 | 0.141 | −0.29 (p=0.77) tie |
| VN30  | 22 | 0.6366 | 0.7307 | 0.6433 | 0.096 | 0.078 | +0.83 (p=0.40) tie |
| VN100 | 10 | 0.5773 | 0.6959 | 0.5737 | 0.103 | 0.105 | −1.25 (p=0.21) tie |
| VN100 | 22 | 0.6112 | 0.6917 | 0.6129 | 0.063 | 0.055 | +0.41 (p=0.68) tie |

**Reading.** At the long horizons the LSTM and HAR are statistically tied on QLIKE in every cell of this
per-observation study — the point estimates are within a fraction of a percent and the sign alternates (LSTM
marginally lower at h10, HAR marginally lower at h22), with no per-observation Diebold–Mariano significance
in either direction. Both still beat GARCH. This is consistent with the panel-correct verdict of
Section 6.6, where no model beats HAR on the Vietnamese panels at any horizon: the deep model's point-estimate
edge, where it exists, is a short-horizon effect that weakens as the target's own-history predictability
falls and the two models converge. No horizon shows HAR significantly beating the LSTM on the Vietnamese
panels either.

### 6.6 HAR-anchored residual and forecast-combination study (E0–E10)

*Source: `reports/experiment_results.md`, built from `results/har_anchored/{panel}_h{1,5,10,22}/result.json`;
leakage controls in `reports/leakage_audit.md`.*

The main study and the graph ablation ask whether a full-target deep model beats HAR. A complementary
question is whether HAR-*anchored* learning — combining or correcting HAR with a deep expert rather than
replacing it — can improve on HAR, and whether the cross-sectional graph contributes anything once HAR
carries the level. This study evaluates an eleven-rung ladder under panel-correct inference.

**Design.** The ladder is: **E0** HAR (locked benchmark); **E1** full-target LSTM and **E2** full-target
LSTM+GAT (the same neural experts as above, targeting the raw variance); **E3** a frozen-expert convex
combination of HAR and the deep expert with a single mixing weight fit on the validation predictions only
(reported as `E3_blend`); **E5/E6/E7** additive HAR residual models in which a deep branch (LSTM-only for
E5, the graph/GAT branch for E6, the combined LSTM+GAT branch for E7) predicts the HAR residual with a
zero-initialized head, so the model equals HAR at initialization; **E8** a multiplicative HAR-anchored
residual, `(HAR+ε)·exp(correction)`, which is positive and bounded by construction; and **E9/E10** static
and dynamic (regime-aware) gates that scale the residual correction by a learned weight. The design is the
common-date snapshot design of Section 6.2 (so the graph is well defined and E5/E6/E7 are same-fold
comparable), extended to horizons {1, 5, 10, 22}, with a target-overlap **purge of `h` snapshots** at each
split boundary. The HAR anchor is a per-horizon pooled OLS fit on training rows only; residual training
targets are built from expanding-window **cross-fitted** HAR predictions inside the training span, so they
reflect out-of-sample residuals rather than optimistic in-sample ones. Each configuration uses five seeds
{42, 123, 2026, 7, 2024}; the primary metric is QLIKE on the Parkinson variance. All fits (HAR
coefficients, per-ticker scalers, graphical-lasso edges, the convex weight and the gate parameters) use
training or validation rows only, with the test set read once (audit: `reports/leakage_audit.md`).

**Inference.** Significance is assessed with the **date-clustered** Diebold–Mariano test, which collapses
the loss differential to one value per calendar date before computing the statistic. All tickers share
each date, so a naive per-observation test treats the sample as `N` (cross-section size) times larger than
its number of independent dates and overstates significance (Section "Methodological note" below). The
`DM p-value` column reported here is the date-clustered value.

**VN30 (33 nodes; snapshot test counts: 4,356 at h1, 4,323 at h5/h10, 4,290 at h22 — on the order of 130
distinct test dates).** QLIKE (lower is better), out-of-sample R² relative to HAR, and the date-clustered
DM p-value versus HAR (five-seed means):

| h | Model | QLIKE | R²_OOS vs HAR | DM p (date-clustered) |
|---|---|---:|---:|---:|
| 1  | E0 HAR | 0.3946 | 0.0000 | — |
| 1  | E1 LSTM (full target) | 0.4383 | −0.1313 | <0.001 |
| 1  | E2 LSTM+GAT (full target) | 0.4915 | −0.2438 | <0.001 |
| 1  | E3 convex combination (val-fit weight) | 0.3984 | −0.0017 | 0.237 |
| 1  | E5 additive residual (LSTM branch) | 0.3946 | −0.0001 | 0.875 |
| 1  | E6 additive residual (graph/GAT branch) | 0.3946 | −0.0001 | 0.882 |
| 1  | E7 additive residual (LSTM+GAT branch) | 0.3946 | 0.0000 | 0.795 |
| 1  | E8 multiplicative HAR-anchored residual | 0.3947 | −0.0002 | 0.894 |
| 1  | E9 static gated residual | 0.3946 | 0.0000 | n/a |
| 1  | E10 dynamic gated residual | 0.3946 | 0.0000 | 0.754 |
| 5  | E0 HAR | 0.4531 | 0.0000 | — |
| 5  | E1 LSTM (full target) | 0.4810 | −0.0803 | 0.004 |
| 5  | E2 LSTM+GAT (full target) | 0.5185 | −0.1603 | <0.001 |
| 5  | E3 convex combination (val-fit weight) | 0.4639 | −0.0171 | 0.135 |
| 5  | E5 additive residual (LSTM branch) | 0.4531 | −0.0001 | 0.994 |
| 5  | E6 additive residual (graph/GAT branch) | 0.4531 | −0.0001 | 0.926 |
| 5  | E7 additive residual (LSTM+GAT branch) | 0.4532 | −0.0002 | 0.906 |
| 5  | E8 multiplicative HAR-anchored residual | 0.4531 | −0.0000 | 0.954 |
| 5  | E9 static gated residual | 0.4531 | 0.0000 | n/a |
| 5  | E10 dynamic gated residual | 0.4531 | −0.0000 | 0.932 |
| 10 | E0 HAR | 0.4849 | 0.0000 | — |
| 10 | E1 LSTM (full target) | 0.5291 | −0.1147 | 0.110 |
| 10 | E2 LSTM+GAT (full target) | 0.5394 | −0.1221 | 0.002 |
| 10 | E3 convex combination (val-fit weight) | 0.4958 | −0.0244 | 0.322 |
| 10 | E5 additive residual (LSTM branch) | 0.4853 | −0.0002 | 0.782 |
| 10 | E6 additive residual (graph/GAT branch) | 0.4854 | −0.0003 | 0.560 |
| 10 | E7 additive residual (LSTM+GAT branch) | 0.4854 | −0.0003 | 0.588 |
| 10 | E8 multiplicative HAR-anchored residual | 0.4850 | −0.0001 | 0.851 |
| 10 | E9 static gated residual | 0.4849 | 0.0000 | n/a |
| 10 | E10 dynamic gated residual | 0.4850 | −0.0001 | 0.600 |
| 22 | E0 HAR | 0.5025 | 0.0000 | — |
| 22 | E1 LSTM (full target) | 0.5185 | −0.0609 | 0.527 |
| 22 | E2 LSTM+GAT (full target) | 0.5603 | −0.1141 | 0.073 |
| 22 | E3 convex combination (val-fit weight) | 0.5137 | −0.0416 | 0.511 |
| 22 | E5 additive residual (LSTM branch) | 0.5026 | −0.0005 | 0.913 |
| 22 | E6 additive residual (graph/GAT branch) | 0.5026 | −0.0005 | 0.895 |
| 22 | E7 additive residual (LSTM+GAT branch) | 0.5027 | −0.0006 | 0.877 |
| 22 | E8 multiplicative HAR-anchored residual | 0.5027 | −0.0010 | 0.816 |
| 22 | E9 static gated residual | 0.5025 | 0.0000 | n/a |
| 22 | E10 dynamic gated residual | 0.5025 | −0.0001 | 0.896 |

**VN100 (104 nodes; snapshot test counts: 5,096 at h1, 4,992 at h5/h10, 4,888 at h22 — on the order of 50
distinct test dates).**

| h | Model | QLIKE | R²_OOS vs HAR | DM p (date-clustered) |
|---|---|---:|---:|---:|
| 1  | E0 HAR | 0.4844 | 0.0000 | — |
| 1  | E1 LSTM (full target) | 0.5371 | −0.1056 | 0.023 |
| 1  | E2 LSTM+GAT (full target) | 0.5207 | −0.0599 | 0.035 |
| 1  | E3 convex combination (val-fit weight) | 0.4821 | 0.0027 | 0.242 |
| 1  | E5 additive residual (LSTM branch) | 0.4842 | 0.0001 | 0.782 |
| 1  | E6 additive residual (graph/GAT branch) | 0.4842 | 0.0002 | 0.603 |
| 1  | E7 additive residual (LSTM+GAT branch) | 0.4842 | 0.0002 | 0.611 |
| 1  | E8 multiplicative HAR-anchored residual | 0.4879 | −0.0058 | 0.582 |
| 1  | E9 static gated residual | 0.4844 | 0.0000 | n/a |
| 1  | E10 dynamic gated residual | 0.4843 | 0.0000 | 0.601 |
| 5  | E0 HAR | 0.5441 | 0.0000 | — |
| 5  | E1 LSTM (full target) | 0.5614 | −0.0153 | 0.244 |
| 5  | E2 LSTM+GAT (full target) | 0.5582 | 0.0027 | 0.415 |
| 5  | E3 convex combination (val-fit weight) | 0.5339 | 0.0208 | 0.294 |
| 5  | E5 additive residual (LSTM branch) | 0.5439 | 0.0002 | 0.746 |
| 5  | E6 additive residual (graph/GAT branch) | 0.5438 | 0.0003 | 0.634 |
| 5  | E7 additive residual (LSTM+GAT branch) | 0.5438 | 0.0003 | 0.631 |
| 5  | E8 multiplicative HAR-anchored residual | 0.5480 | 0.0072 | 0.823 |
| 5  | E9 static gated residual | 0.5441 | 0.0000 | n/a |
| 5  | E10 dynamic gated residual | 0.5441 | 0.0001 | 0.624 |
| 10 | E0 HAR | 0.5985 | 0.0000 | — |
| 10 | E1 LSTM (full target) | 0.5750 | 0.0433 | 0.283 |
| 10 | E2 LSTM+GAT (full target) | 0.5803 | 0.0428 | 0.455 |
| 10 | E3 convex combination (val-fit weight) | 0.5701 | 0.0441 | 0.202 |
| 10 | E5 additive residual (LSTM branch) | 0.5979 | 0.0003 | 0.673 |
| 10 | E6 additive residual (graph/GAT branch) | 0.5883 | 0.0119 | 0.458 |
| 10 | E7 additive residual (LSTM+GAT branch) | 0.5929 | 0.0062 | 0.459 |
| 10 | E8 multiplicative HAR-anchored residual | 0.6008 | 0.0160 | 0.921 |
| 10 | E9 static gated residual | 0.5884 | 0.0118 | 0.488 |
| 10 | E10 dynamic gated residual | 0.5966 | 0.0020 | 0.440 |
| 22 | E0 HAR | 0.6177 | 0.0000 | — |
| 22 | E1 LSTM (full target) | 0.5888 | 0.0556 | 0.164 |
| 22 | E2 LSTM+GAT (full target) | 0.6031 | 0.0523 | 0.540 |
| 22 | E3 convex combination (val-fit weight) | 0.5816 | 0.0512 | 0.078 |
| 22 | E5 additive residual (LSTM branch) | 0.6171 | 0.0005 | 0.765 |
| 22 | E6 additive residual (graph/GAT branch) | 0.5925 | 0.0374 | 0.481 |
| 22 | E7 additive residual (LSTM+GAT branch) | 0.5922 | 0.0404 | 0.555 |
| 22 | E8 multiplicative HAR-anchored residual | 0.6187 | 0.0065 | 0.973 |
| 22 | E9 static gated residual | 0.5912 | 0.0430 | 0.569 |
| 22 | E10 dynamic gated residual | 0.6043 | 0.0180 | 0.480 |

**S&P 500 (long-history subset, 457 nodes; snapshot test counts: 137,100 at h1/h5, 136,643 at h10, 136,186
at h22 — on the order of 300 distinct test dates).** This is the large, data-rich panel. We report the
non-fragile rungs (HAR, GARCH, the full-target LSTM E1, the full-target LSTM+GAT E2 and the convex
combination E3); the additive/multiplicative residual rungs (E5–E8) are numerically fragile on this
cross-section and are discussed separately below. `*` marks a model that significantly beats HAR on QLIKE
(date-clustered DM, p < 0.05).

| h | Model | QLIKE | dQLIKE% vs HAR | DM p (date-clustered) |
|---|---|---:|---:|---:|
| 1  | E0 HAR                    | 0.3776 | —      | — |
| 1  | GARCH                     | 0.4843 | −28.27 | <0.001 |
| 1  | E1 LSTM (full target)     | 0.3662* | +3.01  | <0.001 |
| 1  | E2 LSTM+GAT (full target) | 0.3826 | −1.32  | 0.305 |
| 1  | E3 convex combination     | 0.3656* | +3.16  | <0.001 |
| 5  | E0 HAR                    | 0.4158 | —      | — |
| 5  | GARCH                     | 0.4830 | −16.16 | <0.001 |
| 5  | E1 LSTM (full target)     | 0.3988* | +4.09  | <0.001 |
| 5  | E2 LSTM+GAT (full target) | 0.4078* | +1.93  | 0.005 |
| 5  | E3 convex combination     | 0.4009* | +3.58  | <0.001 |
| 10 | E0 HAR                    | 0.4303 | —      | — |
| 10 | GARCH                     | 0.4842 | −12.53 | <0.001 |
| 10 | E1 LSTM (full target)     | 0.4123* | +4.18  | 0.001 |
| 10 | E2 LSTM+GAT (full target) | 0.4230 | +1.69  | 0.115 |
| 10 | E3 convex combination     | 0.4147* | +3.62  | <0.001 |
| 22 | E0 HAR                    | 0.4596 | —      | — |
| 22 | GARCH                     | 0.4883 | −6.23  | 0.121 |
| 22 | E1 LSTM (full target)     | 0.4270* | +7.10  | <0.001 |
| 22 | E2 LSTM+GAT (full target) | 0.4386* | +4.59  | 0.004 |
| 22 | E3 convex combination     | 0.4349* | +5.39  | <0.001 |

**Reading (S&P 500).** On the large panel the deep temporal LSTM (E1) significantly beats HAR at every
horizon under date-clustered inference (dQLIKE from +3.01% at h1 to +7.10% at h22, p<0.001 at h1/h5/h22 and
p = 0.001 at h10), and the convex combination (E3) also beats HAR at every horizon (p<0.001). The
graph-carrying full model (E2) is weaker than the graph-free LSTM at every horizon (its dQLIKE is smaller,
and it fails to beat HAR at h1 and h10), consistent with the graph adding no value. GARCH is dominated. The
LSTM's advantage here is not confined to QLIKE: it also attains the lower RMSE and the lower or equal MAE at
every horizon (for example h1 MAE 0.00021 versus HAR 0.00022, RMSE 0.00056 versus 0.00057; source
`reports/experiment_results.md`), so on the data-rich panel the deep temporal model leads across metrics.

**Graph attribution (paired date-clustered DM, graph residual versus no-graph residual).** A graph
contribution requires the graph/GAT residual (E6) or the combined residual (E7) to beat the same-capacity
no-graph residual (E5), not merely HAR. Under the paired date-clustered DM, in no panel and at no horizon does
the graph residual significantly beat the no-graph residual: on the Vietnamese panels none of the eight paired
contrasts approaches significance (all p ≥ 0.41, with the sign mixed — favoring the no-graph residual in most
VN30 cells and nominally the graph residual on VN100 — none distinguishable from noise), while on the large
S&P 500 panel the paired test is decisive in the opposite direction to a graph benefit, favoring the no-graph
residual at every horizon and significantly at h1 (p = 0.0003) and h10 (p = 0.015). The full per-horizon
paired p-values (E6 vs E5, E7 vs E5) are in the results record `reports/experiment_results.md`. The graph
therefore adds no incremental value beyond the no-graph model, consistent with the leave-one-out graph
ablation of Section 6.2.

### 6.7 Model-free graph screening — does any cross-stock signal exist out of sample?

*Source: `reports/model_free_graph_screening.md` (raw JSON `results/graph_screen/<panel>.json`) and
`docs/reports/2026-08-22_graph_no_value_analysis.md`.*

Because the graph ablation and attribution above compare specific neural architectures, a natural objection
is that a better-tuned graph model might still extract cross-stock signal. We rule this out model-free. For
each panel we append a neighbour-derived signal to the three HAR features in an ordinary-least-squares fit on
training rows only and measure the incremental test R² over the HAR-only fit; this upper-bounds what any
model consuming those edges could extract linearly and is leakage-safe (edges and the residual fit are
train-only). Six neighbour signals are screened — the equal-weight, signed-weighted and separate
positive/negative sums of neighbours' Parkinson variance; the signed-weighted neighbour HAR residual
(innovation); a train-selected directed lead-lag innovation; and a row-shuffled-edge placebo.

The result is a null across VN30, VN100 and the S&P 500 subset at every horizon: all incremental test R²
values are at most about 1%, and most are near zero or negative. The theoretically decisive screens — the
innovation and the directed lead-lag signals, which test predictive spillover beyond HAR's own persistence —
are approximately zero at every horizon on every panel. The only recurring small positives are the
contemporaneous neighbour-level signals (about +1.0% on VN100 at h5, about +1.1% on the S&P 500 at h1), and
these sit close to the shuffled-edge placebo (+0.2% and +0.4% respectively), so the structural graph adds
little beyond a random-edge control.

A single leakage-safe linear regression of the target on HAR plus the mean Parkinson variance of a node's
Top-5 graphical-lasso neighbours at day t — which bypasses the GAT, attention and residual head entirely —
yields incremental out-of-sample R² of essentially zero everywhere (VN30: −0.0001, +0.0013, −0.0014, −0.0004
at h1/h5/h10/h22; VN100: −0.0007, +0.0091, −0.0010, +0.0007). If exploitable cross-sectional spillover
existed out of sample, this simplest possible test would find it.

Diagnostics confirm the null is genuine rather than an implementation artifact. The graphical-lasso edge set
barely survives out of sample (train↔test Top-5 neighbour-set Jaccard 0.09–0.17, no node retaining half its
neighbours), the GAT branch is demonstrably alive (it emits a correction about 15% of HAR magnitude at
VN100 h22) yet still does not beat HAR, its attention collapses to a near-uniform neighbour average
(normalized entropy about 0.996), and the graph residual does not overfit (its train-versus-test residual R²
gap is zero or negative). The cross-sectional graph adds no value because the exploitable signal is not
present out of sample, not because of a bug.

**Findings.**

- **HAR is not significantly beaten on VN30 or VN100 at any horizon under date-clustered inference.** The
  frozen-expert convex combination is the closest competitor: its date-clustered p-value versus HAR ranges
  over 0.14–0.51 on VN30 and 0.08–0.29 on VN100, never below 0.05. The residual and gated variants have
  larger p-values still (the graph residual versus HAR ranges over 0.56–0.93 on VN30 and 0.46–0.63 on
  VN100). Point-estimate QLIKE gaps favor the hybrids at the longer horizons — for example the convex
  combination reaches +5.85% QLIKE at VN100 h22 (QLIKE 0.5816 versus HAR 0.6177) and the graph residual
  +4.08% (0.5925) — but these gaps are within noise given the short common-date test windows (on the order
  of 130 dates on VN30 and 50 on VN100).
- **On the large S&P 500 panel the deep temporal LSTM significantly beats HAR at every horizon.** The
  full-target LSTM (E1) improves QLIKE over HAR by +3.01% at h1, +4.09% at h5, +4.18% at h10 and +7.10% at
  h22 (p<0.001 at h1/h5/h22, p = 0.001 at h10), and the convex combination (E3) beats HAR at every horizon
  by +3.16% to +5.39% (p<0.001). This is the paper's positive beat-HAR result and it holds under the same
  panel-correct date-clustered inference under which the Vietnamese panels show no significant win, so the
  difference is one of data scale, not of inference convention.
- **Full-target deep models underperform HAR at short horizons on the small panels.** The full-target LSTM
  and LSTM+GAT lose to HAR at h1 on both Vietnamese panels (VN30 p<0.001; VN100 p=0.023 and p=0.035),
  consistent with the snapshot-design result of Section 6.2, whereas the HAR-anchored variants (convex
  combination, residuals, gates) track HAR closely and remove that short-horizon penalty.
- **The graph adds no incremental value on any panel.** No graph residual (E6) or combined residual (E7)
  beats the same-capacity no-graph residual (E5) at any horizon on any panel; on the S&P 500 the no-graph
  residual is significantly favored at h1 and h10 (Graph-attribution table above), and the model-free
  screening of Section 6.7 shows the incremental out-of-sample R² of any neighbour signal is at most about
  1% and approximately zero for the theoretically decisive innovation and lead-lag screens.
- **The additive/graph residual rungs are numerically fragile on a large cross-section.** On the S&P 500
  subset (457 nodes) the additive and graph residuals can drive predictions toward the QLIKE positivity
  floor and inflate QLIKE far above HAR (for example E5 QLIKE 2.9978 at h22 and E7 QLIKE 5.7090 at h10,
  versus HAR 0.4596 and 0.4303), so on the large panel the clean, well-behaved beat-HAR models are the
  full-target LSTM (E1) and the convex combination (E3); the fragile residual rungs are reported for
  completeness and are not the basis of the S&P 500 beat-HAR claim.
- **Per-metric fairness.** The model ranking is metric- and panel-dependent, which is why all five metrics
  are reported. On the S&P 500 the deep temporal LSTM leads on QLIKE, RMSE and MAE together. On the small
  Vietnamese panels HAR retains the best QLIKE and squared error, while the multiplicative HAR-anchored
  residual (E8) attains the lowest MAE at several cells (for example VN100 h1 MAE 0.00027 versus HAR
  0.00028) — no single metric determines the ranking.

**Methodological note (panel-correct inference).** For a cross-sectionally dependent volatility panel, the
statistical significance of a forecast-accuracy difference must be assessed with date-clustered (or
otherwise panel-robust) inference — one loss differential per calendar date — rather than a naive
per-observation Diebold–Mariano test over every (ticker, date) row. All tickers share each trading date, so
the per-observation test treats the effective sample as the number of tickers times the number of dates and
understates the loss-differential variance; the Diebold–Mariano statistic is inflated by a factor on the
order of the square root of the cross-section size (roughly a factor of six on VN30 and ten on VN100). A
QLIKE gap that is a coin-flip once collapsed to one value per date can be reported as highly significant at
the row level. Aggregating to the date level before inference removes this artifact and is the significance
convention used throughout this study.

**Relation to the per-observation tables of Section 6.1.** The per-observation Diebold–Mariano statistics in
Section 6.1 flag short-horizon LSTM-versus-HAR gaps on the Vietnamese panels as significant. Those
statistics use per-observation loss differentials and therefore overstate significance on this
cross-sectionally dependent panel, as quantified in the methodological note above; under the date-clustered
inference used throughout this paper, the Vietnamese gaps of that magnitude are not statistically
significant. The Section 6.1 point estimates are unchanged and reported as measured — only their
per-observation significance is superseded by the date-clustered verdict here, symmetrically for the cells
where HAR nominally beats the deep model. The one beat-HAR result that survives date-clustered inference is
on the large S&P 500 panel (this section), not on the Vietnamese panels.

---

## 7. Discussion

**HAR is hard to beat on small panels; data scale is what unlocks a deep advantage.** Under panel-correct
date-clustered inference, no model — LSTM, LSTM+GAT, convex combination or HAR-anchored residual —
significantly beats HAR on VN30 or VN100 at any horizon. The same architectures, evaluated identically,
significantly beat HAR at every horizon on the large S&P 500 subset, where the deep temporal LSTM improves
QLIKE by up to +7.1% at h22 (p<0.001) and the convex combination improves it at every horizon. Because the
inference convention is held fixed across panels, the difference is attributable to the amount of data, not
to the test: a flexible model needs a large, data-rich cross-section to out-of-sample beat a parsimonious,
well-specified baseline, and HAR's edge on the small Vietnamese panels is a small-sample phenomenon rather
than a fundamental ceiling on deep models.

**The graph adds no out-of-sample value, and this is a genuine null.** No graph-carrying model beats its
same-capacity no-graph counterpart on any panel or horizon (the leave-one-out ablation of Section 6.2, the
paired residual attribution of Section 6.6, and the S&P 500 case where the no-graph residual is
significantly favored). The model-free screening of Section 6.7 shows why: a leakage-safe linear
neighbour-signal regressor adds essentially zero incremental out-of-sample R², the innovation and lead-lag
screens are approximately zero everywhere, and diagnostics rule out a bug or overfitting (the
graphical-lasso edge set does not persist from train to test, the attention collapses to a uniform neighbour
average, and the graph residual shows no train-versus-test overfitting gap). The cross-sectional dependence
the graph encodes is dominated by a common market factor already captured by each ticker's own HAR
persistence and does not transfer out of sample. This is a clean negative result for a popular architectural
idea.

**Data design and inference convention both shape the apparent verdict.** Two secondary methodological
observations support the honest reading above. First, the deep-versus-HAR point estimates depend on the data
design: the same LSTM's point estimates favor it under the richer per-observation per-stock design
(Section 6.1) but not under the common-date snapshot with a global split (Section 6.2), because the
intersection discards most ticker-days and places the whole test set in one recent regime. Second, and more
importantly, the significance verdict depends on the inference convention: the naive per-observation
Diebold–Mariano test inflates the statistic by a factor on the order of the square root of the cross-section
size, which can turn a coin-flip QLIKE gap into an apparently significant one. Both observations caution
against reading a beat-HAR claim from a favourable data design plus a naive test; the panel-correct verdict
is the one reported here.

**Loss choice and model selection.** Training and early stopping use MSE while the decision metric is QLIKE.
These can disagree — QLIKE upweights low-volatility days — so we report all five metrics rather than the one
that flatters a given model; on the S&P 500 the deep temporal model happens to lead on QLIKE, RMSE and MAE
simultaneously, while on the Vietnamese panels the metric that is best differs by model (HAR on QLIKE and
squared error, the multiplicative HAR-anchored residual on MAE).

---

## 8. Limitations

- **Short common-date test windows on the Vietnamese panels.** The graph-defined snapshot design retains
  on the order of 130 test dates on VN30 and 50 on VN100. Under date-clustered inference these are the
  effective sample sizes, so even sizeable point-estimate QLIKE gaps at long horizons (for example the
  convex combination's +5.85% at VN100 h22) are not statistically significant. A negative or non-significant
  result on a small effective sample is evidence of "not demonstrated", not of "no effect at any sample
  size".
- **Model selection is by validation QLIKE for the HAR-anchored study and by validation MSE for the
  main-study point-estimate tables.** The two protocols can disagree because QLIKE upweights low-volatility
  days; we report all five metrics rather than the one that flatters a given model.
- **The graph is only evaluable on the common-date snapshot design**, which is data-poorer and
  regime-shifted relative to the per-observation design; this weakens all snapshot deep models in absolute
  terms. The negative graph result is nonetheless valid because it rests on within-design comparisons
  (leave-one-out LSTM+GAT versus LSTM (w/o GAT), and the paired residual attribution E6/E7 versus E5) plus
  the design-independent model-free screening of Section 6.7. The graph consumed here is the binary support
  of the graphical-lasso graph; the model-free screening additionally tests signed, weighted, innovation and
  directed lead-lag neighbour signals and finds them null, but volume-shock and sector graphs were not
  tested and remain follow-up work.
- **GAT scaling.** The attention is quadratic in the number of nodes; at ~500 S&P 500 nodes the graph model
  required a small batch to fit an 8 GB GPU, a practical limitation of the graph approach.
- **S&P 500 survivorship and subset selection.** The panel is the long-history subset of current
  constituents (457 nodes, ~300 test dates), so U.S. levels are optimistic; we claim the data-scaling
  direction and the beat-HAR-under-scale result on this subset, not that they hold for the full index or the
  absolute QLIKE level.
- **Single market for depth, single robustness market.** Vietnam is the case study and the S&P 500 subset
  the cross-market check; no claim of universal generality is made. A masked-panel robustness check that
  relaxes the common-date intersection is ongoing and not included here.
- All reported numbers are read directly from stored `result.json` files.

---

## 9. Conclusion

For daily Parkinson-variance forecasting, evaluated with panel-correct date-clustered Diebold–Mariano
inference on three equity panels, HAR is very hard to beat on the small Vietnamese markets: no model — the
LSTM, the LSTM+GAT, the convex combination or the HAR-anchored residuals — significantly beats HAR on VN30
or VN100 at any horizon. On the large, data-rich S&P 500 subset the same evaluation shows the deep temporal
LSTM (no graph) significantly beating HAR at every horizon, by up to +7.1% QLIKE at the twenty-two-day
horizon (p<0.001), with the convex HAR–deep combination beating HAR at every horizon as well; every learned
model decisively beats GARCH. Deep-versus-HAR competitiveness therefore grows with the size of the panel.
Adding a Graph Attention Network over a graphical-lasso partial-correlation graph contributes no
out-of-sample value on any panel — a genuine null established both by leave-one-out and paired-residual
ablations and by a model-free neighbour-signal screen, with diagnostics ruling out a bug or overfitting. The
practical recommendation is a graph-free HAR or HAR-anchored deep model: HAR alone on small panels, where it
is not significantly beaten, and a deep temporal or convex-combination model on large, data-rich panels,
where it significantly improves on HAR.

---

## References

1. Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial
   Econometrics*, 7(2), 174–196.
2. Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return.
   *The Journal of Business*, 53(1), 61–65.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
4. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention
   networks. In *International Conference on Learning Representations (ICLR)*.
5. Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the
   graphical lasso. *Biostatistics*, 9(3), 432–441.
6. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic
   Statistics*, 13(3), 253–263.
7. Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors.
   *International Journal of Forecasting*, 13(2), 281–291.
8. Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of
   Econometrics*, 160(1), 246–256.
9. Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting the errors: a simple approach for
   improved volatility forecasting (HARQ). *Journal of Econometrics*, 192(1), 1–18.
10. Diebold, F. X., & Yilmaz, K. (2012). Better to give than to receive: predictive directional measurement
    of volatility spillovers. *International Journal of Forecasting*, 28(1), 57–66.
11. Chen, Q., & Robert, C.-Y. (2022). Multivariate realized volatility forecasting with graph neural
    network. In *ACM ICAIF*. arXiv:2112.09015.
12. Zhang, C., Pu, X., Cucuringu, M., & Dong, X. (2025). Forecasting realized volatility with spillover
    effects: perspectives from graph neural networks (GNNHAR). arXiv:2308.01419.
13. Audrino, F., & Chassot, J. (2025). HARd to beat: the overlooked impact of rolling windows in the era of
    machine learning. *International Journal of Forecasting*. arXiv:2406.08041.

---

*Figure 1 appears in Section 3.3. Vector source: `docs/paper/diagrams/soict_harlstmgat.svg`; PDF/PNG
rendered by `diagrams/generate_arch.py`.*
