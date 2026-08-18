# GNN Volatility Forecasting and the Conditions for Beating HAR: Literature Synthesis and Ranked Recommendations for VN30

Date: 2026-08-09. Scope: read-only literature and methods research. Objective: establish, from the
published record, when a graph neural network (GNN) or spatio-temporal model genuinely beats the
HAR / HARQ / GARCH baseline for volatility forecasting, what design choices produce that edge, and
which of those choices are realistic for this project's data (daily OHLCV, 33 VN30 tickers,
5-day-ahead Parkinson variance, static k-NN-8 correlation graph, PhoBERT news branch, per-ticker
gate). Every empirical claim is attributed; claims that could not be confirmed from a primary source
are marked `[UNVERIFIED]`.

---

## 1. Executive summary

1. **HAR is genuinely hard to beat at daily frequency with a limited information set.** The
   largest-scale controlled study (1,445 US stocks) finds tuned machine-learning models fail to beat
   a carefully re-estimated HAR when both use only realized volatility plus VIX, on QLIKE, MSE, and
   realized utility; the paper attributes most apparent ML "wins" to rolling-window/re-estimation
   choices rather than model class (Audrino & Chassot, IJF 2025).

2. **In the best-controlled GNN-vs-HAR study, the graph component was not what beat HAR.** On DJIA-30
   with a Model Confidence Set (MCS), multi-hop graph spillover gave "no clear advantage"; the gains
   came from (a) modeling nonlinearity and (b) switching the training loss from MSE to **QLIKE**, and
   only at horizons up to one week (Zhang, Pu, Cucuringu & Dong, IJF 2025 — "GNNHAR"). This directly
   corroborates this project's own null graph-ablation result.

3. **Every rigorous GNN/deep win over HAR uses realized/spot variance derived from intraday
   high-frequency data**, not daily range-based variance (GNNHAR; DeepVol; SpotV2Net; Boetti & Nunes
   2026). Universe size is not the discriminator (GNNHAR and SpotV2Net win on 30 assets); the
   discriminating input is **HF-derived RV + nonlinearity + a proper (QLIKE) loss**.

4. **No located study beats HAR on daily *range-based* (Parkinson-type) variance for a ~30-stock
   equity universe.** That is precisely this project's regime, and it is close to a structural ceiling
   for HAR. A broad average-case win over a well-specified HAR is not supported by the literature and
   would be an exception, not the expectation.

5. **The evidence-backed levers that are realistic here** (no new data required) are: **(i) QLIKE
   training loss**, **(ii) richer per-node HAR-family features derived from daily OHLC** — additional
   range estimators + overnight volatility, semivariance-style asymmetry proxies, a HARQ-style
   measurement-error term — and **(iii) directed volatility-spillover edges** in place of symmetric
   correlation. The levers this project structurally lacks are intraday-derived realized measures,
   large-universe cross-sectional commonality, and implied volatility.

---

## 2. Taxonomy of design choices in successful GNN-volatility work

### 2.1 Node features (ranked by strength of out-of-sample RV evidence)

| Rank | Feature family | Evidence it helps RV forecasting | Key source(s) |
|---|---|---|---|
| 1 | **Semivariance / signed measures** (RS⁺, RS⁻, signed jump ΔJ) as separate channels | Negative-return variation drives future vol most; helps OOS from 1 day to 3 months; explains ~10–20% more future-vol variation than RV-only | Patton & Sheppard, REStat 2015; Bollerslev realized semicovariation 2022 |
| 2 | **Realized quarticity / HARQ** (shrink daily-RV coeff when RV noisily measured) | HARQ "always improves over HAR" on S&P 500 / Dow, largest gains at daily horizon; WLS-HAR up to ~24% QLIKE reduction | Bollerslev, Patton & Quaedvlieg, J. Econometrics 2016; Clements & Preve, JBF 2021 |
| 3 | **Implied volatility / VIX** | Consistently valuable; can subsume past-RV at the monthly horizon; adding VIX to HAR "notably improves" across indices | Busch, Christensen & Nielsen, J. Econometrics 2011; Kambouroudis et al., J. Futures Markets 2021 |
| 4 | **Jump / continuous decomposition** (HAR-J, HAR-CJ, threshold bipower) | Improves OOS but state-dependent (concentrated on post-jump days); use threshold BV to avoid noise | Andersen, Bollerslev & Diebold, REStat 2007; Corsi, Pirino & Renò, J. Econometrics 2010 |
| 5 | **Range estimators + overnight volatility in HAR-RV-X** | In a HAR-RV-X framework on G7 markets (QLIKE/HMSE/MCS), range info improves RV forecasts; overnight most consistent; **simpler range estimators outperform complex ones** | Korkusuz, Kambouroudis & McMillan, Finance Research Letters 2023 (`docs/paper/`) |
| 6 | **Cross-asset / cross-market inputs** (own + market/neighbor RV, US RV/VIX for international) | Moderate; multi-hop spillover alone gives no clear advantage — nonlinearity + QLIKE loss is what pays | Zhang et al., IJF 2025 (GNNHAR) |
| 7 | **News / sentiment (deep learning)** | Conditional gains that concentrate at short horizons and on normal days; degrade during volatility jumps | Rahimikia & Poon, SSRN 2020; FinText, arXiv:2108.00480; Lei et al., J. Forecasting 2024 |
| 8 | **Technical indicators** | Weak/mixed for RV; help price/direction not variance; no general evidence nonlinear ML beats linear | "Does Anything Beat Linear Models?", J. Empirical Finance 2024 |
| 9 | **Order-book / microstructure** | Informative only at tick-to-intraday horizons; noisy, asset-specific; poor fit for daily RV | Ding, Cui & Zhang, IRFA 2022 |
| 10 | **Sector / fundamental characteristics** | Weakest for time-series RV; useful cross-sectionally, not for the forecast series | Ang, Hodrick, Xing & Zhang, J. Finance 2006 |

Cross-cutting note: statistical gains (MSE/QLIKE) frequently do not translate into economic
(vol-timing) value, and NN feature-importance studies find price features > volume features, recent
lags weighted highest, and importance is regime-dependent.

### 2.2 Edge / graph construction (ranked by OOS evidence over a correlation baseline)

| Rank | Edge construction | Evidence | Key source(s) |
|---|---|---|---|
| 1 | **Volatility-spillover graph (Diebold–Yilmaz, directional)** | Clearest head-to-head win over correlation: TemporalGAT with DY spillover adjacency reports ~25–40% lower MAFE/MSE vs correlation on 8 global indices; also beats GARCH | Diebold & Yilmaz, IJF 2012 / J. Econometrics 2014; Kumar, Umeorah & Alochukwu, arXiv:2410.16858 (2024) |
| 2 | **Realized-covariance / precision-matrix (graphical-LASSO) learned graph** | Learned precision-matrix graphs beat fixed sector weights, recover systemic hubs, statistically and economically significant OOS gains to 1 month; found **stable over time** | Zhang et al., GHAR, J. Financial Econometrics 2025 (nbae026) |
| 3 | **Spillover edge *features* on a GAT (vol-of-vol, co-vol)** | SpotV2Net attributes its gains over panel-HAR/XGBoost/LSTM specifically to spillover edge features, not plain correlation | Brini & Toscano, SpotV2Net, IJF 2025 (arXiv:2401.06249) |
| 4 | **Directed causal edges (transfer entropy / ETE, Granger)** | Qualitative ranking ETE ≥ TE ≥ Granger/Pearson: correlation too dense, Granger too sparse, TE/ETE the sweet spot; per-edge magnitudes `[UNVERIFIED]` | Kim et al., Fractal & Fractional 2025; Boetti & Nunes, arXiv:2606.03828 (2026) |
| 5 | **Economically-grounded relational edges (sector, supply-chain, ownership)** | Beat correlation when edges carry economic content; a supply-chain placebo test (reshuffle edges 100×, p=0.000) shows edge *content*, not topology, drives predictability | Feng et al., ACM TOIS 2019 (RSR); supply-chain KG, arXiv:2606.29290 (2026) |
| 6 | **Learned / attention adjacency (end-to-end, GAT/MTGNN-style)** | Dominates generic multivariate-TS benchmarks; **mixed, often no** advantage over a good correlation/static graph in equity volatility; hybrid (prior + learned) beats purely-learned | Wu et al., Graph WaveNet IJCAI 2019 / MTGNN KDD 2020; no clean finance-vol win located |
| — | **Pearson correlation (the ubiquitous baseline)** | Symmetric (cannot represent directional spillover) and over-dense (dilutes signal, invites overfitting); the economically wrong structure for volatility transmission | ACM Computing Surveys 2024; Kim et al. 2025 |

Static vs dynamic: "dynamic beats static" is commonly claimed (Kumar 2024; Feng 2019) but is not
universal — GHAR obtained significant OOS gains with a graph that was stable over time. Sparsity k:
generic node-classification work finds robustness for k∈[10,100] with a sweet spot ~30–80 (GARNET,
arXiv:2201.12741), but **no volatility paper pins an optimal k for a ~30-node universe**, where k is
a large fraction of N — a genuine gap directly relevant to a VN30-scale design.

### 2.3 Architecture

- The winning generic template is a **learned/adaptive graph + a temporal encoder (dilated TCN or
  LSTM/transformer) + graph convolution/attention, trained end-to-end**, with the forecast loss
  shaping the graph (MTGNN, KDD 2020; StemGNN, NeurIPS 2020; Graph WaveNet, IJCAI 2019; GTS, ICLR
  2021). End-to-end joint training consistently beats decoupled two-stage pipelines in the STGNN
  literature; two-stage is justified mainly for industrial scalability or warm-start.
- **For volatility specifically the marginal architectural wins are narrower than in traffic
  forecasting.** Gains concentrate in: nonlinear spillover terms (only ≤1 week), QLIKE training,
  economically-motivated (spillover-index) graphs, and short horizons (Zhang et al., IJF 2025).
  DeepVol's win comes from dilated causal convolutions consuming raw high-frequency data, not a graph
  (Moreno-Pino & Zohren, Quantitative Finance 2024). One financial STGNN paper explicitly warns
  spatio-temporal GNNs underperform on non-periodic stock data and scale poorly in node count (TCGPN,
  arXiv:2407.18519).
- A structural decomposition that recurs in HAR-GNN hybrids: **omit self-loops** so the HAR term owns
  own-history and the GNN owns pure cross-asset spillover (crypto GNN-HAR line).

### 2.4 Regime / frequency / universe conditions

- **Frequency is the dominant axis.** Every rigorous win uses intraday-derived RV or spot vol; in
  copper futures HAR wins daily RV on QLIKE by an order of magnitude, but the DL–HAR gap "can even be
  ignored" at hourly frequency (arXiv:2409.08356). Daily range-based variance is coarser still than
  daily RV computed from 5-minute returns.
- **Large-universe cross-sectional commonality is where NNs win.** Pooling hundreds–thousands of
  stocks plus a market-vol proxy lets NNs dominate linear/tree models for intraday RV, and models
  transfer to unseen stocks — the advantage comes from the cross-section, not any single small panel
  (Zhang, Zohren & Roberts, J. Financial Econometrics 2024).
- **Turbulent/crisis regimes** are where nonlinear-spillover and dynamic-graph advantages concentrate;
  they largely vanish in calm markets (Zhang et al. 2025; regime-dependent TemporalGAT, Mathematics
  2026).
- **Counter-evidence for balance:** parsimonious own-market HAR matches multivariate/spillover models
  OOS (Two-Step Regularized HARX, arXiv:2601.03146; Buccheri et al. 2021) — spillover networks add
  minimal incremental forecasting power beyond own-history, even where they matter for systemic-risk
  interpretation.

### 2.5 Crypto specifics (what transfers, what does not)

- **Transfers:** the methods — learned/adaptive graphs (MTGNN-style), QLIKE / nonlinear spillover
  terms, self-loop-free GNN-HAR decomposition, spillover-index edges; and the qualitative finding
  that cross-asset volatility connectedness carries predictive signal (EMGNN, Financial Innovation
  2025; multi-relational crypto attention, ESWA 2026).
- **Does not transfer:** crypto's structural enablers — 24/7 continuous trading, abundant
  high-frequency data, and unusually tight cross-asset coupling (BTC as dominant hub). Notably, even
  in crypto with all these advantages, the most direct realized-volatility study still finds **HAR
  beating ML at the daily/RV horizon** (cross-crypto RV connectedness, Financial Innovation 2025).
  Crypto results therefore do not imply a daily 30-asset equity GNN will beat HAR; they reinforce that
  the edge lives at high frequency and in tightly-coupled regimes.

### 2.6 Position of the local backbone-source papers

- **Sonani et al. (2025), "Stock Price Prediction Using a Hybrid LSTM-GNN Model," arXiv:2502.15813**
  (this project's backbone-idea source): predicts stock **price**, reports ~10.6% MSE reduction over
  standalone LSTM, with **no HAR baseline, no volatility target, and no DM/MCS test**. It motivates the
  architecture but is not evidence that the design beats HAR for volatility.
- **CryptoMamba (Sepehri et al., 2025, arXiv:2501.01010):** Mamba state-space model for Bitcoin
  **price**; no GNN, no HAR — a temporal-model reference, not a volatility-vs-HAR result.
- **Das et al. (2024), Decision Analytics Journal 10:100417:** survey of sentiment + GNN for stock
  prediction — useful for taxonomy framing, not an empirical HAR comparison.

---

## 3. Synthesis: what separates beats-HAR from ties-HAR

A GNN's edge over HAR is reliably present only when several of the following co-occur, most of which
this project's data does not supply:

| Condition | Present in beats-HAR studies | Available in this project (daily, 33 VN30, Parkinson variance)? |
|---|---|---|
| Intraday-derived realized/spot variance | GNNHAR, DeepVol, SpotV2Net, Boetti-Nunes | **No** — daily range-based variance only |
| QLIKE (not MSE) training loss | GNNHAR (the decisive lever) | **Yes** — currently trained on MSE; changeable |
| Nonlinearity at short horizons (≤1 week) | GNNHAR | Partial — model is nonlinear; horizon is 5-day |
| Directed volatility-spillover edges | Kumar 2024, SpotV2Net, GHAR | **Yes** — computable from the existing return/vol panel |
| Rich HAR-family node decompositions (semivar, RQ, jumps) | Patton-Sheppard, BPQ, ABD | Partial — semivar/RQ/overnight/range derivable from daily OHLC; true jumps need intraday |
| Large-universe cross-sectional commonality | Zhang-Zohren-Roberts (1,000s of stocks) | **No** — 33 tickers |
| Implied volatility / VIX input | Busch 2011, Kambouroudis 2021 | **No** — no liquid VN30 options surface |
| Turbulent/crisis sub-period focus | GNNHAR, regime TemporalGAT | Possible as a sub-period analysis |

The pattern is consistent with this project's measured null result: on a daily, small-universe,
range-variance panel, the graph and gate mechanisms are the wrong levers, and a well-specified HAR is
near a structural ceiling.

---

## 4. Ranked, actionable recommendations for this project

Effort scale: **S** = a few hours, **M** = 1–3 days, **L** = >3 days or a new pipeline. "Data OK"
means feasible with the current daily-OHLCV / 33-ticker data; "Needs new data" means it requires
inputs the project does not have.

| # | Recommendation | The change | Evidence it helps | Effort | Feasible on our data? |
|---|---|---|---|---|---|
| 1 | **Train and evaluate on QLIKE loss, not MSE** | Replace the MSE training objective with QLIKE (or WLS-HAR-style heteroskedasticity weighting); keep MSE only as a reported metric | The single decisive lever in the most rigorous GNN-vs-HAR study; up to ~24% QLIKE reduction from WLS weighting (Zhang et al. IJF 2025; Clements & Preve JBF 2021) | S–M | **Yes** — highest leverage, cheapest try |
| 2 | **Add range-estimator + overnight node features (HAR-RV-X)** | Augment the 3 HAR scales with Garman-Klass / Rogers-Satchell / Yang-Zhang range variances and an overnight (close-to-open) volatility term | Range info improves RV forecasts in HAR-RV-X on G7 (MCS), overnight most consistent, simpler estimators win (Korkusuz et al., FRL 2023 — local PDF) | S–M | **Yes** — all derivable from daily OHLC |
| 3 | **Add semivariance / signed-return asymmetry channels** | Feed proxies for good vs bad volatility as separate node channels (daily signed-return variation; from OHLC, an approximate signed range) | Most horizon-robust node feature; ~10–20% more future-vol variation explained (Patton & Sheppard, REStat 2015) | M | **Partial** — true HF semivariance needs intraday; a daily signed-return proxy is a defensible approximation to test |
| 4 | **Add a HARQ-style measurement-error term** | Include a daily realized-quarticity-analogue (or its range-based proxy) so the daily-vol coefficient shrinks when the estimate is noisy | HARQ "always improves over HAR", strongest at the daily horizon (Bollerslev, Patton & Quaedvlieg, J. Econometrics 2016) | M | **Partial** — exact RQ needs intraday; a range-based proxy is testable |
| 5 | **Replace correlation k-NN edges with directed volatility-spillover edges** | Build a Diebold–Yilmaz generalized-FEVD spillover matrix (VAR on the vol panel) as a directed adjacency; keep the graph but change what it encodes | Clearest published win of an edge construction over correlation (~25–40% lower error, Kumar et al. 2024); GHAR precision-matrix graphs beat sector weights | M | **Yes** — computable from the existing daily vol/return panel |
| 6 | **Sweep sparsity k and add a static-vs-dynamic edge check** | Grid k over the k-NN graph; compare static vs a rolling-window spillover graph, with DM tests per setting | No volatility paper pins optimal k for ~30 nodes (GARNET sweet spot ~30–80 is non-financial); dynamic-beats-static is not universal (GHAR stable graphs) | S–M | **Yes** — addresses a genuine literature gap for small N |
| 7 | **Omit GNN self-loops in the HAR-GNN hybrid** | Remove self-loops so HAR owns own-history and the GNN carries only pure cross-asset spillover | Structural decomposition used in HAR-GNN hybrids to isolate spillover from persistence (crypto GNN-HAR line) | S | **Yes** — one-line adjacency change |
| 8 | **Report a turbulent-subperiod / crisis breakdown** | Split OOS evaluation into calm vs turbulent regimes; report DM/MCS per regime | Nonlinear-spillover and dynamic-graph advantages concentrate in turbulent periods and vanish in calm markets (Zhang et al. 2025; regime TemporalGAT 2026) | S | **Yes** — reframes where any edge can appear |
| 9 | **Strengthen the statistical protocol** | Move to ≥5 seeds; report Diebold–Mariano and a Model Confidence Set across HAR / HARQ / GNN variants; match HAR and deep-model splits | MCS/DM is the standard that separates real wins from weak comparisons; n=3 is the paired-t minimum (project's own limitation) | M | **Yes** — improves credibility regardless of outcome |
| 10 | **Add implied-volatility input** | Feed VN30-index option-implied vol (or a regional IV proxy) as a node/market feature | IV is among the most valuable added features, can subsume past-RV at monthly horizon (Busch 2011; Kambouroudis 2021) | L | **Needs new data** — no liquid VN30 single-name options; index IV may be partial |
| 11 | **Move to intraday-derived realized variance** | Replace daily Parkinson variance with 5-minute realized variance (and true jumps/semivariance) | Every rigorous GNN/deep win over HAR uses intraday-derived RV (GNNHAR, DeepVol, SpotV2Net) | L | **Needs new data** — requires intraday VN30 tick/bar data the project does not have |
| 12 | **Enlarge the universe for cross-sectional commonality** | Expand from 33 tickers to hundreds (full HOSE/HNX or pooled markets) and add a market-vol proxy | NN wins come from large-cross-section commonality and transfer (Zhang-Zohren-Roberts 2024) | L | **Needs new data** — beyond the fixed VN30 scope |

---

## 5. Feasibility verdict

**Cheap, high-leverage tries (do first, no new data):** Recommendations 1, 2, 5, 6, 7, 8. Of these,
**QLIKE loss (#1)** is the best-evidenced single change and the cheapest, followed by **HAR-RV-X range
+ overnight features (#2)** — directly supported by a local PDF using this exact daily-OHLC input set
— and **directed spillover edges (#5)**, the one edge change with a clear published win over
correlation. These four target the exact levers the literature says separate beats-HAR from ties-HAR,
and all are computable from the existing daily panel.

**Worth testing but bounded by daily data (partial):** Recommendations 3 and 4 (semivariance and
HARQ). Their strongest forms require intraday data; daily range-based proxies are defensible to test
but should be reported as approximations, not the canonical measures.

**Protocol upgrade (do regardless of outcome):** Recommendation 9 (≥5 seeds, DM + MCS, matched
splits) raises the credibility of any conclusion, positive or null, and is what the literature uses to
distinguish real wins from weak comparisons.

**Needs data the project does not have (defer / out of scope):** Recommendations 10 (implied vol), 11
(intraday realized variance), 12 (large universe). These are the three structural enablers behind
essentially every published GNN-beats-HAR result. Their absence is the core reason a daily,
33-ticker, range-variance panel sits near a HAR ceiling.

**Candid overall assessment:** The literature offers no precedent for a GNN beating HAR on daily
range-based variance with a ~30-asset universe; the closest rigorous analogue (GNNHAR on DJIA-30)
found the graph component null and located its gains in QLIKE loss and nonlinearity. A broad
average-case win over a well-specified HAR in this project's regime is unlikely. The realistic,
defensible outcomes are: (a) a modest QLIKE/short-horizon improvement from the cheap levers above,
(b) an edge that appears only in turbulent sub-periods, or (c) a well-documented null that is itself a
publishable measurement result — the current paper's framing. A GNN-beats-HAR claim on daily
range-based VN30 data would be a novel result precisely because no prior work establishes it, and it
would require a clean MCS/DM demonstration to be credible. The higher-value near-term work is
tightening node features (QLIKE, range/overnight, spillover edges) and the statistical protocol,
rather than adding graph/gate complexity the evidence does not support at this data scale.

---

## 6. Sources

Local PDFs (in `docs/paper/`):
- Korkusuz, Kambouroudis & McMillan (2023). Do extreme range estimators improve realized volatility forecasts? Evidence from G7 Stock Markets. Finance Research Letters 55:103992.
- Sonani, Badii & Moin (2025). Stock Price Prediction Using a Hybrid LSTM-GNN Model. arXiv:2502.15813.
- Sepehri, Mehradfar, Soltanolkotabi & Avestimehr (2025). CryptoMamba: Leveraging State Space Models for Accurate Bitcoin Price Prediction. arXiv:2501.01010.
- Das, Sadhukhan, Chatterjee & Chakrabarti (2024). Integrating sentiment analysis with graph neural networks for enhanced stock prediction: A comprehensive survey. Decision Analytics Journal 10:100417.
- Liu, Ye & Yu (2022). Volatility Prediction via Hybrid LSTM Models with GARCH Type Parameters. Proceedings of Business and Economic Studies 5(6).
- Ouyang, Yang & Lai (2021). Systemic financial risk early warning of financial market in China using Attention-LSTM model. North American Journal of Economics and Finance 56:101383.

GNN-vs-HAR and rigorous benchmarks:
- Zhang, Pu, Cucuringu & Dong (2025). Forecasting realized volatility with spillover effects: Perspectives from graph neural networks. International Journal of Forecasting 41(1):377–397. https://www.sciencedirect.com/science/article/abs/pii/S0169207024000967 ; code https://github.com/chaozhang-ox/GNNHAR
- Zhang, Pu, Cucuringu & Dong (2025). GHAR. Journal of Financial Econometrics 23(2), nbae026. https://academic.oup.com/jfec/article/23/2/nbae026/7889003
- Audrino & Chassot (2025). HARd to Beat: The Overlooked Impact of Rolling Windows in the Era of Machine Learning. International Journal of Forecasting. https://arxiv.org/abs/2406.08041
- Moreno-Pino & Zohren (2024). DeepVol: Volatility Forecasting from High-Frequency Data with Dilated Causal Convolutions. Quantitative Finance. https://arxiv.org/html/2210.04797v3
- Brini & Toscano (2025). SpotV2Net. International Journal of Forecasting 41(3):1093–1111. https://arxiv.org/pdf/2401.06249
- Boetti & Nunes (2026). Network Time Series Models for Multivariate Volatility Forecasting. arXiv:2606.03828.
- Wade (2026). Do Better Volatility Forecasts Lead to Better Portfolios? Evidence from Graph Neural Networks. https://arxiv.org/abs/2605.19278 [abstract: the lowest-MSE, best-ranking, and best-Sharpe models differ; graph models add value only when the portfolio rule exploits cross-sectional structure]
- Son, Lee, Park & Lee (2023). Forecasting global stock market volatility... spatial-temporal graph-based model. Journal of Forecasting. https://onlinelibrary.wiley.com/doi/abs/10.1002/for.2975 [DM/MCS/HAR rigor `[UNVERIFIED]`]
- Chen & Robert (2022). Multivariate Realized Volatility Forecasting with Graph Neural Network. ACM ICAIF 2022; arXiv:2112.09015 [baseline rigor `[UNVERIFIED]`]
- Kumar, Umeorah & Alochukwu (2024). Dynamic / Regime-Dependent GNNs for volatility. arXiv:2410.16858 ; Mathematics 14(2):289 (2026) [HAR-as-benchmark `[UNVERIFIED]`; win shown over GARCH]
- Two-Step Regularized HARX (2026). arXiv:2601.03146. Buccheri et al. (2021) [spillover adds minimal incremental forecasting power].
- Copper futures DL vs HAR (2024). arXiv:2409.08356.

Node features:
- Corsi (2009). A Simple Approximate Long-Memory Model of Realized Volatility. J. Financial Econometrics 7(2):174–196.
- Bollerslev, Patton & Quaedvlieg (2016). Exploiting the errors (HARQ). J. Econometrics 192(1):1–18.
- Clements & Preve (2021). WLS-HAR. Journal of Banking & Finance. https://www.sciencedirect.com/science/article/abs/pii/S0378426621002417
- Patton & Sheppard (2015). Good Volatility, Bad Volatility. Review of Economics and Statistics 97(3):683–697.
- Bollerslev (2022). Realized semicovariation. https://www.researchgate.net/publication/363248108
- Andersen, Bollerslev & Diebold (2007). Roughing It Up. REStat 89(4):701–720. Corsi, Pirino & Renò (2010). Threshold bipower variation. J. Econometrics 159(2):276–288.
- Busch, Christensen & Nielsen (2011). J. Econometrics 160(1):48–57. Kambouroudis, McMillan & Tsakou (2021). J. Futures Markets. https://onlinelibrary.wiley.com/doi/full/10.1002/fut.22241
- Zhang, Zohren & Roberts (2024). Volatility Forecasting with Machine Learning and Intraday Commonality. J. Financial Econometrics 22(2):492. https://academic.oup.com/jfec/article/22/2/492/7081291
- Rahimikia & Poon (2020). SSRN 3707796. FinText (Rahimikia, Zohren & Poon). arXiv:2108.00480. Lei et al. (2024). FinBERT+MTGNN. J. Forecasting, https://doi.org/10.1002/for.3101.
- "Does Anything Beat Linear Models?" (2024). J. Empirical Finance. https://www.sciencedirect.com/science/article/abs/pii/S0927539824000598

Edges / architectures:
- Diebold & Yilmaz (2012). Better to give than to receive. IJF 28(1):57–66; (2014) J. Econometrics 182(1):119–134.
- Kim et al. (2025). Transfer-Entropy / Hurst-regime graph volatility. Fractal & Fractional 9(6):339. https://www.mdpi.com/2504-3110/9/6/339 [per-edge magnitudes `[UNVERIFIED]`]
- Feng et al. (2019). Temporal Relational Ranking (RSR). ACM TOIS 37(2); arXiv:1809.09441. Supply-chain KG placebo (2026). arXiv:2606.29290.
- Wu et al. (2019). Graph WaveNet. IJCAI; arXiv:1906.00121. Wu et al. (2020). MTGNN. KDD; arXiv:2005.11650. Cao et al. (2020). StemGNN. NeurIPS; arXiv:2103.07719. Shang, Chen & Bi (2021). GTS. ICLR; arXiv:2101.06861.
- TCGPN (2024). arXiv:2407.18519. GARNET (2022). arXiv:2201.12741.

Crypto:
- Cross-crypto RV / connectedness (2025). Financial Innovation. https://link.springer.com/article/10.1186/s40854-025-00881-x
- EMGNN crypto volatility (2025). Financial Innovation. https://link.springer.com/article/10.1186/s40854-025-00768-x
- Multi-relational crypto attention (2026). Expert Systems with Applications. https://www.sciencedirect.com/science/article/abs/pii/S0957417426023845

`[UNVERIFIED]` marks claims whose direction was confirmed but whose exact magnitude or baseline/test
rigor could not be extracted from a primary source; verify against the source PDFs before quoting
exact figures.
