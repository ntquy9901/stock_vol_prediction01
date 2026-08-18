# GNN + X Hybrid Combinations in Financial Forecasting: Literature Review and Project Recommendation

Date: 2026-08-09
Scope: hybrid (GNN combined with some component X) architectures for financial forecasting — volatility AND price/return, equity AND crypto, 2020–2026. This report focuses on the COMBINATION/hybrid dimension; node-feature/edge-construction taxonomy and single-architecture feasibility for volatility are covered by a separate research track and are not re-derived here.

Method: fan-out web search (arXiv/SSRN/journals) across nine GNN+X families plus verification of the three papers the project already cites. Rows marked [PREPRINT] are non-peer-reviewed; [UNVERIFIED] means the paper's existence is confirmed but its results tables could not be read directly (paywall/403/PDF binary-extraction failure) and rest on abstract or search-snippet text. Every empirical claim carries a citation.

Grounding context (our setup): daily OHLCV, 33 VN30 tickers (small universe), 5-day-ahead Parkinson variance target. Current model G1 = parallel GNN (masked kNN-8 correlation graph, per-ticker input-independent gate) wrapping a per-node backbone of [3 HAR features + PhoBERT Vietnamese-news LSTM]. Current result: G1 ties HAR on continuous-error metrics (news helps QLIKE/RMSE; the graph adds no significant value over HAR — verdict B).

---

## 1. Master table of GNN+X hybrids

Combination -> task -> market -> frequency/universe -> evidence vs HAR/GARCH/baseline -> citation. "DM" = Diebold-Mariano test; "MCS" = Model Confidence Set.

| GNN + X | Mechanism | Task / target | Market | Freq / universe | Evidence vs HAR/GARCH/baseline | Citation |
|---|---|---|---|---|---|---|
| **Econometric: GNN-HAR (anchor)** | GNN generalizes GHAR: replaces linear neighbor aggregation with nonlinear graph-conv; HAR terms are node features | Realized volatility | US equity | Intraday 5-min RV; tens of stocks | Beats HAR & GHAR under **MCS+QLIKE**, BUT decomposition shows gain is from **nonlinearity + QLIKE loss**, NOT multi-hop neighbors; helps mainly short horizon (<=1wk) & turbulent regimes | Zhang, Pu, Cucuringu, Dong, *Int. J. Forecasting* 41(1):377-397, 2025; arXiv:2308.01419; code github.com/chaozhang-ox/GNNHAR |
| **Econometric: DCRNN-HAR** | Diffusion-conv GRU on **dynamic Diebold-Yilmaz spillover graph**, added **additively to HAR term**; masking to use asynchronous trading days | Realized volatility | 8 global equity indices | Intraday 5-min RV; 8 indices | Lowest MSE/MAE all 8 markets/horizons; **in MCS 48/48** vs 11-17 for baselines (SPX h=22 MAE 0.251 vs HAR 0.357). Beats HAR AND GNN-HAR | Chi et al., *J. Forecasting* 2026, DOI 10.1002/for.70081; arXiv:2409.15320 |
| **Econometric: GNHAR / Network-HAR** | Network edges = Granger-causal / Diebold-Yilmaz connectedness; HAR augmented by neighbor aggregation | Realized variance | 10 global equity indices | 5-min->daily RV; 10 assets; h=1..44 | Global-alpha GNHAR beats HAR **+14% MAFE (h=1) -> +43% (h=44)**, DM p<0.01, all network models in MCS, both HAR benchmarks excluded. Graph "essential, not marginal" | Boetti & Nunes, arXiv:2606.03828, 2026 [PREPRINT] |
| **Econometric: GHAR (origin)** | HAR augmented by **linear** neighbor aggregation over sector / graphical-LASSO graph | Realized covariance | US equity | Intraday RV/RCov | GHAR beats HAR-DRD parsimoniously; linear aggregation already captures most cross-sectional benefit | Zhang, Pu, Cucuringu, Dong, *J. Financial Econometrics* 23(2), 2025; SSRN 4274989 |
| **Econometric: GSP-HAR** | DY spillover matrix -> magnetic-Laplacian graph-signal filters fused into HAR (spectral domain) | Realized volatility | Global indices | Daily RV from intraday; 24-40 indices | Beats HAR-type AND GNN-HAR; DM p<0.05 "in most cases" (exact deltas [UNVERIFIED]) | Chi, Gao, Wang, arXiv:2410.22706, 2024 [PREPRINT] |
| **Transformer / attention: TemporalGAT (regime)** | Sequential LSTM->GCN->GAT; edges = **Diebold-Yilmaz spillover index**; regime-switching | Multi-horizon volatility | Global equity indices | Daily; 8 indices | Beats GARCH + ML + graph baselines, DM + bootstrap CI significant; **baseline is GARCH, not a strong HAR** | Kumar/Nkomo et al., *Mathematics* 14(2):289, 2026; arXiv:2410.16858 [PREPRINT/pub, deltas UNVERIFIED] |
| **Transformer: Graph-Transformer RV** | Graph-transformer block fusing LOB features + cross-sectional relations (attention-as-relation) | Short-term realized vol | US equity (S&P 500) | Intraday LOB; ~500 stocks | Beats non-graph benchmarks in tables; no numeric deltas / DM/MCS in abstract | Chen & Robert, ACM ICAIF 2022; arXiv:2112.09015 |
| **Transformer: THGNN** | 4-layer Transformer temporal encoder + edge-aware GAT over signed equity graph | 10-day stock-stock **correlation** (not vol) | US equity | Daily; ~500 S&P | MAE 0.307->0.230, Pearson 0.31->0.78 vs rolling baseline; strategy Sharpe 1.84 vs 0.65; bootstrap p=0.02. Target is correlation | Fanshawe, Masih, Cameron, arXiv:2601.04602, 2026 [PREPRINT] |
| **Dynamic/learned graph: MTGNN** | Graph-learning layer learns sparse adjacency from node embeddings; jointly trained with mix-hop GCN + dilated TCN; no predefined graph | Generic MTS forecasting | Traffic/energy (not finance) | Various incl. daily | SOTA vs STGCN/DCRNN; **no financial-vol / HAR test** — foundational recipe only | Wu et al., KDD 2020; arXiv:2005.11650 |
| **Dynamic/learned graph: SDGL** | Learns static + dynamic (self-attention) adjacency simultaneously | Generic MTS | Non-finance | Various | Beats MTGNN/AGCRN; no financial-vol/HAR test | Li et al., *Pattern Recognition* 2023 |
| **Dynamic graph benchmark (honest daily)** | GraphSAGE over rolling-correlation / sector / Granger graphs vs HAR & LSTM | Realized volatility + portfolios | US equity | Weekly; 465 S&P (2015-25) | HAR 0.0329, GNN-corr 0.0322, **GNN-corr+macro 0.0298** MSE; lift small, **no DM/MCS**, best-MSE/best-Sharpe are different models | Wade, arXiv:2605.19278, 2026 [PREPRINT] |
| **Temporal encoder: LSTM-GNN (project-cited)** | Parallel LSTM + GNN (Pearson-correlation graph), concatenated | Stock **price** (not vol) | US equity | Daily; 10 large-caps | MSE 0.00144 vs standalone LSTM 0.00161 (-10.6%); **no HAR/GARCH baseline**, no DM/MCS | Sonani, Badii, Moin, arXiv:2502.15813, 2025 [PREPRINT] |
| **Vol-of-vol GAT: SpotV2Net** | GAT with nodes=assets, edges=vol-of-vol spillover | Intraday spot volatility | DJIA (~30 stocks) | Intraday; ~30 | "Statistically significant gains" vs panel-HAR | Brini & Toscano, *Int. J. Forecasting* 2024; arXiv:2401.06249 |
| **Foundation model (no graph, key control)** | Zero-shot TSFMs (TimesFM/Chronos/Moirai/Lag-Llama/TTM) applied univariately | Realized volatility 1/5/22-day | US equity+FX+futures | Daily (5-min RV); 50 assets | **Zero-shot FMs mostly FAIL to beat Log-HAR**; only TTM wins ~1.3-1.8%, mostly calibration (Mincer-Zarnowitz). DM+MCS run | Brini, arXiv:2607.05291, 2026 [PREPRINT] |
| **Foundation + graph** | TSFM as node encoder inside a GNN | (finance) | -- | -- | **No published finance paper does this** — genuine white space, no evidence it helps | (gap; nearest analog THGNN uses from-scratch Transformer encoder) |
| **Decomposition + GNN: VMGCN** | Per-node VMD modes -> attention ChebNet graph conv, fused | Forecasting | Traffic (not finance) | Intraday | Methodological template; not finance | Javid et al., arXiv:2408.16191, 2024 [PREPRINT] |
| **Decomposition + GNN: VGC-GAN** | VMD subsequences -> multi-graph GCN generator + CNN discriminator | Stock price/movement | Equity indices | Daily | Claims beat GAN/GCN baselines; **high lookahead-leakage risk** (whole-series decomposition); no HAR/DM | *Expert Systems w/ Applications* 2023 [UNVERIFIED] |
| **Decomposition done right (no graph)** | Rolling-window CEEMDAN re-decomposition per step (leakage-safe) + DeepAR | Realized crypto volatility (probabilistic) | Crypto | High-freq | The correct causal-decomposition standard any graph attempt must meet | *Physica A* 2026, S0378437126001007 [UNVERIFIED] |
| **News/KG as node feature: MAN-SF** | Node init = bilinear(price + tweet embedding); GAT edges from Wikidata | Movement direction | US equity | Daily; ~88 | 60.8% acc / 0.195 MCC; beats StockNet & Adversarial-LSTM. Text = node feature (our style) | Sawhney et al., EMNLP 2020 |
| **News as EDGE: NIST-GNN/SCRG** | News co-occurrence company statistics -> cosine-similarity **edges** (news-derived structure) | Movement + portfolio | Equity | Daily | Sharpe +0.40 over benchmarks; no DM test. Directly tests news-as-edge | *Quantitative Finance* 2025, DOI 10.1080/14697688.2025.2548897 |
| **LLM-inferred dynamic edges** | ChatGPT infers latent inter-company relations -> daily-evolving edges | Movement | Equity | Daily | Higher annualized return + reduced volatility vs static-graph; LLM edges > fixed graph. No DM | Chen et al., arXiv:2306.03763, 2023 [PREPRINT] |
| **Hypergraph: HGTAN** | Two heterogeneous hypergraphs (industry + fund-holding); tri-level attention | Trend/movement | CN A-share, NASDAQ, NYSE | Daily; hundreds | Multi-relational hyperedges beat pairwise-correlation graph; no DM/MCS; not vs HAR | Li et al., *Pattern Recognition* 2022; arXiv:2107.14033 |
| **Multi-relational: HATS** | 75 Wikidata relation types + hierarchical attention that selectively aggregates edge types | Movement + index | S&P 500 | Daily; 500 | Selective multi-relation > single relation; geographic relations get lowest attention (not all edge types useful) | Kim et al., arXiv:1908.07999, 2019 |
| **Multi-relational: RSR / STHAN-SR** | Sector + Wikidata relations; temporal graph conv / spatiotemporal hypergraph + Hawkes attention | Ranking / return | NASDAQ, NYSE, TSE | Daily; hundreds-1000 | Relations improve ranking over price-only LSTM; ranking metric, not volatility | Feng et al., TOIS 2019; Sawhney et al., AAAI 2021 |
| **Crypto vol: EMGNN** | Evolving multiscale graph + crypto<->traditional-market cross-graph | Realized volatility | Crypto | Daily; multi-asset | Beats AR/ARIMAX/**HAR**, RF/LightGBM, LSTM/MTGNN under DM+MCS+QLIKE; value from **cross-market spillover** we lack | Zhou et al., *Financial Innovation* 11:87, 2025, DOI 10.1186/s40854-025-00768-x [deltas UNVERIFIED] |
| **Crypto vol (honest counter-example)** | Rolling-window dependency edges + temporal node model | Volatility / systemic risk | Crypto+banks | Rolling | **Plain strong LSTM BEAT the GNN**; dynamic graph helps only vs static graph in stress | *Int. J. Accounting & Econ. Studies* 2026 [UNVERIFIED] |
| **Generative: GAN/diffusion + GNN** | GCN cross-stock structure + WGAN/diffusion generator | Price scenarios / synthetic returns | Equity | Daily | Distributional fidelity claims; **no QLIKE + DM/MCS vs HAR/GARCH** found in this family | DASF-Net *JRFM* 18(8):417 2025; VGC-GAN; Sig-Graph-GAN arXiv:2605.22215 [UNVERIFIED] |
| **RL + GNN** | GNN encodes inter-asset relations into RL state; PPO/DDPG output portfolio weights | Portfolio (Sharpe/return) | Equity | Daily | All results are portfolio metrics (return/Sharpe/drawdown); **no forecast-accuracy, off-target for volatility** | e.g. WCG-RL *Finance Research Letters* 2024; NGDRL DASFAA 2024 [UNVERIFIED] |

---

## 2. Verification of the three project-cited papers

1. **Sonani, Badii & Moin (2025)** — "Stock Price Prediction Using a Hybrid LSTM-GNN Model." CONFIRMED, arXiv:2502.15813 (19 Feb 2025). Parallel LSTM + Pearson-correlation GNN on 10 US large-caps; only baseline is standalone LSTM (-10.6% MSE). It predicts **price, not volatility**, and tests **neither HAR nor GARCH** and no DM/MCS. Cite only as a generic LSTM+GNN precedent for the parallel-backbone idea — not as evidence a GNN beats HAR.

2. **Zhang, Pu, Cucuringu & Dong (2025)** — "Forecasting realized volatility with spillover effects: perspectives from GNNs." CONFIRMED, *Int. J. Forecasting* 41(1):377-397, DOI 10.1016/j.ijforecast.2024.09.002 (arXiv:2308.01419; code github.com/chaozhang-ox/GNNHAR). This is the anchor paper. Its load-bearing findings: (a) multi-hop graph neighbors alone give **no clear advantage** over HAR; (b) nonlinear spillover helps mainly at short horizons (<=1 week) and in turbulent (not calm) regimes; (c) the **largest, most consistent gain comes from training with QLIKE loss instead of MSE**. Evaluated with MSE, QLIKE, MCS on intraday 5-min RV.

3. **Chi et al. (2026)** — "Global Stock Market Volatility Forecasting Incorporating Dynamic Graphs and All Trading Days," *J. Forecasting*, DOI 10.1002/for.70081 (arXiv:2409.15320). Model = DCRNN-HAR: diffusion-conv GRU on a dynamic Diebold-Yilmaz spillover graph, added additively to the HAR term, with masking to use asynchronous trading days. Beats HAR and GNN-HAR decisively (in MCS 48/48). CAUTION: the "QLIKE-weighted loss" attribute the project's context file assigns to this paper was NOT confirmed in the arXiv text read (which reports MSE/MAE); the QLIKE emphasis belongs to the related Chi/Gao/Wang GSP-HAR paper. Verify before citing that specific detail.

---

## 3. Synthesis — which hybrid families most reliably beat HAR, and why

The families sort cleanly by evidence quality against a strong HAR baseline.

**Tier 1 — reliably beats HAR under DM/MCS, but on data we do not have.** The GNN+econometric family (GNN-HAR, DCRNN-HAR, GNHAR, GSP-HAR, SpotV2Net) is the only group that beats HAR with proper significance testing. Three consistent lessons emerge:
- The win is **not from "adding a graph."** The anchor paper's own ablation shows multi-hop correlation-graph neighbors give no clear advantage over HAR. The levers that actually move QLIKE/MSE are (i) **nonlinearity**, (ii) **QLIKE training loss instead of MSE** (the single most consistent, architecture-agnostic gain), and (iii) **directional spillover edges** (Diebold-Yilmaz connectedness / Granger causality / magnetic Laplacian), NOT static correlation.
- These wins are concentrated in **intraday realized-variance** targets on multi-index or multi-stock cross-sections. Intraday RV has far higher signal-to-noise than daily Parkinson variance, and the spillover edges are estimated from high-frequency series. The gain is largest at short horizons in turbulent periods — the conditions least like a 5-day-ahead daily target.
- A GNN-models-the-HAR-residual design is best realized as the **DCRNN-HAR additive decomposition** (HAR handles own-persistence, the graph module handles the cross-sectional spillover term), not a monolithic backbone.

**Tier 2 — architecturally borrowable, weak or off-target evidence.** Dynamic/learned-graph methods (MTGNN, SDGL) are mature but validated on traffic/energy, never on daily equity volatility vs HAR. Multi-relational/hypergraph methods (HGTAN, HATS, RSR, STHAN-SR) reliably beat single-graph baselines but on **ranking/direction** tasks, rarely with DM/MCS, and almost never against HAR. Transformer+GNN daily results are either GARCH-benchmarked (a weaker bar than HAR) or target the correlation matrix rather than volatility.

**Tier 3 — off-target for a volatility thesis.** Generative/diffusion+GNN solves price-scenario generation and synthetic-data fidelity, not variance point-forecast accuracy, and no paper in that family benchmarks QLIKE+DM/MCS against HAR/GARCH. RL+GNN optimizes portfolio Sharpe/return, a different problem entirely. Foundation-model node encoders in a graph are unpublished for finance, and the best-tested foundation-model volatility control (Brini 2026) shows zero-shot TSFMs mostly fail to beat Log-HAR even before a graph is added.

**The honest daily-data benchmark.** Wade (2026) — the closest published analog to our setup (daily/weekly US equity, learned-correlation GNN vs HAR) — finds the GNN beats HAR by only ~0.003 MSE (0.0329->0.0298) and only after adding macro features, with no significance test. This reproduces our own tie-with-HAR result. Our null result is consistent with the literature, not an implementation failure: on daily single-name data with a static correlation graph, HAR already captures the cross-sectional signal and the extra graph capacity has little to add.

**Candid bottom line on the central question.** The literature does suggest that **daily, small-universe GNNs rarely beat HAR regardless of the X**, unless the X supplies either (a) intraday realized-variance inputs, (b) a large/macro cross-section with directional spillover edges, or (c) cross-market information (crypto<->traditional). We have none of these. The minimal changes most likely to move the needle are the two architecture-adjacent, data-feasible levers below, not a new hybrid family.

---

## 4. Ranked recommendation for our project

### (a) Cheap, high-leverage, feasible on daily 33-ticker data — try these

1. **Switch training loss from MSE to QLIKE.** This is the single most consistent HAR-beating lever in the anchor GNN-HAR paper, it is architecture-agnostic (no data or structure change), and it directly targets the QLIKE metric that is already our headline. Lowest cost, best evidence. [Zhang et al. 2025]

2. **Replace the static kNN-8 correlation graph with a directional spillover graph** (Diebold-Yilmaz connectedness or Granger-causality edges), estimated from our own daily volatility panel — no external data needed. Every DM-significant HAR-beating volatility paper (GNN-HAR, DCRNN-HAR, GNHAR, GSP-HAR, TemporalGAT) uses spillover/causality edges; correlation graphs are the documented weakest construction. Caveat: DY spillover estimation on daily data with 33 tickers will be noisier than the intraday multi-index setups where it wins — treat as an experiment, not a guaranteed lift. [Boetti & Nunes 2026; Chi et al. 2026]

3. **Reframe the backbone as a HAR + graph-residual additive decomposition** (DCRNN-HAR style): let HAR carry own-persistence and the graph module model only the cross-sectional spillover term. This is the principled realization of "GNN models what HAR misses" and is cheap to prototype on existing features. [Chi et al. 2026]

4. **Make the graph dynamic/learned rather than fixed** (MTGNN graph-learning layer or a self-attention adjacency), if 1-3 are exhausted. Well-established recipe, but no daily-equity-volatility-vs-HAR evidence — moderate cost, speculative payoff. [Wu et al. 2020; Li et al. 2023]

5. **News-as-EDGE ablation (novelty, conditional on data).** If our Vietnamese news corpus contains multi-ticker articles, build news co-mention edges (NIST-GNN style) and run the edge-vs-node-feature ablation that, per this search, **nobody has published**. This is a legitimate paper-contribution angle regardless of outcome. Infeasible if articles are single-ticker. [NIST-GNN 2025; Chen et al. 2023]

6. **Add exogenous/macro regime features.** In the one honest daily benchmark (Wade 2026), macro features were the only lever that moved GNN MSE below HAR. Cheap to add. [Wade 2026]

### (b) Needs data we lack — do not pursue for this paper

- **Intraday realized-variance GNN-HAR / Graph-Transformer / SpotV2Net.** The strongest, MCS-significant HAR-beating results require 5-minute RV / limit-order-book data. We have daily OHLCV only. Off the table.
- **Large-universe multi-relational / knowledge-graph edges** (supply-chain, ownership/fund-holding, Wikidata relations). VN30 has thin Wikidata coverage, no public supply-chain graph, and sparse institutional-holding disclosures. Sector/ICB membership is the only relational data we cheaply have, and it is likely as inert as correlation.
- **Cross-market crypto-style spillover** (EMGNN). Its edge value comes from crypto<->traditional-market linkages absent in a VN30-only universe.
- **Foundation-model node encoders, generative/diffusion, and RL+GNN.** Foundation encoders are unpublished for finance and fail to beat HAR even standalone; generative and RL families target price/scenario/portfolio, not variance, and lack HAR/DM/MCS benchmarks. If a foundation encoder is tried at all, it should be framed as exploratory novelty (first TSFM-encoder GNN on an emerging market) using TTM (cheapest, only FM that beat HAR in Brini 2026), not as a performance bet.

### The minimal change most likely to move the needle

If only one change is made: **train with QLIKE loss and swap the correlation graph for a Diebold-Yilmaz spillover graph, evaluated with a Diebold-Mariano or MCS test.** These are the two levers the significance-tested literature credits for beating HAR, both computable from existing daily OHLCV, and DM/MCS is exactly the test that dissolves most unsubstantiated "GNN beats HAR" claims — including, plausibly, keeping our own conclusion honest.

---

## 5. Caveats and verification status

- Rows/claims marked [UNVERIFIED] rest on abstracts or search snippets (paywalls, HTTP 403, or PDF binary-extraction failures) — treat their numeric deltas as unconfirmed pending full-text access. This includes EMGNN deltas, GSP-HAR deltas, the MDPI TemporalGAT/*Mathematics* full text, DASF-Net, and several crypto entries.
- [PREPRINT] rows are non-peer-reviewed arXiv submissions.
- The project context file's attribution of a "QLIKE-weighted loss" to Chi et al. 2026 (*J. Forecasting*) was not confirmed in the arXiv text and should be verified before citing.
- Local PDFs under `docs/paper/` could not be page-rendered in this environment (poppler not installed); the project-cited papers were verified instead through their arXiv/publisher landing pages.
