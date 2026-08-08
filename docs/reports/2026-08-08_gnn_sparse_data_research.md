# Handling Sparse / Irregularly-Available Node Data in Financial Graph Neural Networks

Date: 2026-08-08
Scope: Literature and method research (2021–2026 sources), read-only on codebase.
Purpose: Determine how the field lets a cross-stock GNN train on more than the synchronized-date intersection, and what this implies for the VN30 volatility project and its paper Limitations.

---

## (a) Problem framing

The project builds a cross-stock GNN over 33 VN30 tickers. Message passing on a fixed 33-node graph requires all nodes present on the same trading date to construct the adjacency and aggregate neighbours. Trading-date availability is strongly unbalanced:

- Union across any stock: 4,989 dates.
- Intersection (all 33 present): 1,296 dates (26% of the union), capped by the newest listing (SSB, 1,299 dates) against the oldest (ACB/STB/VNM, ~4,887 dates).
- Consequence: the full 33-node graph exists on only ~1,296 dates (~900 train snapshots), while per-stock models can use up to 4,868 days × 33 observations.

The graph ablation in this project found message passing does not help (G1 ≈ G0). That result is confounded: the synchronized-date (intersection) requirement discards ~74% of the timeline before the GNN ever trains, so the graph model is evaluated in a low-data regime the per-stock baselines never face. The question is which method families let the GNN use snapshots on which only a subset of the 33 nodes is present, without introducing look-ahead / survivorship bias, and at what implementation cost.

The core insight from the literature is that the intersection requirement is not intrinsic to GNNs. It is an artifact of assuming a fixed node set. Most modern temporal-graph and missing-feature methods explicitly relax that assumption: they operate on a per-timestep available node set, or impute/propagate features so that absent nodes do not force dropping the whole snapshot.

---

## (b) Method families

### 1. Masked / availability-aware message passing (variable node set per snapshot)

What it is: instead of intersecting to a fixed node set, build the graph per timestep over only the nodes present on that date, and mask absent nodes and their incident edges. Aggregation is over present neighbours only. A binary missingness mask can also be concatenated to node features so the model knows which inputs are observed.

How it addresses the issue: directly. A snapshot on which only 20 of 33 tickers trade becomes a valid 20-node training example rather than being discarded. This is the cheapest structural change because it needs no imputation — only a per-snapshot adjacency and a presence mask.

Key real papers (verified):
- Rossi, Kenlay, Gorinova, Chamberlain, Dong, Bronstein. "On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs with Missing Node Features." Learning on Graphs Conference (LoG) 2022, PMLR 198. arXiv:2111.12128. Handles partially-available node features by Dirichlet-energy diffusion; reports ~4% mean accuracy drop even at 99% missing features. Directly relevant to missing node feature rows.
- You, Du, Leskovec. "ROLAND: Graph Learning Framework for Dynamic Graphs." KDD 2022. Treats layer-wise node embeddings as hierarchical recurrent states, letting any static GNN run over evolving snapshots with changing node sets.
- Cini, Marisca, Alippi. "Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks" (GRIN). ICLR 2022. arXiv:2108.00298. Message-passing RNN that conditions on available neighbours (see family 4).
- Marisca, Cini, Alippi. "Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations" (SPIN). NeurIPS 2022. arXiv:2205.13479. Sparse spatiotemporal attention that conditions reconstruction only on available observations, explicitly designed for the highly-sparse regime and avoiding error propagation from imputed values (see family 4).

Note: the "concatenate a binary missingness mask" trick (MIM-style augmentation) appears in recent GNN-with-missing-features work as a lightweight, assumption-light baseline; specific 2026 arXiv instances were seen in search results but are marked [unverified] here as their venue/authorship were not confirmed.

### 2. Dynamic / temporal graph networks (nodes appearing/disappearing)

What it is: models purpose-built for graphs that evolve, split into discrete-time (snapshot, DTDG) and continuous-time (event stream, CTDG) families. New listings and delistings are exactly node addition/removal.

How it addresses the issue:
- EvolveGCN (Pareja et al., AAAI 2020, arXiv:1902.10191): evolves the GCN's weights with an RNN rather than node embeddings, so the model is decoupled from any fixed node set and handles frequent node-set changes (inductive). EvolveGCN-H and -O variants.
- TGN (Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs," ICML 2020 Workshop on Graph Representation Learning; arXiv:2006.10637): per-node memory state initialized to zeros when a node first appears; keeps representations fresh for inactive nodes.
- TGAT (Xu, Ruan, Körpeoglu, Kumar, Achan, "Inductive Representation Learning on Temporal Graphs," ICLR 2020; arXiv:2002.07962): functional time encoding + self-attention; inductively infers embeddings for new and observed nodes.
- DySAT (Sankar et al., WSDM 2020): structural + temporal self-attention over snapshots; includes an incremental (IncSAT) variant for streaming/new nodes.
- JODIE (Kumar, Zhang, Leskovec, KDD 2019): coupled RNNs with a projection operator for interaction networks; continuous-time.

All verified. For a daily-frequency panel of 33 assets, the discrete-time snapshot models (EvolveGCN, DySAT, ROLAND) are the natural fit; the continuous-time event models (TGN, TGAT, JODIE) are designed for high-volume timestamped interaction streams and are heavier than this problem needs.

### 3. Inductive GNNs (GraphSAGE-style)

What it is: learn an aggregation function over sampled neighbourhoods rather than a per-node embedding table, so embeddings can be generated for nodes unseen at training time.

How it addresses the issue: a short-history stock (e.g. SSB) can join the graph and receive an embedding from its neighbours' features without the model having a trained embedding for it, so adding it does not require shrinking the panel to its history. Inductive capability is a prerequisite for families 1 and 2.

Key paper (verified): Hamilton, Ying, Leskovec. "Inductive Representation Learning on Large Graphs" (GraphSAGE). NeurIPS 2017. arXiv:1706.02216. Sampling + aggregation (mean/LSTM/pool aggregators); generalizes to unseen nodes and unseen graphs.

### 4. Missing-data / imputation on graphs

What it is: reconstruct absent node features from neighbours and history, then run the downstream GNN on a completed panel.

How it addresses the issue: fills the pre-listing / non-trading cells so a fixed 33-node graph can exist on more dates. Powerful but the highest-risk family for this project.

Key papers (verified):
- GRIN — Cini, Marisca, Alippi, ICLR 2022, arXiv:2108.00298. Bidirectional message-passing RNN for multivariate time-series imputation on graphs.
- SPIN — Marisca, Cini, Alippi, NeurIPS 2022, arXiv:2205.13479. Sparse spatiotemporal attention; conditions only on observed values; explicitly argues autoregressive imputers (GRIN-like) can propagate biased/imputed values and destabilize in the highly-sparse regime — directly relevant given VN30's 74%-sparse intersection.
- Feature Propagation — Rossi et al., LoG 2022, arXiv:2111.12128 (also family 1).

Risk (critical for this project): imputing a stock's volatility before its actual IPO date is fabricating history. Bidirectional imputers (GRIN) and any smoother that uses future observations to fill a gap create look-ahead leakage unless strictly causal (past-only) imputation is enforced. Imputing pre-listing values is also economically meaningless (the asset did not trade) and risks survivorship-style artifacts. If imputation is used at all, it must be causal and must not manufacture pre-listing observations — prefer masking (family 1) over imputation for the pre-listing region.

### 5. Node/edge dropout and subgraph sampling

What it is: randomly drop edges (DropEdge), nodes (DropNode/GRAND), messages (DropMessage), or sample subgraphs/neighbourhoods during training. Originally regularizers against overfitting and over-smoothing; mechanically they also train the model on partial node/edge sets.

How it addresses the issue: partially. These techniques normalize the model to variable-size neighbourhoods, which makes a masked/variable-node-set training scheme (family 1) more robust. They are a complement, not a standalone solution — they do not by themselves recover discarded dates.

Key paper (verified): Rong, Huang, Xu, Huang. "DropEdge: Towards Deep Graph Convolutional Networks on Node Classification." ICLR 2020. arXiv:1907.10903. Note the documented train/inference distribution-shift caveat: training on dropped subgraphs while inferring on the full graph can create an OOD gap — relevant if the graph node set differs systematically between train and test periods (which it does here, since early dates have fewer listed tickers).

### 6. Subset / panel-selection strategies

What it is: trade node count for temporal depth. E.g. use only the long-history subset (drop SSB and the other late listings) to get a smaller graph over far more dates; or staged/curriculum panels that grow the node set over time.

How it addresses the issue: increases usable snapshots by removing the tickers that cap the intersection. A panel of the ~25 stocks listed by the early boundary could push usable dates from ~1,296 toward the multi-thousand range.

Bias introduced (must be disclosed): selecting stocks by listing date is a form of look-ahead / survivorship-adjacent selection — the panel is conditioned on which firms had long histories, a fact known only ex post. The bias literature is explicit: constructing a historical universe from today's constituents ("preinclusion bias," Garcia & Gould 1993, cited in current surveys) and restricting to names observed throughout the window both inflate backtested performance. Sources surveyed: survivorship/look-ahead treatments in current backtesting and ML-in-finance literature (multiple 2024–2026 secondary sources; primary formal treatment: "Look-Ahead Benchmark Bias in Portfolio Performance Evaluation," arXiv:0810.1922). For a volatility-forecasting paper (not a trading backtest) the bias is milder but still must be stated as a Limitation.

### 7. Financial-GNN-specific handling

What it is: how stock-relation GNN papers actually treat asynchronous calendars and different listing dates in real markets.

Key papers (verified):
- Feng, He, Wang, Luo, Liu, Chua. "Temporal Relational Ranking for Stock Prediction" (RSR / Temporal Graph Convolution). ACM TOIS 37(2), 2019. arXiv:1809.09441. Foundational stock-relation GNN; NASDAQ/NYSE.
- Kim, So, Jeong, Lee, Kim, Kang. "HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction." arXiv:1908.07999, 2019. Selective aggregation over relation types.
- Hsu, Tsai, et al. "FinGAT: Financial Graph Attention Networks for Recommending Top-K Profitable Stocks." IEEE TKDE 35(1):469–481, 2023. arXiv:2106.10159. Builds fully-connected stock/sector graphs when no relations are predefined.
- Sawhney, Agarwal, Wadhwa, Derr, Shah. "Stock Selection via Spatiotemporal Hypergraph Attention Network" (STHAN-SR). AAAI 2021, 35(1):497–504.
- Sawhney et al. "Deep Attentive Learning for Stock Movement Prediction from Social Media Text and Company Correlations" (MAN-SF). EMNLP 2020.
- Cheng, Li. "Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks" (AD-GAT). AAAI 2021, 35(1):55–62.
- Chen, Robert. "Multivariate Realized Volatility Forecasting with Graph Neural Network" (Graph Transformer Network for Volatility). ACM ICAIF 2022. arXiv:2112.09015. Directly on realized-volatility forecasting with cross-stock relations on ~500 S&P names.

Important honest finding: across the financial-GNN literature surveyed, papers overwhelmingly sidestep the different-listing-date problem rather than solve it. The dominant practice is to fix a universe (e.g. index constituents) over a chosen window and use only stocks with complete data over that window — i.e. exactly the intersection approach the project already uses, with its survivorship caveat usually undiscussed. Volatility-spillover GNN papers (Chen & Robert 2022; and forecasting-with-spillover work such as the 2024 IJF article "Forecasting realized volatility with spillover effects: perspectives from graph neural networks" [venue verified: International Journal of Forecasting via ScienceDirect listing]) operate on large, long-lived, calendar-synchronized universes where the problem is negligible. No surveyed financial-GNN paper was found that explicitly trains on a per-day variable stock set with masking. This is a genuine gap and a defensible contribution angle for the VN30 paper: emerging-market panels (VN30) have severe listing-date imbalance that mature-market studies do not confront.

### 8. Data-efficiency compensators

What it is: methods that make GNNs work with fewer snapshots — self-supervised / contrastive pretraining, transfer, regularization.

How it addresses the issue: does not add dates, but reduces the penalty for having few. Pretrain the encoder self-supervised on the abundant per-stock series, then fine-tune the relational head on the scarce synchronized snapshots.

Key paper (verified): You, Chen, Sui, Chen, Wang, Shen. "Graph Contrastive Learning with Augmentations" (GraphCL). NeurIPS 2020. arXiv:2010.13902. Node/edge drop, feature masking, subgraph augmentations; strong in the semi-supervised / low-label regime. Related: GraphMAE (masked graph autoencoder) and BGRL are the standard generative/BYOL-style alternatives (names verified in survey results; treat exact venues as secondary).

---

## (c) Ranked recommendation for the VN30 problem

Objective: cheaply let the GNN use more than the 26% intersection, without leakage/selection bias, and disentangle "graph doesn't help" from "graph was starved."

### Rank 1 — Masked, availability-aware message passing (variable node set per day)
- What: build the adjacency per trading date over only the tickers present that day; mask absent nodes/edges; concatenate a binary presence mask to node features. No imputation.
- Why first: it is a data/collate change, not a new architecture. Reuses the existing GNN layer. It is the direct fix for the confound — the graph would train on the full ~4,900-date union (as variable-size graphs) instead of 1,296 dates.
- Effort: low–moderate. New per-snapshot adjacency builder + masked aggregation (mask attention/normalization over present neighbours) + masked loss. Mostly a dataset/dataloader change; the message-passing op needs to honour the mask.
- Leakage/selection risk: low. No future information used; no fabricated pre-listing data. Only real observations enter. The one caveat (family 5): the node set is smaller in early years — report train/test node-set composition and confirm the test period has the full 33.
- New model code vs data change: primarily a data/masking change, with a small masked-aggregation adjustment to the layer.
- Expected payoff: if G1 stays ≈ G0 even with 3–4× more snapshots and full temporal depth, the "graph does not help for VN30 volatility" conclusion becomes robust rather than confounded — which is itself a publishable, defensible result.

### Rank 2 — Long-history panel-selection ablation (as a controlled comparison, not the main model)
- What: additionally train the GNN on the subset of tickers listed before the early boundary (drop the late listings that cap the intersection), giving a smaller graph over many more synchronized dates.
- Why second: near-zero new code (just filter the ticker list) and it isolates the effect of temporal depth at the cost of graph width. Combined with Rank 1 it triangulates whether the null graph result is about data volume or about the graph signal itself.
- Effort: very low.
- Leakage/selection risk: moderate and must be disclosed — selecting by listing date is a look-ahead/survivorship-adjacent choice. Frame it explicitly as a diagnostic ablation, not the headline model, and cite the survivorship/look-ahead literature in the Limitations.
- New model code vs data change: pure data change.

### Rank 3 — Causal masking + optional self-supervised pretraining (only if Ranks 1–2 show the graph is genuinely data-starved)
- What: keep Rank 1's masking, and pretrain the per-stock temporal encoder self-supervised (GraphCL-style augmentations or a masked-autoencoder objective) on the full per-stock series, then fine-tune the relational head on synchronized snapshots.
- Why third: only worth the added complexity if evidence says few snapshots (not weak graph signal) is the bottleneck.
- Effort: moderate–high (pretraining loop + augmentations).
- Leakage/selection risk: low if augmentations and pretraining are causal/past-only.
- New model code vs data change: new model/training code.

Explicitly NOT recommended as a first move: full imputation of the panel (GRIN/SPIN) to force a dense 33-node graph. It is the highest-effort and highest-risk option here. Bidirectional imputation leaks future information; imputing pre-listing volatility fabricates data for assets that did not trade. SPIN's own motivation — that propagating imputed values in the highly-sparse regime destabilizes learning — argues against it at 74% sparsity. If imputation is ever attempted, it must be strictly causal and must never synthesize pre-listing observations; masking (Rank 1) achieves the data-recovery goal without these hazards.

### Comparison table

| Approach | Recovers >26% of timeline? | Leakage / selection risk | New model code vs data change | Effort |
|---|---|---|---|---|
| 1. Masked variable-node message passing | Yes — trains on the full union as variable-size graphs | Low (no future info, no fabricated data) | Data/masking change + small masked-aggregation tweak | Low–Moderate |
| 2. Long-history panel selection | Yes — more dates, fewer nodes | Moderate (listing-date selection = look-ahead/survivorship-adjacent) — disclose | Data change only | Very low |
| 3. Masking + self-supervised pretraining | Same dates as (1); better sample efficiency | Low if causal | New model/training code | Moderate–High |
| 4. Graph imputation (GRIN/SPIN) to densify | Yes — dense 33-node graph | High (bidirectional = future leak; pre-listing = fabricated data) | New model + imputation pipeline | High |
| Dynamic graph models (EvolveGCN/DySAT/ROLAND) | Yes — natively variable node set | Low–Moderate | New architecture | High |
| DropEdge/DropNode | No (regularizer; complements 1) | Low; watch train/infer OOD gap | Small code change | Low |

---

## (d) Implications for the paper's Limitations

The finding is defensible either way, and the framing should change depending on which experiments are run:

1. If only the current intersection result stands (no masked re-run): the Limitations must state that the graph ablation is confounded with data volume — the GNN saw ~1,296 synchronized dates versus up to ~4,868 per-stock observations because a fixed-node graph requires the date intersection, and ~74% of the timeline was discarded. The "message passing does not help (G1 ≈ G0)" claim must be explicitly qualified as holding only in this reduced-data regime, not as evidence that cross-stock structure is uninformative in general.

2. If the Rank 1 masked re-run is done: the paper can make a substantially stronger, cleaner claim. Either (a) the graph still does not help with full temporal depth and per-day node sets — a robust null result that removes the data-volume confound and is itself a contribution; or (b) the graph now helps, in which case the original null was an artifact of the intersection requirement and the masked formulation is the headline method. Either outcome strengthens the paper over the current confounded state.

3. Regardless: disclose the listing-date imbalance as a structural property of emerging-market panels (SSB ~1,299 vs ACB/STB/VNM ~4,887 dates) and note that mature-market stock-GNN studies (S&P/NASDAQ) largely avoid this problem by construction, which is why their fixed-universe practice does not transfer cleanly to VN30. If the Rank 2 panel-selection ablation is included, cite the survivorship / look-ahead bias literature and state that the long-history subset is a diagnostic, not a bias-free universe.

Suggested one-line honest positioning: the project confronts a listing-date-imbalanced emerging-market panel where the standard fixed-universe intersection either starves the GNN (confounding the ablation) or, via panel selection, introduces survivorship-adjacent bias; a masked variable-node-set formulation is the low-risk way to resolve the confound.

---

## Verified reference list

All entries below were confirmed against arXiv / proceedings / DBLP listings during this research (2026-08-08). None are fabricated. Items whose venue could not be independently confirmed are marked [unverified] and were not relied upon for core claims.

1. Hamilton, Ying, Leskovec. Inductive Representation Learning on Large Graphs (GraphSAGE). NeurIPS 2017. arXiv:1706.02216.
2. Feng, He, Wang, Luo, Liu, Chua. Temporal Relational Ranking for Stock Prediction (RSR/TGC). ACM TOIS 37(2), 2019. arXiv:1809.09441.
3. Kim, So, Jeong, Lee, Kim, Kang. HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction. arXiv:1908.07999, 2019.
4. Kumar, Zhang, Leskovec. Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks (JODIE). KDD 2019.
5. Pareja et al. EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs. AAAI 2020. arXiv:1902.10191.
6. Xu, Ruan, Körpeoglu, Kumar, Achan. Inductive Representation Learning on Temporal Graphs (TGAT). ICLR 2020. arXiv:2002.07962.
7. Rong, Huang, Xu, Huang. DropEdge: Towards Deep Graph Convolutional Networks on Node Classification. ICLR 2020. arXiv:1907.10903.
8. Sankar, Wu, Gou, Zhang, Yang. DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks. WSDM 2020.
9. Rossi, Chamberlain, Frasca, Eynard, Monti, Bronstein. Temporal Graph Networks for Deep Learning on Dynamic Graphs (TGN). ICML 2020 Workshop on Graph Representation Learning. arXiv:2006.10637.
10. You, Chen, Sui, Chen, Wang, Shen. Graph Contrastive Learning with Augmentations (GraphCL). NeurIPS 2020. arXiv:2010.13902.
11. Sawhney, Agarwal, Wadhwa, Derr, Shah. Stock Selection via Spatiotemporal Hypergraph Attention Network (STHAN-SR). AAAI 2021, 35(1):497–504.
12. Sawhney et al. Deep Attentive Learning for Stock Movement Prediction from Social Media Text and Company Correlations (MAN-SF). EMNLP 2020.
13. Cheng, Li. Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks (AD-GAT). AAAI 2021, 35(1):55–62.
14. Cini, Marisca, Alippi. Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks (GRIN). ICLR 2022. arXiv:2108.00298.
15. Marisca, Cini, Alippi. Learning to Reconstruct Missing Data from Spatiotemporal Graphs with Sparse Observations (SPIN). NeurIPS 2022. arXiv:2205.13479.
16. Rossi, Kenlay, Gorinova, Chamberlain, Dong, Bronstein. On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs with Missing Node Features. LoG 2022, PMLR 198. arXiv:2111.12128.
17. You, Du, Leskovec. ROLAND: Graph Learning Framework for Dynamic Graphs. KDD 2022.
18. Hsu, Tsai, et al. FinGAT: Financial Graph Attention Networks for Recommending Top-K Profitable Stocks. IEEE TKDE 35(1):469–481, 2023. arXiv:2106.10159.
19. Chen, Robert. Multivariate Realized Volatility Forecasting with Graph Neural Network (Graph Transformer Network for Volatility). ACM ICAIF 2022. arXiv:2112.09015.
20. Look-Ahead Benchmark Bias in Portfolio Performance Evaluation. arXiv:0810.1922 (formal treatment of look-ahead/benchmark bias).
21. Forecasting realized volatility with spillover effects: perspectives from graph neural networks. International Journal of Forecasting, 2024 (ScienceDirect listing; volatility-spillover GNN on a synchronized universe).

[unverified] MIM-style "concatenate binary missingness mask" GNN augmentation and SDA-GRIN / ImputeFormer extensions appeared in 2023–2026 search results but their exact venues/authors were not independently confirmed here; not relied upon for core recommendations.
