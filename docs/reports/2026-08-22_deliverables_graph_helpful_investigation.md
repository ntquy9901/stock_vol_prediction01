# Investigation: why the graph "appears helpful" in deliverables_20260817

Question: `deliverables_20260817` (the earlier SOICT submission) is framed around a graph model and in
places the graph looks helpful, which seems to contradict the current study's "graph adds no OOS value".
This documents the graph approaches tried there and pinpoints exactly which lenses make the graph look
helpful — and shows the deliverable's own rigorous conclusion actually agrees with the current finding.

## 1. Graph approaches used in deliverables_20260817 (the list)
Edge definitions:
- **A1. Directed vol→PK lead-lag edge** (main): `A[j,i] = |corr(vshock_i(t), sqrt(PK)_j(t+1))|`, Top-K
  source stocks whose volume shock today leads target j's Parkinson vol tomorrow, estimated on TRAIN,
  frozen. Directed, predictive, cross-variable (volume→volatility). File:
  `baselines/2026-08-11_eda_gnn_baseline/code/edges.py`.
- **A2. Graphical-LASSO partial-correlation edge** (Top-5, market-factor-robust) — P5 alternative.
- **A3. Raw correlation edge** (baseline; re-encodes the market factor already in `market_pk`).
Architecture / training:
- **A4.** Self-written multi-head GAT (Velickovic-style), masked by `adjacency > 0` (support only),
  learned attention (`gat.py:27`); plus a residual message-passing variant `sum_k softmax(A[j,k])·feat[k]`.
- **A5.** GAT depth 1-hop vs **2-hop** (P2; 2-hop kept — better on VN).
- **A6.** 5 node features `[parkinson, har_weekly, har_monthly, market_pk, volume_zscore_20]` (vs the
  current study's 3 HAR features).
- **A7.** FULL 3-branch model: price-LSTM + GAT + gated PhoBERT news; leave-one-out ablation
  (minus_graph / minus_gate / minus_news / lstm_only).
- **A8.** Two training losses: **MSE** vs **QLIKE**; regime split (calm/turbulent); MAD depth diagnostic;
  HAR-X fair linear baseline; sub-period stability; Model Confidence Set.

## 2. Where the graph "appears helpful" — and why each is not robust
1. **Under MSE training loss (the main one).** The MSE-trained leave-one-out shows minus_graph
   significantly WORSE than FULL at h5/h10 — i.e. the graph contributes under MSE
   (`reports/2026-08-16_1600_gnnhar_p1p2p3_results_report.md`, Follow-up 1). **This benefit disappears
   under QLIKE** (the volatility-appropriate metric): the graph becomes neutral-to-harmful. So "graph
   helps" is largely a squared-error-loss artefact.
2. **On the MAE metric.** Paper Table 4 (DM, sign convention: positive = comparator lower loss): FULL vs
   minus_graph on absolute error is −5.20 (p<0.001) at h10 → FULL (with graph) better on MAE. But on
   QLIKE the same cell is +0.80 (ns), and at h1/h22 QLIKE is **+8.70 / +5.88 (p<0.001) meaning
   minus_graph is better → the graph HURTS QLIKE**. Graph helps only on MAE at h10.
3. **3-seed marginal false positives.** P5: the graphical-LASSO edge beat HAR and no-graph at h5 with
   p=0.048 on 3 seeds — but a 5-seed re-run erased it (p=0.286);
   `reports/2026-08-16_2330_p5_glasso_edge_vs_vol2pk_report.md` explicitly retracts it.
4. **Graph-containing variants beating HAR.** `minus_news` (= LSTM + graph, no news) beats HAR at h5
   (p=0.002), and FULL beats HAR at h1 / on calm days — so a model that CONTAINS the graph beats HAR.
   But the leave-one-out attributes the win to the LSTM, not the graph: `effect(graph)` on QLIKE is
   POSITIVE (graph hurts) at h1/h10/h22 and only −0.0013 at h5 (Table 2), and the HAR-X fair linear
   baseline shows the deep edge survives at **h1 only** and is driven by **LSTM nonlinearity**, not the
   extra features or the graph.
5. **Narrative framing.** The paper's title/abstract centre on a "directed-spillover graph-attention"
   model, which reads as graph-centric even though §1/§6 conclude (verbatim, paper line 121) "multi-hop
   cross-stock graph spillover gives no clear advantage."

## 3. The deliverable's own rigorous verdict = the current study's verdict
Under the primary metric (QLIKE), 5 seeds, MCS, and the HAR-X fair baseline, deliverables_20260817
concludes: HAR is best at h10/h22; a deep model beats HAR only at h1 (and h5 for a graph-free variant),
driven by **LSTM nonlinearity plus the `market_pk` and `volume_zscore` node features — not the graph**;
"no graph edge (vol→PK or graphical-LASSO) significantly beats HAR or the graph-removed model at any
horizon" (P5 conclusion). This is the same conclusion the current HAR-anchored study reached by a
different route (model-free screening S0–S5 + placebo across VN30/VN100/S&P 500: innovation/lead-lag/
volume neighbour signals add ≈0 incremental OOS R²).

## 4. What is genuinely different between the two studies (so the comparison is fair)
| Aspect | deliverables_20260817 | current HAR-anchored study |
|---|---|---|
| Edge | directed vol→PK lead-lag (+ glasso alt) | symmetric glasso Top-5 (also screened directed lead-lag S4, volume S5) |
| GAT depth | 2-hop | 1-hop |
| Node features | 5 (incl market_pk, volume_zscore) | 3 HAR |
| Graph consumes edge weight? | support-mask + learned attention (weight via residual MP variant) | support-mask only (V2: sign/weight discarded) |
| Deep-beats-HAR source | LSTM nonlinearity + extra node features (h1) | LSTM nonlinearity; on large S&P 500 the LSTM beats HAR +7% (h22) |
| Graph verdict | no clear advantage (QLIKE/MCS/HAR-X) | no OOS-transferable spillover (screening + null under date-clustered DM) |

Note the current study should adopt two things this deliverable did that strengthen attribution:
report **MSE/SE, MAE, AND QLIKE side by side** (the graph's apparent help is metric-specific), and always
include the **HAR-X fair same-feature linear baseline** (so a deep win is attributed to nonlinearity, not
to the extra node features). The vol→PK directed edge + 5 node features + 2-hop are the main design
levers the current study has not exactly replicated; the model-free S4/S5 screens are the cheap way to
check them before building a GAT, and they were null.

## 5. Bottom line
The graph is not actually helpful in deliverables_20260817 under the primary metric and robust settings;
it only "appears" helpful under (a) MSE loss, (b) the MAE metric at h10, (c) 3-seed marginal cells that
die at 5 seeds, and (d) attribution confusion where a graph-containing model beats HAR but the win is the
LSTM. Both studies agree: the beatable margin over HAR is temporal nonlinearity (plus, on large panels,
scale), not cross-stock graph spillover.
