# Optimal GNN architecture for HAR-Parkinson volatility forecasting — data analysis + research + prototype

Date: 2026-08-21
Scope: answers four questions for the `submission/soict_lstm_gat` HAR-LSTM-GAT model — (1) is GAT the
right GNN, (2) are node/edge constructions optimal, (3) would more data (S&P500, 500 nodes) make a GNN
worth it, (4) the single recommended architecture. Builds on `2026-08-21_gat_why_no_help.md` (attention
collapse, non-transferable glasso edges, over-smoothing, capacity overfit) — that diagnosis is not
re-derived here. Combines new data analysis (VN30/VN100/S&P500), a deep literature review, and a quick
CPU prototype of 6 architecture variants. Read-only; no data or committed model files were modified.

## TL;DR verdict

- **GAT is the wrong GNN — and the worst tested.** In the prototype a cheap fixed mean-aggregation
  (GCN-style) beats the multi-head GAT by ~0.06 QLIKE on VN30 (0.421 vs 0.481); message-passing on the
  LSTM output beats it further (0.415). GAT's damage is its extra 256-dim head fed a noisy graph signal
  it cannot weight (attention collapses to a mean anyway — prior report). If any graph is used, a
  **1-layer GCN / mean-aggregation trained with QLIKE loss** (the GNNHAR recipe, arXiv:2308.01419), or
  at least **GATv2** (arXiv:2105.14491), dominates the current GAT.
- **But no graph variant beats the no-graph LSTM or HAR** on any panel. Every one of the 5 graph
  variants tested loses to the plain LSTM with p<1e-4 on VN30.
- **More data helps the deep model, not the graph.** The GAT-minus-LSTM QLIKE gap collapses ~10× from
  VN30 (+0.072) to S&P500 (+0.007) — more nodes/data mostly *stop the graph from hurting* (regularise
  the overfit). On the 500-node S&P500 the no-graph LSTM matches HAR at h1 and **beats** HAR at h5
  (0.358 vs 0.368), while the GAT still adds nothing positive. Edge weight-transferability stays flat at
  ~0.15–0.20 regardless of panel size; the residual cross-sectional signal reaches only +0.005 OOS R²
  even at 500 nodes.
- **Recommended architecture: no learned graph.** Ship HAR as the primary model; the best *deep* option
  is a **HAR-LSTM trained with QLIKE loss plus a single market/sector-median-volatility exogenous
  regressor (HARX-style)** — this injects the one stable common factor as a scalar instead of a noisy
  256-dim smoothed vector. Keep the GAT in the paper only as a documented negative ablation.

---

## (A) Data analysis — is there exploitable, transferable cross-sectional signal?

All measurements on lookback 10, horizon 1, the submission's own loaders (`snapshots._load_panel`,
`edges.glasso_adjacency`), 80/10/10 chronological split. Script:
`_tmp_gnn_analysis/edge_stability.py`.

### A.1 Edge stability train-vs-test (does more data fix the graph?)

Top-5 neighbour-set Jaccard and full off-diagonal weight-correlation between the graph estimated on
TRAIN dates and the same estimator on TEST dates. Random-chance Jaccard = expected overlap of two
random size-5 sets from N−1 candidates (≈ 25/((N−1)·2−…)), shown for scale.

| panel | N | edge | Top-5 Jaccard | random-chance Jaccard | ratio to chance | weight corr (train vs test) |
|---|---|---|---|---|---|---|
| VN30 | 33 | glasso (current) | 0.150 | 0.085 | 1.8× | **0.193** |
| VN30 | 33 | correlation | 0.149 | 0.085 | 1.8× | 0.129 |
| VN30 | 33 | vol lead-lag | 0.104 | 0.085 | 1.2× | 0.055 |
| VN100 | 104 | glasso | 0.088 | 0.025 | 3.5× | 0.154 |
| VN100 | 104 | correlation | 0.113 | 0.025 | 4.5× | 0.175 |
| VN100 | 104 | vol lead-lag | 0.024 | 0.025 | 1.0× | −0.001 |
| S&P500 | 500 | glasso | 0.062 | 0.005 | 12.4× | **0.203** |
| S&P500 | 500 | correlation | 0.062 | 0.005 | 12.4× | 0.115 |
| S&P500 | 500 | vol lead-lag | 0.006 | 0.005 | 1.2× | 0.004 |

Reading:
- **Absolute Jaccard falls with N** (0.150 → 0.062) but this is mechanical — the random baseline falls
  faster (0.085 → 0.005). **Relative to chance, edge persistence actually rises with panel size**
  (1.8× → 12.4× for glasso): with 500 nodes the Top-5 partial-correlation neighbours are far more
  above-chance-stable than on VN30.
- **Weight-transferability is the scale-free measure, and it is flat at ~0.15–0.20** for glasso across
  all three panels (0.193 / 0.154 / 0.203). In *weight* terms the graph is only ~15–20% persistent no
  matter how many nodes/observations there are — **more data does NOT make the edge weights
  substantially more transferable**. This confirms and generalises the prior report's VN30 wcorr 0.19.
- **Vol lead-lag / spillover edges are essentially pure noise** out of sample (weight corr ≈ 0.00–0.06,
  Jaccard at chance level) — worse than the contemporaneous edge on every panel. A lead-lag / Granger
  edge would not rescue the model; this rules out that specific "better edge" hypothesis empirically.

### A.2 Residual cross-sectional signal beyond HAR

Per-node HAR-OLS forecast (fit on train), then regress the HAR residual on the node's own current pk +
the mean pk of its Top-5 correlation neighbours (contemporaneous cross-sectional info). Incremental R²
of predicting the residual, in-sample vs OOS:

| panel | N | residual-xsec R² (train) | residual-xsec R² (OOS) |
|---|---|---|---|
| VN30 | 33 | +0.014 | **−0.014** |
| VN100 | 104 | +0.009 | **−0.012** |
| S&P500 | 500 | +0.038 | **+0.005** |

Reading: on the small VN panels the neighbour signal is *negative* out of sample — it overfits the
train residual and hurts OOS (exactly the mechanism behind the GAT's damage). On the 500-node S&P500 it
crosses into marginally positive (+0.005 R², i.e. ~0.5% of residual variance). So **more data does
surface a faint transferable cross-sectional signal, but it is tiny** — nowhere near enough to justify a
256-dim GAT branch, and consistent with the committed S&P500 result where the graph neither clearly
helps nor badly hurts.

### A.3 Over-smoothing vs panel size

The model uses Top-K=5 edges per node, so node **degree stays ~5–8 regardless of N** (500-node graph is
not denser per node than the 33-node graph). Per-node smoothing intensity is therefore roughly constant
across panels — over-smoothing does not inherently worsen at 500 nodes. The dominant depth lever
(GNNHAR's MAD result: 3-layer most over-smoothed, worst) does not bite here because the model is a
single GAT layer. Over-smoothing is real but secondary (prior report: GAT output 0.65× dispersion, not
a collapse); the binding constraints are edge non-transferability (A.1) and near-absent residual
cross-signal (A.2).

---

## (B) Literature review — what architecture actually helps, and when

Full citations at the end. Key extracted, cited findings:

- **GNNHAR (arXiv:2308.01419, *Int. J. Forecasting* 2024)** — the closest prior art and the single most
  relevant paper. It is a **1-layer GCN on a linear-HAR backbone** (the GCN models only the
  cross-sectional spillover *on top* of a linear own-history term), **glasso edges**, trained with
  **QLIKE**. It beats HAR by only **~4% QLIKE / ~13% MSE at 1 day**, gains persist to 1 week, and
  **vanish / reverse at 1 month** (MSE 1.18× HAR). A **Diebold-Mariano test cannot distinguish 2-layer
  from 1-layer** (p≈0.75); 3-layer is actively worst (lowest MAD = most over-smoothed). Conclusion:
  **1-hop GCN, QLIKE loss; depth and attention are not where the gain is.**
- **GAT is over-parameterised / statically attentive for this setting.** GATv2 (arXiv:2105.14491) shows
  standard GAT computes only *static* attention (neighbour ranking unconditioned on the query node) and
  that **single-head GATv2 often beats 8-head GAT** — direct evidence GAT spends heads without buying
  expressiveness. Knyazev et al. (arXiv:1905.02850): on small/weak data "the effect of attention is
  negligible or even harmful." This matches the prior report's measured attention collapse (entropy
  0.999).
- **Learned/adaptive adjacency does not reliably beat fixed graphs.** Graph WaveNet's own ablation
  (arXiv:1906.00121): adaptive-only ≈ predefined-only (~1% difference) on data-rich traffic; MTGNN
  (arXiv:2005.11650) only on-par where a real graph exists. No evidence learned graphs beat fixed ones
  in the low-sample, high-noise financial regime — they add O(N·d) parameters that overfit.
- **Raw correlation edges are the least stable; cleaned/sparse (glasso) or static economic
  (sector/supply-chain) edges transfer better** (Laloux 1999 cond-mat/9810255, Plerou 2002
  cond-mat/0108023 — most correlation eigenvalues are noise; Millington-Niranjan 2020 partial-corr
  networks; Feng RSR arXiv:1809.09441 for static sector edges). The current model already uses glasso —
  i.e. the "best" statistical edge — and it still does not transfer (A.1).
- **Simple-beats-complex is the benchmark consensus** when node features are strong (Errica
  arXiv:1912.09893 — structure-agnostic baselines competitive with GNNs; GLNN arXiv:2110.08727 — MLP +
  distillation matches GNNs; DLinear arXiv:2205.13504 — a linear layer beats Transformers for TS). HAR
  lags are exactly such a strong own-history feature set.
- **The parsimonious rival to a graph is a low-dim common factor.** "Risk Everywhere" (Bollerslev et
  al., RFS 2018) — a single common volatility factor improves forecasts broadly and cheaply; Corsi HARX
  (2009) is the standard way to inject cross-sectional info as one regressor. This is the hard baseline
  a graph must beat, and usually does not.
- **The one clear counter-example, SpotV2Net (arXiv:2401.06249), needs intraday vol-of-vol *edge
  features*** — its ablation shows removing them collapses it toward HAR. Those features come from
  5-min data and are unavailable from daily OHLC/Parkinson, so its "graph helps" evidence transfers
  weakly here. Every domain win (GNNHAR, Chen-Robert arXiv:2112.09015, SpotV2Net) is on
  5-min realized variance, not daily Parkinson — the project's setting is genuinely under-explored.

---

## (C) Prototype — 6 architectures head-to-head

VN30 lb10 h1, single seed (42), ≤15 epochs, early-stop, CPU (GPU was 98% busy; VRAM free but kept off
it to be respectful). Script: `_tmp_gnn_analysis/prototype.py`. Variants: `lstm` (no graph),
`gat_glasso` (current), `mean_glasso` (fixed sym-norm mean aggregation over glasso, no attention),
`mean_corr` (same over correlation edge), `lstmnode_mean` (mean aggregation over the LSTM *output* =
message-passing on learned reps), `learnable` (MTGNN-style adaptive adjacency).

### VN30 (N=33)

| model | test QLIKE | test MSE | test R² | DM vs LSTM (QLIKE) | DM vs HAR (QLIKE) |
|---|---|---|---|---|---|
| **HAR** | **0.3946** | 2.15e-7 | +0.311 | — | — |
| LSTM (no graph) | 0.3980 | 2.22e-7 | +0.291 | — | +0.003 (p=0.2, tie) |
| lstmnode_mean | 0.4147 | 2.27e-7 | +0.274 | +0.017 worse (p=3e-6) | +0.020 worse |
| learnable (adaptive adj) | 0.4151 | 2.28e-7 | +0.271 | +0.017 worse (p=3e-5) | +0.021 worse |
| mean_glasso (GCN-style) | 0.4208 | 2.35e-7 | +0.247 | +0.023 worse (p=4e-7) | +0.026 worse |
| mean_corr | 0.4234 | 2.40e-7 | +0.232 | +0.025 worse (p=3e-9) | +0.029 worse |
| gat_glasso (**current**) | 0.4813 | 2.61e-7 | +0.167 | **+0.083 worse (p=1e-28)** | +0.087 worse |

Findings:
- **The current GAT is the worst model tested.** Every cheaper graph variant beats it: an explicit mean
  aggregation (0.421) beats GAT (0.481) by 0.06 QLIKE — confirming that since GAT attention already
  collapses to a mean, paying for the attention machinery (and its 256-dim head) only adds overfit.
- **Message-passing on the LSTM output (0.415) > raw-feature graph (0.421) > GAT (0.481)** — so the
  node construction (raw 3 HAR feats at day t) is suboptimal, but improving it is not enough.
- **No graph variant beats the no-graph LSTM (0.398) or HAR (0.395).** Adding cross-sectional structure
  hurts regardless of architecture, edge, or learnability. The adaptive/learnable adjacency does not
  rescue it (0.415, still significantly worse).

### S&P500 subsample (N=100) — data-size probe

Same variants, 100 S&P500 tickers, ~895 train / 112 test snapshots.

| model | test QLIKE | DM vs LSTM (QLIKE) |
|---|---|---|
| **HAR** | **0.3333** | — |
| LSTM (no graph) | 0.3615 | — |
| learnable (adaptive adj) | 0.3635 | +0.002 (**p=0.62, tie**) |
| lstmnode_mean | 0.3721 | +0.011 worse (p=2e-4) |
| gat_glasso (current) | 0.3723 | +0.011 worse (p=7e-3) |
| mean_glasso | 0.3737 | +0.012 worse (p=2e-3) |
| mean_corr | 0.3924 | +0.031 worse (p=4e-19) |

Data-size effect (prototype): the **GAT-minus-LSTM gap shrinks from +0.083 (VN30) to +0.011
(S&P500-100)**, and the learnable-adjacency graph becomes *statistically tied* with the no-graph LSTM
(p=0.62). The graph stops **hurting** with more data — but still does not **help**, and HAR still wins
this subsample. (The 100-ticker subsample understates the deep model; the authoritative full-panel
numbers are in section D.)

---

## (D) Verdict

### Authoritative panel-scaling evidence (committed 5-seed runs, `results/soict/*`)

| panel | N | HAR QLIKE | LSTM (no graph) | GAT | GAT − LSTM gap |
|---|---|---|---|---|---|
| VN30 h1 | 33 | 0.395 | 0.429 | 0.500 | **+0.072** |
| VN100 h1 | 104 | 0.484 | 0.520 | 0.530 | +0.009 |
| S&P500 h1 | 500 | 0.339 | 0.340 | 0.347 | +0.007 |
| S&P500 h5 | 500 | 0.368 | **0.358** | 0.370 | +0.012 |

The GAT-minus-LSTM gap collapses ~10× as the panel grows (VN30 → S&P500), and on 500 nodes the
**no-graph LSTM matches HAR at h1 and beats it at h5** (0.358 < 0.368). More data rescues the *deep
model*; it does not make the *graph* add value.

### The four questions

1. **Is GAT the best GNN choice?** No — it is the worst of the six architectures tested. GAT's
   multi-head attention is over-parameterised for a 3-feature, small-panel volatility target: the
   attention collapses to uniform (prior report; GATv2 theory), so it computes a mean but pays for a
   256-dim head that overfits. A plain 1-layer GCN / mean-aggregation beats it by ~0.06 QLIKE; if
   attention is wanted at all, use GATv2, not GAT. The literature's best domain recipe (GNNHAR) is a
   **1-layer GCN on a linear-HAR backbone with QLIKE loss** — not attention, not depth.

2. **Are the current node and edge constructions optimal?** No, and improving them is insufficient.
   Node = raw 3 HAR features at day t is suboptimal — message-passing on the LSTM's learned
   representation is measurably better (0.415 vs 0.481). Edges: glasso ≈ correlation (both only ~15–20%
   weight-transferable OOS), vol lead-lag is pure noise (weight corr ≈ 0). No edge choice is
   OOS-stable, and the residual cross-sectional signal the edges would carry is negative-to-+0.005 R²
   OOS. The node/edge design is not the bottleneck — the target has almost no transferable
   cross-sectional residual once own-history (HAR/LSTM) is removed.

3. **Would more data (S&P500, 500 nodes) make a GNN competitive?** Partially, and not enough. More
   nodes/data (i) shrink the graph's *damage* ~10× (overfit regularised), (ii) push the residual
   cross-signal from negative to a faint +0.005 OOS R², and (iii) make the *no-graph* LSTM competitive
   with / better than HAR (h5). But edge weight-transferability stays flat at ~0.15–0.20 regardless of
   size, and even on 500 nodes the GAT still adds nothing positive over the no-graph LSTM. So the small
   VN30 panel is not the whole problem — the graph is not the thing that more data fixes. **A GNN does
   not become worth it at 500 nodes for this daily-Parkinson target.** (Caveat: the domain papers where
   a graph clearly wins all use 5-min realized variance with intraday vol-of-vol *edge features*, which
   daily OHLC cannot provide — a different data regime, not just more of the same.)

4. **Single recommended architecture.** **No learned graph.** Ship **HAR** as the primary model. The
   best *deep* option is a **HAR-LSTM trained with QLIKE loss (not MSE) plus one exogenous
   market/sector-median-volatility regressor (HARX-style)** — inject the single stable common factor as
   a scalar, avoiding both the 256-dim capacity blow-up and the non-transferable per-pair graph. If the
   paper requires a graph model for narrative completeness, replicate the **GNNHAR recipe (1-layer GCN,
   glasso edges, QLIKE loss)** and report it as the honest near-tie/negative result it is, noting GATv2
   over GAT — and keep the current multi-head GAT strictly as a negative-result ablation.

---

## Reproduction

```
.venv_gpu_encode/Scripts/python.exe _tmp_gnn_analysis/edge_stability.py            # (A) edge stability + residual signal
CUDA_VISIBLE_DEVICES="" .venv_gpu_encode/Scripts/python.exe _tmp_gnn_analysis/prototype.py               # (C) VN30 6-variant prototype
CUDA_VISIBLE_DEVICES="" .venv_gpu_encode/Scripts/python.exe _tmp_gnn_analysis/prototype.py --sp500sub 100 # (C) S&P500-100 data-size probe
```

Committed panel-scaling numbers: `results/soict/{vn30,vn100,sp500}_lb10_h{1,5}/result.json`.

## Key citations

- Zhang, Pu, Cucuringu, Dong. "GNNs for Forecasting Multivariate Realized Volatility with Spillover
  Effects" (GNNHAR). arXiv:2308.01419; *Int. J. Forecasting* 2024. — 1-layer GCN + linear HAR + QLIKE;
  ~4% QLIKE at h1, vanishes at h22; depth does not help.
- Brody, Alon, Yahav. "How Attentive are Graph Attention Networks?" (GATv2). arXiv:2105.14491. — GAT is
  only statically attentive; single-head GATv2 beats 8-head GAT.
- Knyazev, Taylor, Amer. "Understanding Attention and Generalization in GNNs." arXiv:1905.02850. —
  attention negligible/harmful on small/weak data.
- Wu et al. "Graph WaveNet." arXiv:1906.00121. — learned adjacency ≈ fixed (~1% gap).
- Chen, Robert. "Multivariate RV Forecasting with GNN." arXiv:2112.09015. — Graph-Transformer, intraday,
  no significance testing.
- "SpotV2Net." arXiv:2401.06249. — GAT wins only via intraday vol-of-vol edge features (unavailable from
  daily OHLC).
- Bollerslev, Hood, Huss, Pedersen. "Risk Everywhere." RFS 2018. — a single common vol factor helps
  broadly and parsimoniously (the graph's cheap rival).
- Errica et al. arXiv:1912.09893; GLNN arXiv:2110.08727; DLinear arXiv:2205.13504 — simple/structure-
  agnostic baselines match GNNs when node features are strong.
- Laloux et al. cond-mat/9810255; Plerou et al. cond-mat/0108023 — most correlation structure is noise
  (edge instability). Full list in the research appendix of this investigation.
