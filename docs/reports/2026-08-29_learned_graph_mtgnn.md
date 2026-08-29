# MTGNN learned-adjacency ablation (HNX volatility) — implementation + run status

Date: 2026-08-29. Baseline: `baselines/2026-08-29_learned_graph_ablation/`.

## Objective
Test whether a learned / adaptive graph adjacency (MTGNN, Wu et al. 2020, arXiv:2005.11650) is a better
edge for the HNX volatility model than no-graph LSTM, the shipped statistical vol→PK GAT edge, and the
static sector-GAT edge — under the same MaskedRichNet / HAR-X pipeline and folds, so only the edge differs.

## MTGNN graph-learning layer — faithful implementation
`code/mtgnn_graph.py::GraphConstructor` implements the paper's §3.1 graph-learning layer, verified against
the paper text (ar5iv.labs.arxiv.org/abs/2005.11650) and the official reference code
(github.com/nnzhan/MTGNN, `layer.py::graph_constructor`):

| Paper (Wu et al. 2020) | Implementation |
|---|---|
| Eq. (1) `M1 = tanh(α·E1·Θ1)` | `m1 = tanh(alpha * theta1(emb1(idx)))` |
| Eq. (2) `M2 = tanh(α·E2·Θ2)` | `m2 = tanh(alpha * theta2(emb2(idx)))` |
| Eq. (3) `A = ReLU(tanh(α·(M1 M2ᵀ − M2 M1ᵀ)))` | `adj = relu(tanh(alpha*(m1@m2.T - m2@m1.T)))` |
| Eq. (5)-(6) `idx = argtopk(A[i,:]); A[i,−idx]=0` | per-row `topk(k,1)` + `scatter_` mask (incl. the official `+rand*0.01` tie-break) |

- `E1,E2 ∈ R^{N×d}` learnable embeddings (`nn.Embedding`); `Θ1,Θ2` linear layers; `α` saturation rate,
  paper default **3**; subgraph size `k` default 20 (paper's large-graph default), capped at N; node-dim d=40.
- Adjacency is directed/asymmetric (the `M1M2ᵀ − M2M1ᵀ` subtraction), top-k sparse (≤k outgoing edges/node),
  and — as in the official code — carries no self-loop internally; the self-loop=1.0 is added by the wrapper
  to match the `WeightedGATLayer` convention.
- The independent-recompute unit test (`test_matches_paper_equations_independent_recompute`) pins the module
  output to the paper formula.

## Integration (edge-only ablation)
`code/mtgnn_graph.py::LearnedGraphNet` subclasses the delivered `MaskedRichNet`: it builds the adjacency
inside `forward` each step from the trainable `GraphConstructor`, applies the self-loop, and masks invalid
source nodes exactly as the training loop masks the fixed adjacencies — then feeds it to the SAME 2-hop
`WeightedGATLayer`. The LSTM temporal branch, 5 node features, masked panel, HAR-X anchor, per-ticker
scalers and QLIKE floor are inherited unchanged, so only the edge mechanism differs. MTGNN originally pairs
the learned graph with mix-hop propagation; a controlled edge-only ablation deliberately keeps the GAT
propagation identical across all four edge choices (documented in `design.md`) rather than confound the edge
with a different conv operator.

`code/run_learned_ablation.py` trains the four variants (`no_graph_LSTM`, `stat_GAT_vol2pk`, `sector_GAT`,
`learned_GAT_mtgnn`) on the same folds/seeds. The three fixed-edge variants reuse the delivered
`run_masked_rich.train_masked_rich` unchanged; the learned variant uses `train_learned`, which mirrors the
delivered `zscore_floor` training path (Adam, ReduceLROnPlateau, per-node standardized target, masked-MSE
loss, grad-clip, early stopping, train/val/test + learning-curve capture). The result JSON is assembled with
all five metrics (ensemble + per-seed mean±std), over/under-fit evidence (train/val/test + fit verdict +
learning curves), and date-clustered Diebold–Mariano tests: learned vs no-graph, learned vs stat-GAT,
learned vs sector-GAT (plus stat/sector vs no-graph).

## Verification (RAM-independent, completed)
- Unit + smoke tests: 18 passed. C0 line coverage = 100% and C1 branch coverage = 99% on the changed lines
  (single partial branch = the `out_dir=None` skip, above the 95% gate). ruff `--select F` clean.
- The graph-learning layer is asserted to be [N,N], directed/asymmetric, top-k sparse, self-loop, and
  differentiable with finite gradients (`test/test_mtgnn_graph.py`), and matched to the paper equations.
- The end-to-end harness was smoke-verified on a small real HNX universe (panel build + one training loop +
  gate-compatible result assembly) when RAM was briefly available.
- Adversarial 3-lens code review completed (`code_review/code_review_2026-08-29.md`): no CRITICAL/MAJOR
  findings.

## Empirical run status — PENDING (resource contention + coordinator pause)
An earlier batch of CPU-forced attempts (batch 256/96/48/32) were OOM-killed while the host was under
sustained RAM exhaustion from concurrent GPU jobs (a rogers_satchell sp500-h22 backfill and a sector-GAT
scale-up run holding ~34 GB via WDDM shared system memory). Once those jobs finished and the machine was
free, the run completed on the **GPU** (RTX 4060, single process, ~2 GB VRAM, ~95% util; the JSON `device`
field reads "cpu" only because that label is derived from the `CUDA_VISIBLE_DEVICES` env heuristic — the
compute ran on CUDA).

Configuration: HNX, horizon 1, lookback 10, 10 epochs (early-stopped), seeds {42, 123, 2026}, **N=154**
nodes, **60 028** scored test observations across **477** test dates. MTGNN graph: k=20, node-dim d=40,
α=3. QLIKE floor 1e-8 shared across all models.

## §3 Metric table (per-seed mean ± std, n=3 seeds, HNX h1)

| Model (edge) | MSE | RMSE | MAE | QLIKE | R² |
|---|---|---|---|---|---|
| no-graph LSTM | 1.375e-06 | 0.001173 ± 0.000001 | 0.000640 ± 0.000005 | **1.8138 ± 0.0025** | 0.2308 ± 0.0011 |
| LSTM + wGAT (stat vol→PK) | 1.379e-06 | 0.001174 ± 0.000001 | 0.000648 ± 0.000007 | **1.8134 ± 0.0032** | 0.2286 ± 0.0008 |
| LSTM + wGAT (sector) | 1.376e-06 | 0.001173 ± 0.000002 | 0.000640 ± 0.000005 | **1.8198 ± 0.0075** | 0.2299 ± 0.0029 |
| LSTM + wGAT (learned MTGNN) | 1.375e-06 | 0.001172 ± 0.000002 | 0.000641 ± 0.000004 | **1.8207 ± 0.0057** | 0.2309 ± 0.0029 |

The reported learned/graph numbers are the mean of seed-level metrics (the paper-reported quantity); the
seed-ensembled prediction gives slightly lower QLIKE (LSTM 1.8112, stat 1.8125, sector 1.8104, learned
1.8140) and is used only for the DM forecast. All four models span a QLIKE range of ~0.007 — within one
seed's standard deviation of each other.

## §4 Diebold–Mariano (date-clustered, QLIKE; positive mean_diff = learned worse)

| Comparison | DM p-value | favours | mean_diff |
|---|---|---|---|
| learned vs no-graph | 0.228 | no-graph | +0.00272 |
| learned vs stat vol→PK | 0.750 | stat | +0.00120 |
| learned vs sector | 0.201 | sector | +0.00332 |
| stat vs no-graph | 0.538 | no-graph | +0.00152 |
| sector vs no-graph | 0.737 | sector | −0.00060 |

Context (standalone sector-GAT run, `results/sector_gat_ablation/`, 5 seeds/15 epochs): sector vs stat
QLIKE DM p=0.50 — statistically indistinguishable, the bar for this comparison.

## §5 Fit verdict (over/under-fit evidence — all `status=ok`)

| Model | status | val→test QLIKE gap (rel) | train→test R² drop |
|---|---|---|---|
| no-graph LSTM | ok | −0.166 | −0.028 |
| LSTM + wGAT (stat) | ok | −0.163 | −0.022 |
| LSTM + wGAT (sector) | ok | −0.164 | −0.027 |
| LSTM + wGAT (learned MTGNN) | ok | −0.165 | −0.026 |

Every variant carries `train_metrics` + `val_metrics` + `fit_diagnostics` + `learning_curves` (train/val
MSE per epoch per seed); none is missing. Negative gaps/drops (test slightly better than val; test R² ≥
train R²) → no overfit and no underfit. Learned-graph early-stop best epochs across seeds: {3, 8, 6}.

## Verdict
**Null / negative result.** On HNX h1 the MTGNN learned adjacency does **not** significantly improve QLIKE
over no-graph (DM p=0.23), over the statistical vol→PK edge (p=0.75), or over the static sector edge
(p=0.20); the point estimates marginally **favour the simpler edges** (learned is nominally the worst on
QLIKE, though within one seed's std). No fixed edge beats no-graph either (stat p=0.54, sector p=0.74).
This matches and extends the sector-GAT finding: on this panel **no graph edge — learned or fixed — is
statistically distinguishable from no-graph.** Making the adjacency learnable (MTGNN) does not change that
conclusion; the added graph-learning parameters buy no out-of-sample QLIKE. A modest, honest, publishable
negative finding.
