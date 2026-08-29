# Design — MTGNN learned-graph ablation

## MTGNN graph-learning layer (faithful)
`code/mtgnn_graph.py::GraphConstructor` implements Wu et al. 2020 (arXiv:2005.11650) §3.1, verified
against the paper (ar5iv) and the official `nnzhan/MTGNN layer.py::graph_constructor`:

| Paper | Code |
|---|---|
| Eq. (1) `M1 = tanh(α·E1·Θ1)` | `m1 = tanh(alpha * theta1(emb1(idx)))` |
| Eq. (2) `M2 = tanh(α·E2·Θ2)` | `m2 = tanh(alpha * theta2(emb2(idx)))` |
| Eq. (3) `A = ReLU(tanh(α·(M1 M2ᵀ − M2 M1ᵀ)))` | `adj = relu(tanh(alpha*(m1@m2.T - m2@m1.T)))` |
| Eq. (5-6) `idx = argtopk(A[i,:]); A[i,−idx]=0` | per-row `topk(k,1)` + `scatter_` mask (with the official `+rand*0.01` tie-break) |

- `E1,E2 ∈ R^{N×d}` = `nn.Embedding`; `Θ1,Θ2` = `nn.Linear(d,d)`; `alpha` (saturation) default **3**.
- Subtraction ⇒ **directed/asymmetric** (if `A_uv>0` then `A_vu=0`).
- Top-k ⇒ each node keeps only its **k largest outgoing** weights. Default k=20, d=40 (MTGNN large-graph
  defaults); k is capped at N.
- **No self-loop inside** `graph_constructor` (matches official code) — the self-loop is added by the
  downstream GNN. `LearnedGraphNet.learned_adjacency()` overwrites the diagonal with **1.0**, matching the
  `adj_vol2pk`/`adj_corr` self-loop=1.0 convention consumed by `WeightedGATLayer`.

## Propagation choice (edge-only ablation, not full MTGNN mix-hop)
MTGNN originally pairs the learned graph with **mix-hop propagation**. For a controlled ablation where
*only the edge differs*, `LearnedGraphNet` subclasses the delivered `MaskedRichNet` and feeds the learned
adjacency into the **same 2-hop `WeightedGATLayer`** used by every fixed-edge variant. This keeps the LSTM
branch, GAT propagation, 5 node features, masked panel, HAR-X anchor, per-ticker scalers and QLIKE floor
identical — so a metric difference is attributable to the learned edge, not to a different conv operator.
(A pure MTGNN mix-hop implementation would confound the edge with the propagator and break the comparison.)

## Model wrapper
`LearnedGraphNet(MaskedRichNet)`:
- `forward(x, nmask)` mirrors the parent `forward(x, adj_b)` but builds `adj_b` internally:
  `base = learned_adjacency()` (self-loop) → `adj_b = base[None] * nmask[:,None]` (same valid-source-node
  masking the training loop applies to fixed adjacencies).
- The `GraphConstructor` parameters are optimised jointly with the rest of the net (as in MTGNN).

## Training loop
`code/run_learned_ablation.py::train_learned` mirrors the delivered `train_masked_rich` **zscore_floor**
path exactly (Adam, ReduceLROnPlateau, per-node standardized target, masked-MSE loss, grad-clip, early
stopping on val MSE, train/val/test + learning-curve capture). The only change: the net is
`LearnedGraphNet` and its adjacency is trainable-internal, so there is no numpy `adj`/`adj_batch`.
The three fixed-edge variants reuse the delivered `train_masked_rich` unchanged.

## Experiment
HNX h1, 10 epochs, seeds {42,123,2026}, batch 256, CPU-forced. Four variants under identical folds:
`no_graph_LSTM`, `stat_GAT_vol2pk`, `sector_GAT`, `learned_GAT_mtgnn`. Report ensemble + per-seed mean±std
for all 5 metrics; date-clustered DM for learned-vs-{no-graph, stat, sector} (plus stat/sector vs no-graph).

## Gate compatibility (SDD gates)
- **Simplicity**: reuse the delivered net/loop; new code = one faithful graph layer + one wrapper + one
  runner. No new abstraction beyond MTGNN's own.
- **Anti-abstraction**: import delivered modules read-only; subclass rather than fork.
- **Performance/batching**: batched `[B,N,seq,5]` tensors, batched `[B,N,N]` adjacency, no per-item loop;
  CPU forced only to avoid contending with the saturated live GPU (documented), GPU allowed via env knob.
- **Over/under-fit evidence**: result JSON carries `train_metrics`/`val_metrics`/`metrics`(test) +
  `fit_diagnostics` + `learning_curves`. Result keys use `LSTM` (no-graph) and `LSTM_wGAT_vol2pk` (stat) so
  the pre-push overfit gate finds its required learned-model evidence; the learned/sector models add extra
  keys.
