# Sparse adjacency for the masked graph — design (plan)

## Scope
Additive edge-sparsification for the Track B masked graph. Default (`dense`) is unchanged;
`knn` / `threshold` trim edges on the PRESENT-node subgraph only. Node-sparsity (per-day
presence mask over the fixed ticker vocabulary) is untouched.

## Data flow
`build_masked_graph_manifest(pooled_manifest, store, adjacency, top_k, corr_threshold)`
→ per (split, target_date): assemble present nodes → `_masked_correlation_adjacency(price,
presence, mode, top_k, corr_threshold)`:
1. `corrcoef` over `price[present, :, 0]` (signed Pearson over present tickers).
2. If mode != dense → `_sparsify_correlation(corr, mode, top_k, corr_threshold)` on the
   present submatrix.
3. Place into the full [N,N] adjacency via `np.ix_(present, present)`; absent rows/cols stay 0.
4. Self-loop: `adjacency[i,i]=1` for every present i.
The intersection path (`build_graph_manifest` → `_correlation_adjacency`) takes the same
mode args for parity (all nodes present there).

## Sparsification semantics (`_sparsify_correlation`)
- Ranking / thresholding uses |corr| (edge strength); retained entries keep the SIGNED value
  (Track B keeps sign; GAT-hybrid reference used |corr| — we keep sign to preserve the
  existing masked model's signed-correlation message weights).
- `threshold`: zero every off-diagonal |corr| <= tau. Symmetric because corr is symmetric.
- `knn`: per row take the top-k by |corr| (excluding self), then **mutual** symmetrization —
  edge i-j kept iff j in top-k(i) AND i in top-k(j).
- Diagonal (self-loop) passed through unchanged.

### Design decision: mutual (AND) vs union (OR) k-NN
The GAT-hybrid reference (`src/lstm_gat_hybrid/graph_correlation.py`) symmetrizes by UNION
(add edge if either endpoint selects the other), which does NOT bound per-node degree — a hub
node can exceed k neighbours. The requirement here is an explicit, testable invariant:
"<= k off-diagonal nonzeros per present row AND symmetric". Only **mutual (AND)** k-NN
satisfies both simultaneously (each row keeps a subset of its own top-k → degree <= k;
symmetric by construction). We therefore use mutual k-NN and note the divergence from the
reference's union operator. Consequence: mutual k-NN is sparser than union; isolated nodes
still carry a self-loop, so message passing (softmax over nonzero adjacency + residual) is
well-defined for them.

## Hash / provenance
`_adjacency_hash_fields(mode, top_k, corr_threshold)` adds `adjacency_mode` (+ hyperparameter)
to the manifest `hashes` dict ONLY for non-dense modes. Dense adds nothing → dense manifest
hash byte-identical to prior runs (prior dense results remain comparable). Non-dense modes
change `content_hash`, so the graph-safe P3 checkpoint (which binds `graph.content_hash("train")`)
cannot be cross-used across modes. Leakage/scaler/provenance, graph-safe P3 boundary,
positivity floor + nonpositive<=1% gate, and seeding are all unchanged.

## Runner (`run_pilot.py`)
- CLI: `--adjacency {dense,knn,threshold}` (default dense), `--top-k` (8), `--corr-threshold`
  (0.7). Rejected on non-graph phases (would be silently ignored otherwise).
- Passes the config into both build functions; records `adjacency` + `edge_density`
  (avg/min/max off-diagonal nonzeros per present row) in the comparison payload.
- Graph G0/G1 training length: `_validate_graph_epochs` permits 1.._MAX_GRAPH_EPOCHS (50).
  Per user direction the GAT/GNN head is trained to convergence (15 epochs), exceeding the
  1-10 pooled experimentation cap (CLAUDE.md training policy: >10 requires explicit
  approval — granted for graph convergence).
- Per-epoch validation loss is recorded (`validation_losses`) and a train-vs-validation
  `learning_curve.png` is written per model, so convergence and any overfitting are
  observable rather than inferred from a single end-of-run point.

## Simplicity / Anti-Abstraction gates
- No new module or class; two internal helpers + three keyword args threaded through existing
  build functions. Uses numpy directly (no graph-library wrapper). Gates pass.

## Files touched
- `code/data.py`: `_sparsify_correlation`, `_validate_adjacency_config`,
  `_adjacency_hash_fields`, mode args on `_correlation_adjacency`,
  `_masked_correlation_adjacency`, `build_graph_manifest`, `build_masked_graph_manifest`.
- `code/run_pilot.py`: CLI args + guard, config pass-through, `_edge_density_stats`,
  `_graph_validation_loss`, `_plot_learning_curve`, per-epoch val loss, `_validate_graph_epochs`.
- `test/test_sparse_adjacency.py`: new (a)-(e) invariants + runner integration + epoch cap.
