# Design — Heterogeneous 2-relation GNN probe (HNX h1)

## Data flow
```
screened HNX tickers (EFA.screened_tickers)
  -> EFA._write_estimator_processed(parkinson)            [read-only writer]
  -> MR.build_masked_rich(files, price_dir, lookback=10, horizon=1)
        -> D: X_{tr,va,te} [n,N,seq,5], nmask/tmask, y, HAR/HAR-X, per-node target scaler, d_va/d_te
  -> hetero_edges.build_relation_adjacencies(load_close_wide(D.tickers), cutoff=D.d_va[0],
                                             corr_thresh=0.25, lift_thresh=1.2)
        -> adj_lin [N,N], adj_nl [N,N]  (train-only, per-relation Min-Max [0,1], self-loop=1), diag
  -> hetero_edges builds the SQUASHED-at-lowered adjacency via CL.build_corrlift_adjacency(...,0.25,1.2)
  -> train 3 variants on the SAME folds/seeds:
       no_graph_LSTM      = RMR.train_masked_rich(D, cfg, seed, use_graph=False, adj=D.adj_vol2pk)
       squashed_lowered   = RMR.train_masked_rich(D, cfg, seed, use_graph=True,  adj=squashed_lowered)
       hetero_2rel_GAT    = train_hetero_rich(D, cfg, seed, adj_lin, adj_nl)
  -> RMR helpers: _pred_dict/_ens/_metrics/seed_metric_stats/_split_metrics/_ens_split/_dm_all/OF.classify_fit
  -> results/hetero_graph_ablation/hetero_graph_ablation_hnx_h1.json
```

## Files (code/)
- `hetero_edges.py` — build the TWO per-relation adjacencies (0.25/1.2) with per-relation train-only Min-Max
  normalization; edge-density + degree diag. Reuses `corrlift_edge` builders READ-ONLY.
- `hetero_model.py` — `HeteroRichNet` (LSTM + two independent WeightedGATLayer branches, SUM-aggregated) and
  `train_hetero_rich` (the batched, mask-aware, GPU training loop; zscore_floor output param only).
- `run_hetero_ablation.py` — orchestration: build panel, build adjacencies, train 3 variants, metrics + DM +
  fit evidence, write JSON. Dry/smoke mode = build adjacencies + one forward pass, no training.
- `__init__.py`

## Model — HeteroRichNet (hetero message passing without a PyG rewrite)
- Shared LSTM(5-feat, 2-layer) temporal branch -> `h` [B,N,hidden] (identical to MaskedRichNet).
- Relation `linear_corr`: `gat_lin1: 5 -> hidden*heads`, `gat_lin2: hidden*heads -> hidden*heads`
  (2-hop, matching the deliverable's `gat_layers=2`), consuming `adj_lin` at both hops.
- Relation `nonlinear_assoc`: `gat_nl1`, `gat_nl2` — SAME shapes, INDEPENDENT parameters, consuming `adj_nl`.
- Both GAT branches read the RAW node features at day t (`x[:,:,-1,:]`), exactly like MaskedRichNet (the GAT
  is PARALLEL to the LSTM, not fed by it — matches the delivered architecture note).
- Aggregate: `g = g_lin + g_nl` (SUM). Head: `Linear(hidden + hidden*heads -> hidden) -> ReLU -> Dropout ->
  Linear(-> 1)`. Head input dim = same as single-relation MaskedRichNet -> controlled comparison.
- `WeightedGATLayer` is imported READ-ONLY from `run_masked_rich`; independence = TWO distinct instances
  (PyTorch clones parameters per instance), so grads diverge whenever `adj_lin != adj_nl`.

## Per-relation Min-Max normalization
For each relation independently, over its FIRED off-diagonal edge weights (train-only graph):
`lo, hi = min(fired), max(fired)`; `norm = clip((w - lo)/(hi - lo), eps, 1.0)` with `eps=1e-6`.
Degenerate `hi <= lo` (all fired weights equal / <=1 edge) -> fired edges map to 1.0. Self-loop diagonal = 1.
Rationale: scale-match the two relations so neither dominates the gradient by raw magnitude; eps keeps the
weakest edge present under the GAT's `adjacency != 0` mask (values still in [0,1]).

## Leakage controls
- Graph cutoff = `D.d_va[0]` (first validation TARGET date): every close row fed to returns/corr/lift is
  strictly before every val/test target (mirrors the corrlift edge; a handful of purge-gap rows looser than
  the delivered `adj_corr` cut, touching no val/test data).
- Min-Max min/max computed from the train-only fired edges -> train-only by construction.
- Per-node target/feature scalers, HAR/HAR-X OLS: all TRAIN-only (inherited from `build_masked_rich`).

## Gates
- **Simplicity Gate:** no new abstraction beyond the two edge builders + one model + one runner; reuse the
  whole MaskedRich stack. `train_hetero_rich` supports ONLY the delivered `zscore_floor` output param
  (drops the unused `ratio_exp` branch) — minimum code.
- **Anti-Abstraction Gate:** use `nn.LSTM` / the existing `WeightedGATLayer` directly; no PyG dependency, no
  wrapper framework. Two `WeightedGATLayer` instances ARE heterogeneous message passing (independent
  per-relation weights) with the masked-panel semantics kept intact.
- **Performance / Batching Gate:** fully batched `[B,N,...]` tensors on GPU; both adjacencies masked per batch
  via `base * nmask.unsqueeze(1)` (mask-aware, block-diagonal-equivalent); no per-item Python loop in the hot
  path; single ReduceLROnPlateau; early stop on val MSE. Two GAT branches double GAT FLOPs but stay batched;
  154 nodes at batch 16-32 is well under 8GB VRAM. GPU selected via `torch.cuda.is_available()`.

## Over/under-fit evidence
`run_training` stamps per-variant `train_metrics` + `val_metrics` + `fit_diagnostics` (RMR.OF.classify_fit) +
`learning_curves` (per-seed train/val MSE), matching the corrlift/sector runners.

## Tests (unique basenames — avoid pytest prepend-import duplicate-basename shadowing)
- `test_hetero_edges.py` — Min-Max into [0,1] with train-only min/max (independent recompute); edge counts
  match thresholds on a fixture; symmetry/self-loop; train-only leakage-frozen; empty/degenerate cases.
- `test_hetero_model.py` — independent conv weights (distinct params + diverging grads under differing adj);
  SUM aggregation output dim; changing `adj_nl` changes the output (relation is actually used); finite forward.
- `test_hetero_runner.py` — `run_training` stubbed on a tiny real HNX slice (train fns monkeypatched);
  `train_hetero_rich` tiny CPU real run (1 epoch) + a stub-net multi-epoch run for early-stop branch coverage;
  device label; adjacency-for real slice; forward-smoke guards; main dry/train branches.
