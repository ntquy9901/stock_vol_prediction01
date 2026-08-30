# Design — SERIAL LSTM→GNN hybrid

**Date:** 2026-08-30 · Paper: Sonani, Badii & Moin 2025 (arXiv:2502.15813) §2.3 + §3.2

## 1. Data flow (SERIAL — the whole point)

```
x  [B, N, SEQ, 5]  (5 node features over the SEQ lookback window; from masked_rich)
        │
        │  Stage 1 — LSTM encoder (shared weights, applied per stock)
        ▼
   reshape → LSTM(in=5, hidden=64, layers=2) → last hidden state
        │
        ▼
h  [B, N, 64]      ← per-stock TEMPORAL EMBEDDING  (h_i ∈ R^64)
        │
        │  Stage 2 — GNN over the LSTM embeddings.  *** GNN INPUT = h  (NOT raw x) ***
        ▼
   WeightedGATLayer(in=64, out=64, heads=4), masked corr+lift adjacency A_b
        │
        ▼
g  [B, N, 256]     ← relational embedding of the temporal embeddings
        │
        ▼
head( concat[h (64), g (256)] ) → ŷ  [B, N]      (residual skip on h; see §3)
```

**Contrast — delivered PARALLEL `MaskedRichNet`:**
```
h = LSTM(x)                       ┐  two INDEPENDENT branches
g = GAT(x[:, :, -1, :], A_b)      ┘  GAT reads RAW day-t features, NOT h
ŷ = head(concat[h, g])
```
The single load-bearing difference: **serial feeds `GAT(h, …)`; parallel feeds `GAT(raw, …)`.** A unit test
(`test_serialhybrid_model.py::test_gnn_input_is_lstm_embedding`) pins this by asserting the GAT sub-module
receives a tensor equal to the LSTM's last hidden state, and that changing SEQ-history changes the GAT input
(impossible in the parallel model, whose GAT only sees day t).

## 2. Edges — combined linear + non-linear (§3.2), TRAIN-only frozen
Reuse READ-ONLY `corrlift_edge.build_corrlift_adjacency` (a single COMBINED undirected weighted adjacency:
edge if `|ρ|>corr_thr` OR `lift>lift_thr`; normalized; self-loop=1; computed from close rows strictly before
`D.d_va[0]` then frozen — no leakage). Primary thresholds **ρ>0.25, lift>1.2** (dense enough to actually
exercise propagation on thin HNX returns). Paper thresholds 0.7/1.7 are ALSO computed and their density
recorded, with an explicit near-empty note. Adjacency is masked per batch by the valid-node mask
(`A_b = A ⊙ nmask`), identical to the corrlift/delivered convention, so invalid neighbours never enter
attention.

## 3. Head choice (documented per requirement)
`head(concat[h, g])` — a **residual skip on `h`**. Rationale: (a) keeps the temporal signal directly
available so the graph adds INFORMATION rather than replacing it (isolates the graph's marginal contribution,
matching the leave-one-out spirit — `no_graph_LSTM` is exactly this head on `h` alone); (b) standard skip
connection eases gradient flow through the GNN. When `use_graph=False` the head degenerates to `Linear(64→…)`
on `h` — i.e. the plain LSTM baseline, giving a clean same-architecture no-graph control.

## 4. Reuse map (all READ-ONLY imports; no live-path edits)
| Piece | Source (read-only) |
|---|---|
| Masked union panel, 5 feats, scalers, splits, `adj_vol2pk`, `d_va` boundary | `masked_rich.build_masked_rich` |
| `WeightedGATLayer` (weight/sign-aware, mask-aware GAT) | `run_masked_rich.WeightedGATLayer` |
| Delivered PARALLEL model + its trainer (variant 2, context) | `run_masked_rich.MaskedRichNet`, `train_masked_rich` |
| RMR helpers `_pred_dict/_ens/_metrics/_split_metrics/_ens_split/seed_metric_stats/_dm_all` | `run_masked_rich` |
| Fit verdict | `overfit_check.classify_fit` (via `RMR.OF`) |
| Combined corr+lift adjacency | `corrlift_edge.build_corrlift_adjacency`, `load_close_wide` |
| Panel build + screened universe + price dirs | `estimator_forecast_ablation`, `volatility_estimators` |
| HAR-X anchor, QLIKE floor, config | `baselines`, `config` |

New code lives only in this baseline: `serial_hybrid_net.py` (model + trainer) and
`run_serial_hybrid.py` (runner). The delivered PARALLEL variant is produced by calling the UNMODIFIED
`RMR.train_masked_rich` (upstream, already covered) — I never re-implement it.

## 5. SDD gates
- **Simplicity Gate:** one new nn.Module + one lean trainer (zscore_floor path only) + one runner mirroring
  `run_corrlift_ablation`. No configurability beyond `use_graph`. 1-hop GAT (paper §2.3 is a single relational
  layer; also fewer branches to cover) — documented, not tunable. PASS.
- **Anti-Abstraction Gate:** reuse `WeightedGATLayer`, RMR helpers, corrlift edge directly; no wrappers. PASS.
- **Performance/Batching Gate:** training is **batched** (`[B, N, …]` tensors, block adjacency `A ⊙ nmask`),
  **mask-aware** masked-MSE loss, **GPU** when `torch.cuda.is_available()`, single process, batch ~16–32 under
  8 GB VRAM. No per-item Python loop in the hot path (LSTM sees `[B·N, SEQ, 5]` in one call; GAT is einsum over
  `[B, N, N, heads]`). PASS.

## 6. Training
`train_serial(D, cfg, seed, use_graph, adj, return_splits=False)` mirrors `train_masked_rich`'s
zscore-floor path (per-node StandardScaler target, linear denorm, shared 1e-2·mean positivity floor, Adam +
ReduceLROnPlateau, grad-clip, early stop on val MSE, per-epoch train/val MSE learning curves). Only the
network is swapped (`SerialLSTMGNN` instead of `MaskedRichNet`). Seeds {42,123,2026}, 10 epochs (early stop),
HNX h1.

## 7. Test plan (UNIQUE basenames — avoid the pytest duplicate-basename collision)
- `test_serialhybrid_model.py` — forward shapes; mask-awareness; **GNN input == LSTM embedding**; SEQ-history
  changes the GNN input; `use_graph=False` degenerates to the LSTM head; finite outputs. Runs `train_serial`
  for real on a tiny synthetic `D` (1–2 epochs, CPU) to cover the train loop + early-stop + return_splits.
- `test_serialhybrid_runner.py` — `run_training` on a tiny REAL HNX slice with `train_serial` stubbed
  (metric/DM/fit-evidence plumbing on CPU, no epochs); `main()` dry + train branches; guard-raise paths;
  `serial_adj_for` on real HNX prices.

## 8. Risks
- Dense 0.25/1.2 graph may still be weak on HNX → likely null vs no-graph (prior 6 probes null). Reported
  honestly; a null is a valid result. The delivered VolGA comparison contextualises architecture-vs-density.
- Single GPU busy overnight → train only when `util<15 && VRAM<1200 MiB` sustained; single process.
