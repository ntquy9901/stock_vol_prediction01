# Code review — MTGNN learned-graph ablation (2026-08-29)

Adversarial 3-lens review (Blind Hunter / Edge-Case Hunter / Acceptance Auditor) of
`code/mtgnn_graph.py` and `code/run_learned_ablation.py`. Findings + resolutions below.

## Blind Hunter (hidden bugs)
- **B1 — self-loop vs node mask.** `build_adj` = `base[None] * nmask[:,None]`; the self-loop `A_ii` is zeroed
  when node `i` is an invalid source (`nmask[i]=0`). VERIFIED this is the SAME behaviour as the delivered
  `train_masked_rich.adj_batch` (`base * nm.unsqueeze(1)`), and invalid targets are excluded from scoring
  by `tmask`. Not a bug — consistent with the fixed-edge variants. (test: `test_wrapper_node_mask_zeros_invalid_source_columns`.)
- **B2 — faithfulness to MTGNN.** Equations (1)-(3) + top-k verified against the paper (ar5iv) and the
  official `nnzhan/MTGNN graph_constructor`, including the `+rand*0.01` tie-break. Independent-recompute test
  (`test_matches_paper_equations_independent_recompute`) pins values to the paper formula. No self-loop inside
  the constructor (matches official code); the self-loop=1.0 is added by the wrapper for `WeightedGATLayer`.
- **B3 — RMR private-symbol reuse.** The runner reuses `RMR._batches/_ens/_metrics/_dm_all/_split_metrics/`
  `_ens_split/seed_metric_stats/_pred_dict/OF` — all present in the read-only delivered module (grep-verified).
  Import is read-only; no delivered file is edited.

## Edge-Case Hunter
- **E1 — k > N.** `GraphConstructor` caps `k=min(subgraph_size, n_nodes)` and raises on `k<1`
  (`test_k_capped_at_n_nodes`, `test_invalid_subgraph_size_raises`).
- **E2 — inference stochasticity.** The `rand*0.01` tie-break runs at inference too, but it only reorders
  columns whose adjacency value is `0` (zeroed by `adj*mask` regardless), so the NONZERO structure and thus
  the prediction are effectively deterministic given fixed weights. Faithful to MTGNN; negligible effect.
- **E3 — early stopping.** wait-increment + break covered deterministically with `lr=0`
  (`test_train_learned_early_stops_when_val_never_improves`).
- **E4 — non-finite output.** Defensive raise in `run_dry`; finiteness asserted by unit + smoke tests.

## Acceptance Auditor
- **A1 — only the edge differs.** `LearnedGraphNet` subclasses `MaskedRichNet`; LSTM branch, 2-hop
  `WeightedGATLayer`, 5 node features, masked panel, HAR-X anchor, per-ticker scalers and QLIKE floor are all
  inherited/shared with the fixed-edge variants. The learned variant's train loop mirrors the delivered
  `zscore_floor` path (same optimizer/scheduler/loss/early-stop). ✓ requirement met.
- **A2 — over/under-fit evidence.** Result JSON carries `train_metrics`/`val_metrics`/`metrics`/`fit_diagnostics`/
  `learning_curves` for all 4 variants; gate-required keys `LSTM` + `LSTM_wGAT_vol2pk` present. ✓
- **A3 — DM tests.** learned-vs-{no-graph, stat, sector} date-clustered DM emitted. ✓
- **A4 — performance.** Batched `[B,N,seq,5]` + batched `[B,N,N]` adjacency, no per-item loop; CPU forced only
  to avoid contending with the saturated live GPU (documented, GPU allowed via env). ✓

## Performance lens
No batch=1 anti-pattern; the graph is built once per forward (`N×N` matmul, cheap) and shared across the
batch. Main cost is the delivered 2-hop GAT einsum (identical across variants), so the comparison is fair.

## Verdict
No CRITICAL/MAJOR findings. All raised items either verified-not-a-bug or covered by tests. C0=100% / C1=99%
on changed lines (single partial = the `out_dir=None` skip, above the 95% gate); ruff-F clean; 18 tests pass.
