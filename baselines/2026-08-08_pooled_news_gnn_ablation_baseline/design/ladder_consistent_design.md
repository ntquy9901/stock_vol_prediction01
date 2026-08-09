# Consistent-basis ladder P0 -> P1 -> P2 -> P3 -> G1 (single source of truth)

## 1. Problem (spec)

The paper carried BOTH a P3 row (pooled screening, full pooled manifest, train-only scalers, no
positivity floor) AND a G0 row (`GraphAblationModel(P3, use_gnn=False)` on the masked graph-bound
backbone, common-date scalers, positivity floor). They are conceptually the same "backbone, graph
off" but showed different numbers only because of different training/eval basis. This is the
confusion to remove.

## 2. Decision

Run the ENTIRE 5-rung ladder on ONE basis and make P3 literally the graph-off G1:

- **One basis (all rungs):** masked manifest (k-NN-8 sparse adjacency), leakage-safe graph-bound
  train set (`target_date <= graph.train_end_date`; here equal to the full pooled train split
  because `train_end_date` is the max pooled-train target date), the SAME per-ticker scalers
  (`graph_store`, fit on the 70% common-date boundary), positivity floor, the SAME held-out
  validation (14,418 obs) and test observations, seeds 42/123/2026, horizon 5, encoder cache.
- **Nested rungs via component toggles:**
  - P0 = pooled HAR regression on the graph-bound train set, scored on the same val/test obs.
  - P1 = price-only LSTM (news OFF, gate OFF, graph OFF).
  - P2 = price + news LSTM (gate OFF, graph OFF).
  - P3 = price + news + per-ticker gate, graph OFF — **the identical trained G1 model read out
    with the message-passing residual disabled.** No separate G0 row anymore.
  - G1 = P3 + GAT/message-passing (masked k-NN-8 adjacency).

## 3. Why P3 = "G1 with the GAT removed" (nesting is exact, not asserted)

`GraphAblationModel.apply_graph_head` gains one flag: `apply_message_passing: bool = True`. The
message passing is a residual (`base = base + mp(base)`), the frozen encoders + gate + trained head
are shared, so evaluating the trained G1 with `apply_message_passing=False` is bit-identical to a
`use_gnn=False` model built from the same weights. Therefore "remove the GAT from G1 = P3" holds by
construction. Evidence:
- Unit test `test_g1_graph_off_readout_equals_the_graph_off_model_within_fp_tolerance`
  (`atol=rtol=0`) + `test_evaluate_graph_off_equals_use_gnn_false_model_on_shared_base`.
- Runtime `nesting_check.json` per seed: graph-off readout is deterministic (max abs diff on a
  second pass) and the graph residual is non-trivial (mean/max abs raw pred diff G1 vs P3).

This changes the P3/G0 definition versus the prior pooled-basis P3 and separately-trained G0; the
graph effect (G1 vs P3) is now a pure component toggle on one shared backbone + head.

## 4. Simplicity / Anti-Abstraction gates

- Simplicity: one optional boolean on an existing method; a thin orchestration driver
  (`code/ladder_consistent.py`) and aggregator (`docs/reports/ladder_consistent_dump.py`) reuse the
  unit-tested building blocks (`run_har_reference`, `run_training`, `build_graph_*`,
  `_run_one_graph_model`, `_build_shared_graph_base`, `paired_losses`, `diebold_mariano`). No new
  model, no new manifest type.
- Anti-abstraction: no wrappers around torch/sklearn; the driver calls the existing functions.

## 5. Data flow

`build_basis` (seed-independent): load price + news, fit `graph_store`, build pooled(+news)
manifest, build masked k-NN-8 graph, derive graph-bound train set, assert the masked val/test
present-node observation set equals the pooled val/test samples. Per seed: P0 HAR, P1/P2 pooled
LSTM (screening 5 epochs, dropout 0.2), then P3-backbone warm-start + graph-safe checkpoint, train
G1 (15 epochs, frozen backbone) with the shared frozen-encoder base cache, read out P3 from the same
model with the residual off. DM (QLIKE + squared error) G1 vs P3 on val and test.

## 6. Known asymmetry (documented, not a bug)

P1/P2 heads train 5 pooled epochs (best-val selection); P3/G1 share a frozen backbone (5 epochs)
plus a 15-epoch message-passing/head refinement on graph snapshots (final-epoch, no early stopping)
— the same regime the prior G0/G1 used. The architecture nesting (component toggles) is exact; the
head-epoch and selection differences between the pooled rungs and the graph rungs are inherent to
absorbing the old G0 into P3 and are reported as a caveat.

## 7. Outputs

Per seed: `results/ladder_consistent_seed{seed}_<TS>/h5/` with `ladder_metrics.json`,
`P3/` + `G1/` prediction dumps, `nesting_check.json`. Canonical:
`docs/reports/ladder_consistent_h5_<TS>.{json,md}` (single source of truth for paper + bundle).
