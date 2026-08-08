# Masked availability-aware graph — design note

Extends the pooled news GNN ablation with an availability-aware MASKED graph path to test
whether the graph-null result (G1 ≈ G0) was a data-scarcity artifact of the 26% common-date
intersection. Source: research report `docs/reports/2026-08-08_gnn_sparse_data_research.md`
Rank 1 (masked variable-node message passing, no imputation).

## Problem

The intersection graph (`build_graph_manifest`) uses a fixed 33-node set on only the dates where
ALL tickers trade: 1,296 of 4,989 dates (26%), capped by the newest listing (SSB). The GNN trains
on ~900 date-snapshots while per-stock models see up to ~4,868 days. "Graph doesn't help" is
therefore confounded with data volume.

## Approach (Rank 1, no imputation)

Build one graph per trading date over only the tickers PRESENT that day (variable present-node
count + presence mask), training on the full ~4,900-date union. Absent tickers on a date are
masked, never imputed (imputing pre-listing volatility would fabricate data / leak).

## Data flow (`build_masked_graph_manifest`)

1. Reuse the news-attached pooled manifest (`build_pooled_manifest` + `attach_news`): per-ticker
   chronological windows, train-only fitted scalers/winsor bounds, per-ticker HAR features, causal
   news. This is byte-identical to the pooled P1–P3 inputs.
2. Group those pooled samples by `(split, target_date)`. Each distinct target date in a split
   becomes one snapshot; the ticker's window is that snapshot's node for that date.
3. Pad every snapshot to the fixed ticker vocabulary (N = 33). `presence_mask[id] = 1` iff that
   ticker has a window on that date; absent nodes carry zeros and are masked.
4. Adjacency = correlation over the PRESENT nodes' price windows (reuses the existing correlation
   construction), placed into an N×N matrix with absent rows/cols = 0 and a present-node self-loop
   on the diagonal. A single-present-node snapshot keeps only its self-loop.
5. `train_end_date` = max train-snapshot target date → the graph-safe P3 boundary is unchanged
   (train targets ≤ boundary). Grouping happens INSIDE each split, so a snapshot never mixes splits
   and each node's window respects that ticker's own chronological split (no cross-split leakage).

### Split semantics (decision)

Per the task, the masked graph keeps the PER-TICKER chronological split (the same split the pooled
P1–P3 baselines already use), realized by grouping pooled samples within each split. The masked
graph is literally "the pooled samples arranged as per-date cross-sectional graphs", so it inherits
the pooled regime's exact split/scaler/news semantics — the cleanest apples-to-apples comparison.
Alternative (a single global union-date split) was rejected: the task mandates keeping the
per-ticker split, and this choice keeps the masked graph consistent with the pooled baseline.

## Model — masked message passing

`_ResidualMessagePassing.forward(node_features, adjacency, presence_mask=None)` and
`GraphAblationModel.forward(..., presence_mask=None)`:

- Absent nodes' features are zeroed; their incident edges are removed (absent columns of the
  adjacency zeroed → present nodes never attend to absent nodes). Present-node output is therefore
  invariant to absent-node features (perturbation-invariance test).
- Absent rows (all −∞ logits → NaN after softmax) are zeroed post-softmax, so absent nodes emit
  nothing and no NaN reaches aggregation.
- Loss (`_mean_snapshot_mse`) and raw-scale metrics aggregate over PRESENT nodes only; absent nodes
  are dropped from the evaluation records.
- The frozen P3 encoder (no-grad), the denormalized positivity floor, and the nonpositive ≤ 1% gate
  are unchanged.

`presence_mask=None` preserves the exact intersection numerics; intersection snapshots keep
`presence_mask=None` and omit the presence key from their content hash (byte-identical to prior runs).

## Switch

`--graph masked|intersection` (default `intersection`, unchanged). `--phase graph --graph masked`
opts into the new path.

## Gates / invariants preserved

- Per-ticker chronological split; train-only scalers; causal news; shuffle=False; seeds; provenance.
- Graph-safe P3 checkpoint boundary (train targets ≤ `train_end_date`).
- Frozen encoders receive no gradients; positivity floor; nonpositive ≤ 1%.
- No imputation; absent tickers masked. Only real observations enter.
