# Code review — sparse adjacency for the masked graph (2026-08-08)

Adversarial 3-layer review of the `--adjacency dense|knn|threshold` change
(`code/data.py`, `code/run_pilot.py`, `test/test_sparse_adjacency.py`). Scope excludes
`archive/`. All HIGH/MEDIUM findings resolved before "done".

## Layer 1 — Blind Hunter (hidden bugs)

| # | Sev | Finding | Resolution |
|---|-----|---------|------------|
| B1 | HIGH | `_plot_learning_curve` only guarded the matplotlib IMPORT, not `savefig`; an IO error (e.g. output dir removed) raised and crashed the whole training run — observed live when a results dir was deleted mid-run. | Widened the `try/except` to wrap the entire plotting+save body. Plotting is now best-effort; test `test_plot_learning_curve_survives_missing_matplotlib` covers the except arm. |
| B2 | MED | k-NN symmetrization could violate the "<= k per row" invariant if UNION (OR) symmetrization were used (a hub node exceeds k). | Implemented **mutual (AND)** k-NN: edge kept iff each endpoint ranks the other in its top-k. Guarantees symmetric + degree <= k. Documented divergence from GAT-hybrid's union operator in `design/sparse_adjacency_design.md`. |
| B3 | MED | Self-ranking: a node could select itself as a neighbour (corr diagonal = 1 is the max |corr|). | `np.fill_diagonal(magnitude, -1.0)` on a COPY excludes the diagonal from ranking/threshold; the signed self-loop is restored via `np.diag(correlation)`. |
| B4 | MED | A sparse snapshot could leave a present node isolated (no neighbours) → message passing requires a self-loop or neighbour. | Diagonal self-loop = 1 is always restored for present nodes; verified `_ResidualMessagePassing` treats the self-loop as satisfying `needs_neighbor`. No crash for isolated nodes. |

## Layer 2 — Edge Case Hunter

| # | Finding | Handling |
|---|---------|----------|
| E1 | `node_count <= 1` present sub-matrix | `_sparsify_correlation` early-returns unchanged (self-loop only). Test `test_sparsify_single_node_returns_unchanged`. |
| E2 | present count <= k+1 → knn keeps all (== dense adjacency) | Correct by construction; hash folding keeps dense/knn manifests distinct (`test_manifest_hash_differs_across_adjacency_modes` uses a 2-ticker manifest where the arrays ARE identical). |
| E3 | threshold tau out of range | `_validate_adjacency_config` rejects tau not in [0,1). Test `test_threshold_config_out_of_range_is_rejected`. |
| E4 | Negative correlations | Ranked/thresholded by |corr|, retained with sign (same as the pre-existing dense signed-correlation semantics). Test `test_knn_retains_signed_correlation_weight`. |
| E5 | Absent nodes under sparsification | Sub-matrix computed over present indices only, placed via `np.ix_`; absent rows/cols stay 0. Test `test_sparse_masked_adjacency_respects_presence`. |
| E6 | `--adjacency` on a non-graph phase | `parse_args` rejects it (would otherwise be silently ignored). Test `test_parse_args_rejects_adjacency_outside_graph_phase`. |

## Layer 3 — Acceptance Auditor (spec vs implementation)

- Default `dense` byte-identical: `test_dense_default_masked_adjacency_byte_identical` +
  `test_dense_default_intersection_adjacency_byte_identical` (np.array_equal vs the exact
  pre-change computation). PASS.
- knn <= k off-diagonal per present row + symmetric: `test_knn_sparsify_bounded_degree_and_symmetric`. PASS.
- threshold zeros |corr| <= tau: `test_threshold_sparsify_zeros_below_tau`. PASS.
- presence respected: `test_sparse_masked_adjacency_respects_presence`. PASS.
- cross-mode hash difference: `test_manifest_hash_differs_across_adjacency_modes`. PASS.
- leakage / scaler / provenance / graph-safe P3 boundary / positivity floor + nonpositive<=1%
  gate / seeding: unchanged (no edits to those code paths; masked run reports
  `nonpositive_prediction_rate` 0.0). PASS.
- All 6 metrics emitted (MSE, RMSE, MAE, R², QLIKE, DirAcc): confirmed present in every
  `results.json`. PASS.

## Coverage
- Diff-coverage C0 = 100% on changed lines of `data.py` and `run_pilot.py`
  (`diff-cover --fail-under=100`). C1 (branch): no changed branch line below 80% (every added
  conditional has both arms tested). Evidence in the summary report.

## Verdict
No unresolved HIGH/MEDIUM findings. One HIGH (B1) was a genuine crash bug found and fixed
during the run. Change is additive; dense default unchanged.
