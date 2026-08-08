# Summary of update — GAT-hybrid-style sparse adjacency for the masked graph

- Date: 2026-08-08
- Branch: `feature/masked-gnn` (worktree `.worktrees/masked-gnn`), base `3665936`
- Scope: additive `--adjacency dense|knn|threshold` edge sparsification on the Track B masked
  graph, plus masked G0/G1 re-run (dense vs knn-8 vs threshold-0.7), seed 42, 15 epochs, GPU.

## What changed

| File | Purpose |
| --- | --- |
| `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/data.py` | `_sparsify_correlation` (mutual k-NN / threshold), `_validate_adjacency_config`, `_adjacency_hash_fields`; adjacency mode threaded through `_correlation_adjacency`, `_masked_correlation_adjacency`, `build_graph_manifest`, `build_masked_graph_manifest`. |
| `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py` | `--adjacency/--top-k/--corr-threshold` CLI + guard; `_adjacency_config`, `_build_graph_manifest_for_mode`, `_edge_density_stats`, `_graph_validation_loss`, `_plot_learning_curve`, `_validate_graph_epochs`; per-epoch validation loss + learning curve; adjacency + edge-density in payload. |
| `.../test/test_sparse_adjacency.py` | New TDD suite (invariants (a)-(e), byte-identity regression, runner integration, epoch cap, coverage-closing unit tests). |
| `.../test/coverage_gate.sh` | Reusable C0=100% / C1>=80% diff-coverage gate for a pre-push hook. |
| `.../requirements/sparse_adjacency_requirements.md`, `.../design/sparse_adjacency_design.md`, `.../code_review/code_review_2026-08-08_sparse_adjacency.md` | SDD spec, plan, adversarial review. |

## Design decision (documented)
k-NN uses **mutual (AND) symmetrization** — edge i-j kept iff each ranks the other in its
top-k — which guarantees the required invariant "<= k off-diagonal neighbours per present row
AND symmetric". This diverges from the GAT-hybrid reference's UNION operator (which does not
bound degree). Ranking/thresholding uses |corr|; retained edges keep the signed value; the
self-loop (=1) is preserved so isolated present nodes stay valid for message passing. Node
sparsity (per-day presence mask) is unchanged. Adjacency mode is folded into the manifest hash
(non-dense only), so a sparse manifest can't be cross-used with a dense one; the dense manifest
hash is byte-identical to prior runs.

## Verification (real output)
- pytest: `154 passed` (baseline suite incl. real-data smoke + the new integration test that
  drives `run_graph_screening` on a 6-ticker slice). `-m "not smoke"`: 18/18 in the new file.
- ruff: `All checks passed!` on the changed files.
- Diff-coverage (C0) on changed lines: `data.py 100%`, `run_pilot.py 100%`
  (`diff-cover --fail-under=100` PASS). C1 (branch): no changed branch line below 80% (every
  added conditional has both arms tested).
- Data-quality gate (touches manifest/data path): Pandera `check_schema()` = **PASS**
  (34/34 artifacts valid); Evidently `check_drift()` = **INFO** (report generated,
  `results/quality_gate/sparse_adjacency/drift.html`; drift is informational, never fails).
- Adversarial code review (3-layer) done; 1 HIGH bug found + fixed (plotting could crash a
  training run — now fully guarded). See `code_review/code_review_2026-08-08_sparse_adjacency.md`.

## Results — masked G0/G1, seed 42, 15 epochs, GPU (RTX 4060)

Snapshots: 6470 over 4941 distinct dates (train 4523 / val 1237 / test 710). All 6 metrics on
the validation split; `nonpositive_prediction_rate = 0.0` for every run.

G0 is identical across all three conditions (val loss 0.83924, QLIKE 0.5101, DirAcc 48.71) —
expected, since G0 does not use the graph, so the adjacency mode affects only G1. This is a
clean control: the comparison isolates the effect of the edge set on G1.

| adjacency | edge density avg (max) | G0 valloss | G1 valloss | delta (G1-G0) | G1 QLIKE | G1 R² | G1 DirAcc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense         | 18.58 (32) | 0.83924 | 0.83797 | **-0.00127** | 0.5065 | 0.7437 | 49.10 |
| knn top-8     | 5.87 (8)   | 0.83924 | 0.83671 | **-0.00254** | 0.5065 | 0.7442 | 48.71 |
| threshold 0.7 | 1.09 (23)  | 0.83924 | 0.83924 | **0.00000**  | 0.5098 | 0.7417 | 48.46 |

G1 6-metric detail (RMSE / MAE identical to G0 at this scale; `nonpositive_prediction_rate = 0`):
dense G1 rmse 0.00146 / mae 0.00046; knn G1 rmse 0.00146 / mae 0.00046; threshold G1 rmse
0.00147 / mae 0.00046. Full per-run JSON:
`results/pooled_news_gnn_masked_{dense,knn8,thr07}_seed42_2026-08-08_230837/h5/{G0,G1}/results.json`.
The knn edge-density max = 8 confirms the mutual-k-NN "<= k off-diagonal per present row"
invariant holds on the real 33-ticker data.

### Convergence / overfitting (per-epoch learning curves)
Each G0/G1 records per-epoch train AND validation loss + a `learning_curve.png` (6 curves total).
- G0: train 0.9375 -> 0.9330, val 0.8422 -> 0.8392 (monotone, plateau by ~ep10).
- G1 dense: train 0.9402 -> 0.9326, val 0.8462 -> 0.8380 (val min at ep15).
- G1 knn-8: train 0.9416 -> 0.9346, val 0.8454 -> 0.8367 (val min at ep15).
- G1 threshold: train 0.9409 -> 0.9345, val 0.8409 -> 0.8391 (val min at ep14).
Every run's validation loss decreases monotonically to its minimum with **no upward divergence**
= converged, **no overfitting** within 15 epochs. dense/knn G1 val is still edging down
marginally at ep15 (a few more epochs could shave a little further); threshold has flattened.

## Honest verdict
At convergence (15 epochs) the masked graph gives a **small but consistent lift** — G1 beats the
no-graph G0 on val loss for dense and knn, reversing the earlier 5-epoch null. **Edge sparsity
matters and has a sweet spot:** moderate GAT-hybrid-style trimming (mutual k-NN top-8, ~5.9
edges/row) gives the **largest** lift (delta -0.00254, roughly double dense's -0.00127), while
aggressive thresholding (|corr|>0.7, ~1.1 edges/row) over-sparsifies the graph to near-empty and
erases the benefit (delta 0.000). The absolute deltas are small (val loss ~0.837-0.839) and this
is a **single seed** — statistical significance is not established here; a Diebold-Mariano test
across seeds (see follow-up) is needed before claiming the k-NN lift is significant. DirAcc stays
~48-49% for all (anti-persistence ceiling), so the lift is a QLIKE/MSE effect, not a direction
effect.

## Notes / follow-ups
- DirAcc stays ~49% (below 50%) — consistent with the project's documented anti-persistence
  structure, not a bug.
- Metrics literature check (independent research): the 6 accuracy metrics are the correct
  accuracy set for a volatility-GNN paper; the notable gap is **inferential rigor** — add a
  Diebold-Mariano test (near-mandatory for "model A beats B") and a Model Confidence Set for
  multi-model comparison; HMSE/HMAE and a Mincer-Zarnowitz regression are optional supplements.
  (Recommendation only; not implemented here.)
- Wiring `coverage_gate.sh` into the shared `.git/hooks/pre-push` is a parent action (the
  worktree must not modify main-tree / shared git infra).
