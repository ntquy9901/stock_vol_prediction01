# Sparse adjacency for the masked graph — requirements (spec)

## Goal
Add a GAT-hybrid-style SPARSE adjacency option to the Track B masked graph so the graph
edge set can be trimmed (k-NN top-k or |corr|>tau threshold) while keeping the existing
per-day variable-node presence masking (data sparsity) unchanged. Determine whether sparse
edges on the full-data masked graph make the GNN (G1) beat the no-graph baseline (G0), or
whether the prior null result persists.

## Input / output
- Input: existing pooled + masked graph pipeline (`code/data.py`, `code/run_pilot.py`),
  33-ticker VN30 processed data + dual-group news panel, forecast horizon 5.
- Output: `--adjacency {dense,knn,threshold}` CLI option (default `dense`), `--top-k`
  (default 8), `--corr-threshold` (default 0.7); sparsified adjacency inside
  `_masked_correlation_adjacency` / `_correlation_adjacency`; masked G0/G1 results under
  dense vs knn-8 (and optional threshold-0.7) with all 6 metrics + edge-density + per-epoch
  learning curves.

## Acceptance criteria
- `--adjacency dense` (default) reproduces the current adjacency arrays byte-identically.
- `knn` yields a symmetric adjacency with <= k off-diagonal nonzeros per present row.
- `threshold` zeros every off-diagonal |corr| <= tau; retained entries keep the signed value.
- Sparse adjacency respects presence: absent nodes have no edges (rows/cols all zero).
- Adjacency mode is folded into the manifest hash so a knn manifest cannot be cross-used
  with a dense one (even when sparsified arrays coincide, e.g. <= k+1 present nodes).
- Leakage/scaler/provenance, graph-safe P3 boundary, positivity floor + nonpositive<=1%
  gate, and seeding are preserved.
- All 6 mandatory metrics (MSE, RMSE, MAE, R², QLIKE, DirAcc) reported for every condition.
- pytest (unit + real-data smoke) and ruff pass; Pandera schema + Evidently drift gate run.

## Edge cases
- <= 1 present node on a date: self-loop only, no neighbours (unchanged).
- <= k+1 present nodes: knn keeps all neighbours (equivalent to dense for that snapshot) —
  hash folding is what keeps dense/knn manifests distinct.
- All-equal correlations (degenerate): deterministic tie-break by index.

## Go / no-go
- GO if dense stays byte-identical, sparse invariants hold, hashes differ, gates pass, and
  masked knn G0/G1 runs complete with all 6 metrics.
- Report the honest verdict on whether the graph helps (masked-knn G1 < G0) or the null
  persists — a null result is a valid, reportable outcome.
