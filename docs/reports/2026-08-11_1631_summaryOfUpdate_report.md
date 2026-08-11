# Summary of update — EDA-recommended GNN baseline

## What changed
New baseline `baselines/2026-08-11_eda_gnn_baseline/` implementing and testing the graph-EDA
recommended GNN configuration (`docs/eda/graph_recommendation.json`, Conclusion C) against HAR, on a
leakage-safe basis directly comparable to `ladder_consistent` (h5). Branch `feature/eda-gnn` off
`feature/masked-gnn`, isolated worktree `.worktrees/eda-gnn`.

## Files (path → purpose)
- `code/features.py` → extended node features: MarketPK (contemporaneous cross-sectional median PK) +
  volume_zscore_20 (trailing rolling z-score), leakage-safe; ExtendedTickerPreprocessor + store.
- `code/edges.py` → TRAIN-frozen directed volume→PK lead-lag Top-5 adjacency + snapshot edge swap.
- `code/eda_model.py` → PriceGraphModel (price LSTM + masked residual message passing + positivity).
- `code/eda_ladder.py` → driver: E0 HAR / E1 +MarketPK / E2 +vol_z / E3 +vol2pk graph, E3off + G1corr
  controls; 3 seeds, 20 epochs, per-obs test dumps.
- `code/aggregate.py` → 3-seed metric means + Diebold–Mariano verdicts.
- `test/` → 15 pytest tests (leakage/causality/nesting/positivity/masking + full-pipeline smoke).
- `requirements/`, `design/`, `code_review/` → SDD artifacts + adversarial review record.
- `docs/reports/2026-08-11_1631_eda_gnn_results.md` → full ladder table + DM verdicts.

## Result (headline)
- E2 (HAR + MarketPK + volume_zscore_20, no graph) beats HAR on **QLIKE, DM-significant** (p=0.012);
  E1 (MarketPK alone) also (p=0.017). Genuine partial win (QLIKE), not an error win (SE p>0.19).
- The **directed vol2pk GNN (E3) does NOT beat HAR at DM significance** (QLIKE p=0.116, SE p=0.19)
  and is significantly worse than the correlation edge (E3 vs G1corr QLIKE p=0.044). The graph adds
  no OOS value — confirms EDA Conclusion C. Recommended final GNN = E3 (strongest GNN, tied-to-HAR on
  error), but its lift is carried by the node features, not the edges. No win claimed beyond DM.

## Tests + coverage
- `pytest baselines/2026-08-11_eda_gnn_baseline/test/ -v` → 15 passed (incl. `smoke`).
- Ruff clean on `code/` + `test/`.
- diff-cover (C0/C1): `Not run` — pytest-cov/diff-cover changed-line gating not exercised for this
  change (repo tooling gap per CLAUDE.md); coverage asserted via the targeted leakage/behaviour tests
  and the full-pipeline smoke instead.

## Code review
Adversarial 3-layer review (`code_review/code_review_2026-08-11.md`): 2 HIGH found and fixed before
the result run — H1 (vol2pk adjacency used a global date boundary over per-ticker splits → val/test
leakage; fixed to per-ticker train split + heterogeneous-history leakage test), H2 (QLIKE confound:
graph rungs floored, E0/E1/E2 not; added identical floor to all rungs). M2 (runtime obs-set guard)
and L3 (cross-seed target check) fixed; L1/L2/L4/L5/L6 documented.

## Commands run (real)
- `python .../code/eda_ladder.py 2026-08-11_144700 cuda` (3 seeds × {E0..E3,E3off,G1corr}, GPU).
- `python .../code/aggregate.py 2026-08-11_144700` → `results/eda_gnn_2026-08-11_144700_summary.json`.
- Obs-set parity check vs pilot 3-feature manifest → train/val/test identical (73026/14418/14464).

## Data-quality gate
`N/A (no data change)` — this baseline consumes the existing `data/processed` + `data/raw/prices`
read-only and adds no new dataset/manifest/pipeline-train artifact; Pandera/Evidently not re-run.

## Risks / follow-ups
- Message-passing softmax(signed corr) + unit self-loop may under-weight weak directed edges (L1);
  edge-weight tuning could be revisited but is out of scope and would not change the HAR verdict.
- diff-cover gating remains a repo tooling gap.

## DoD checklist
- [x] Code satisfies request (EDA config built + tested vs HAR, leakage-safe, honest verdict)
- [x] Tests + smoke pass; ruff clean
- [x] Code review run + HIGH/MEDIUM findings fixed
- [x] Summary + results reports generated
- [x] Every number from a real run
- [ ] diff-cover C0/C1 — Not run (tooling gap)
