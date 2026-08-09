# Consistent-basis Track-B ladder P0 -> P1 -> P2 -> P3 -> G1 (summary)

Run timestamp: `2026-08-09_154402`. Branch: `feature/masked-gnn`.

## What changed

The Track-B ablation carried BOTH a P3 row (pooled screening: full pooled manifest, train-only
scalers, no positivity floor) AND a G0 row (`GraphAblationModel(P3, use_gnn=False)` on the masked
graph-bound backbone: common-date scalers, positivity floor). They are conceptually the same
"backbone, graph off" but differed numerically only because of a different training/eval basis.

This change rebuilds the ladder as ONE nested ladder on ONE basis and eliminates the separate G0
row. P3 is now literally the graph-off model: the identical trained G1 read out with the
GAT/message-passing residual disabled.

- Basis (all rungs): masked manifest, k-NN-8 sparse adjacency, leakage-safe graph-bound train set
  (`target_date <= graph.train_end_date`; equal to the full pooled train split, since
  `train_end_date` is the max pooled-train target date), shared per-ticker scalers (`graph_store`),
  positivity floor, identical held-out validation (14,418 obs) and test (14,464 obs) observations,
  seeds 42/123/2026, horizon 5, frozen-encoder base cache.
- Rungs: P0 = pooled HAR; P1 = price-only LSTM (news/gate/graph off); P2 = price + news (gate/graph
  off); P3 = price + news + gate, graph off (= graph-off G1); G1 = P3 + GAT.

## Files

- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/models.py` -> add
  `apply_message_passing: bool = True` to `GraphAblationModel.apply_graph_head` (skippable graph
  residual; head + frozen encoders shared, so G1-minus-GAT equals P3 bit-identically).
- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/ladder_consistent.py` -> ladder
  driver (one basis, 5 rungs, 3 seeds, nesting check).
- `docs/reports/ladder_consistent_dump.py` -> aggregator -> canonical
  `docs/reports/ladder_consistent_h5_2026-08-09_154402.{json,md}`.
- Tests: `test/test_models.py` (nesting mechanism, `atol=rtol=0`),
  `test/test_ladder_consistent.py` (driver graph-rung smoke + P0/P1/P2 rung integration + helpers).
- `design/ladder_consistent_design.md`.

## Results (3-seed mean)

Nesting verification: graph-off readout determinism = 0.0 (bit-identical) for all 3 seeds; graph
residual magnitude (mean abs raw pred diff G1 vs P3) ~3.2e-5 to 4.4e-5. "Remove the GAT from G1 =
P3" holds exactly.

VAL:

| rung | rmse | r2 | qlike | dir_acc |
|---|---|---|---|---|
| P0 | 0.0014724 | 0.7394 | 0.5096 | 48.52 |
| P1 | 0.0014922 | 0.7324 | 0.5062 | 48.54 |
| P2 | 0.0014790 | 0.7371 | 0.5031 | 48.52 |
| P3 | 0.0014681 | 0.7410 | 0.5130 | 48.44 |
| G1 | 0.0014555 | 0.7454 | 0.5091 | 48.68 |

TEST:

| rung | rmse | r2 | qlike | dir_acc |
|---|---|---|---|---|
| P0 | 0.0022893 | 0.7668 | 0.5676 | 48.53 |
| P1 | 0.0022646 | 0.7718 | 0.5648 | 47.98 |
| P2 | 0.0022703 | 0.7706 | 0.5599 | 48.04 |
| P3 | 0.0023129 | 0.7620 | 0.5765 | 47.88 |
| G1 | 0.0023053 | 0.7635 | 0.5759 | 48.22 |

Graph effect (G1 vs P3), Diebold-Mariano (h=5, HLN-corrected):

- VAL: G1 QLIKE < P3 in 3/3 seeds; paired-t p=0.0096; DM-QLIKE per seed p=0.208/0.016/0.006
  (not significant in all seeds); DM-MSE p=0.026/0.015/0.066. **Verdict B.**
- TEST: G1 QLIKE < P3 in 2/3 seeds; paired-t p=0.79; DM mixed. **Verdict B.**

The graph effect is a genuine null on this consistent basis (small ~0.4% VAL QLIKE improvement that
is not robust under per-seed DM and does not hold on held-out test), confirming the prior
new-backbone multi-seed finding (`verdict_masked_g0g1_newbackbone_2026-08-09_120512`).

## Changes vs prior (pooled-basis) P0-P3, for the paper/bundle

- P3 is redefined: it is now the graph-off G1 (common-date scalers, positivity floor, graph-phase
  15-epoch head refinement), not the separately-trained pooled screening P3 (train-only scalers,
  no floor, best-val 5-epoch). Its numbers shift accordingly and it is the SAME model as G1 minus
  the GAT.
- The separate G0 row is removed; "remove the graph from G1" now lands exactly on the P3 row.
- P1/P2 are trained on the same graph-bound basis and scored on the identical val/test observations.
- Graph verdict unchanged (B / null).

## Known asymmetry (caveat, documented)

P1/P2 heads train 5 pooled epochs (best-val selection); P3/G1 share a frozen backbone (5 epochs)
plus a 15-epoch message-passing/head refinement (final-epoch, no early stopping) — the regime the
prior G0/G1 used. The architecture nesting (component toggles) is exact; the head-epoch/selection
difference between the pooled rungs and the graph rungs is inherent to absorbing the old G0 into P3.

## Checks run

- `pytest test/test_models.py` -> 39 pass (incl. nesting mechanism, atol=rtol=0).
- `pytest test/test_ladder_consistent.py` -> 6 pass (graph-rung smoke determinism 0.0; P0/P1/P2 rung
  integration; helper contracts).
- Full baseline suite `pytest test/` -> exit 0 (178 tests).
- `ruff check` on all changed/new files -> clean.
- Real GPU run (`.venv_gpu_encode`, cuda): 3 seeds x 5 rungs, ~26-30 min/seed; nesting determinism
  0.0 all seeds; canonical report written.

## Data-quality gate

N/A (no data change). The change adds a driver + one model flag and trains on the unchanged
`data/processed` + `dual_group_news_panel.parquet`; no data/features/manifest edits.

## Code review

Focused adversarial self-review of the diff (stale/mis-aligned base cache across P3/G1 readout;
graph-off vs graph-on head sharing; present-node masking in the P3 readout; obs-set equality
between graph present-nodes and pooled samples; DM alignment by sorted (id,date)). Top risk =
P3 readout reading a wrong/detached base -> guarded: P3 reuses the identical shared frozen base as
G1 and the nesting test (atol=rtol=0) + runtime determinism 0.0 are standing evidence. No blocking
findings; the interactive 3-layer `/code-review` skill is not available in this autonomous context,
so a focused adversarial self-review was performed and recorded here.

## DoD

- [x] 5-rung ladder re-run on one basis (3 seeds, h5) + consolidated table (6 metrics, val+test).
- [x] Nesting verified numerically (G1-minus-GAT == P3, determinism 0.0) + test (RED->GREEN).
- [x] G1-vs-P3 Diebold-Mariano (QLIKE + MSE), verdict B (null).
- [x] Canonical `docs/reports/ladder_consistent_h5_2026-08-09_154402.{json,md}`.
- [x] Tests + ruff clean; committed + pushed to `feature/masked-gnn`.
- [ ] diff-cover: Not run (pytest-cov/diff-cover not installed in repo, per CLAUDE.md tooling gap).
