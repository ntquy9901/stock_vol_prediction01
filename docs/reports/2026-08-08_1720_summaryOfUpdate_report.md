# Summary of Update — T0.2 Batch the Track B Graph Path

Date: 2026-08-08
Branch: `feature/pooled-news-gnn-pilot`
Baseline: `baselines/2026-08-08_pooled_news_gnn_ablation_baseline`

## What changed

The Track B graph phase used batch-size-1 loops (GPU ~34% utilised). A GPU profile
(8 tickers, 2 epochs) located the cost:

| Stage | Time | Share |
| --- | ---: | ---: |
| `build_graph_bound_p3_warm_start` | 145.1s | 40.4% |
| `build_graph_safe_p3_checkpoint` | 126.5s | 35.2% |
| `_run_one_graph_model` (train + eval) | 32.7s | 9.1% |

Both P3 builders TRAIN P3 (`optimizer.step()` per sample), so they are not forward-only;
batching them is an approved R10 SGD re-baseline (not a numerically-identical change).
All three loops were batched.

- Commit `29babb2` (part 1): `_pooled_training_batches` stacks pooled samples into
  `[batch, time, feature]` (each sample keeps its own ticker target scaler); both builders
  take `train_batch_size` (default 256, wired to `--batch-size`). At `train_batch_size=1`
  the loop reproduces the historical per-sample trajectory exactly.
- Commit `b96e4d2` (part 2): `_run_one_graph_model` stacks train snapshots into
  equal-weighted mini-batches via the existing `_graph_prediction_batch` + `_mean_snapshot_mse`
  (one `optimizer.step` per batch, for G0 and G1 identically). Adds `--graph-train-batch-size`
  (default 32) and records `nonpositive_prediction_rate` in the graph `results.json`.

Files: `code/run_pilot.py` (builders, training loop, CLI, results field);
`test/test_models.py` (7 new tests).

## Preserved invariants

- Graph manifest content hashes and `graph_train_hash` are byte-identical to the Task 7 run
  (leakage / provenance contract unchanged). Only the shared graph-safe P3 weights re-baseline;
  G0 and G1 load the same checkpoint (fair ablation).
- Denormalized positivity floor + `evaluate_records` nonpositive<=1% gate kept.
- Frozen-encoder no-gradient assertion kept inside the batched training loop.
- Training order stays deterministic/chronological (unshuffled), documented in-code.
- Seeds recorded in results.json/runtime metadata.

## Verification (captured)

- `python -m pytest baselines/.../test/ -q` → 117 passed (110 prior + 7 new). RED confirmed
  before implementation (7 failed), GREEN after.
- `python -m pytest ... -m smoke` → 2 passed.
- `ruff check code/ test/` → All checks passed.
- Test env: default Python 3.14 (torch 2.12 CPU) — has pytest/ruff. GPU run env:
  `.venv_gpu_encode` (Python, torch 2.6.0+cu124, RTX 4060) — no pytest/ruff, used only for the run.

## Batched GPU run (SUPERSEDES Task 7 batch-1 seed-42 numbers)

Command: `run_pilot.py --phase graph --epochs 5 --seed 42 --device cuda`
(builder batch 256, graph train/val batch 32). 33 tickers.
Output: `results/pooled_news_gnn_g0g1_batched_2026-08-08_171457_seed42/`.

Wall clock: **235s (~3.9 min)** vs the ~40 min batch-1 baseline → **~10x speedup**.

| Metric | G0 (batched) | G1 (batched) | G0 (Task 7 batch-1) | G1 (Task 7 batch-1) |
| --- | ---: | ---: | ---: | ---: |
| val_loss | 0.8607 | 0.8627 | 0.9498 | 1.0596 |
| MSE | 5.79e-06 | 5.81e-06 | 6.82e-06 | 8.12e-06 |
| RMSE | 0.002407 | 0.002410 | 0.002611 | 0.002850 |
| MAE | 0.000670 | 0.000660 | 0.000663 | 0.000707 |
| R2 | 0.7447 | 0.7440 | 0.6997 | 0.6421 |
| QLIKE | 0.6876 | 0.6963 | 0.8388 | 4.3811 |
| Dir Acc | 48.61% | 48.52% | 49.24% | 47.67% |
| nonpositive frac | 0.0% | 0.0% | — | — |

paired_delta (G1 − G0 val_loss): **+0.00198** (was +0.1098 at batch-1).

## Verdict (honest)

Direction is STABLE: graph message-passing (G1) still does not help — G1 val_loss and QLIKE
are marginally worse than the no-graph G0, Dir Acc marginally lower. The batched re-baseline
did narrow the gap sharply (delta +0.1098 → +0.00198) and removed the pathological batch-1 G1
QLIKE (4.38 → 0.696), i.e. batched training produced a better-behaved but still non-improving
G1. Nonpositive fraction 0.0% (<=1% gate holds). These numbers supersede Task 7's batch-1
seed-42 numbers.

## Code review

Adversarial 3-layer self-review recorded at
`code_review/code_review_2026-08-08_T0.2_batched_graph.md`. No HIGH/MEDIUM findings open.
Parent will run the final `/code-review` before merge.

## Follow-ups / risks

- Batch sizes (256 builders, 32 graph train) are defaults; not tuned. Multi-seed re-baseline
  (seeds 123/2026) not run here — single seed 42 per task scope.
- Diff-coverage gate (`diff-cover --fail-under=100`) not run — tool not installed in repo
  (documented tooling gap); coverage argued via targeted RED→GREEN tests for every changed branch.
- Not pushed (parent verifies + pushes).
