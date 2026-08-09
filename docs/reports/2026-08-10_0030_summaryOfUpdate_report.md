# Multi-horizon Track-B consistent ladder (h1 / h10 / h22) — summary

Run timestamp: `2026-08-09_180326` (h1/h10/h22); h5 basis unchanged (`2026-08-09_154402`).
Branch: `feature/masked-gnn`.

## What changed

The consistent-basis nested ladder P0 -> P1 -> P2 -> P3 -> G1, previously built only at horizon 5,
was extended to horizons 1, 10 and 22 on the same procedure. The target is `volatility.shift(-h)`;
the masked k-NN-8 manifest, leakage-safe graph-bound train set, shared per-ticker scalers, positivity
floor, identical val/test observations per horizon, seeds 42/123/2026, and the exact nesting
(P3 = the trained G1 read out with the GAT/message-passing residual disabled) are preserved from h5.

Code change: `main()` in `ladder_consistent.py` and `main()`/`_collect()` in
`ladder_consistent_dump.py` accept a horizon; the run-config horizon is set at entry (restored in a
`finally`), results route to `results/.../h{horizon}`, and the aggregator reads the `h{horizon}`
subdir and runs Diebold-Mariano at the requested horizon (HAC truncation lag `h-1`). `build_basis`
and `run_seed` read the horizon constant unchanged. Default remains h5.

## Files

- `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/ladder_consistent.py` — `main(ts,
  device, horizon)` horizon routing (commit ea0d359).
- `docs/reports/ladder_consistent_dump.py` — `main`/`_collect` horizon parametrization (commit ea0d359).
- `baselines/.../test/test_ladder_consistent.py` — `test_main_routes_horizon_to_subdir_and_stamp`
  (RED->GREEN smoke: tiny basis, writes `h10`, stamps horizon, restores default).
- `baselines/.../test/test_ladder_dump.py` — `test_dump_reads_horizon_subdir_and_stamps_horizon`
  (fabricated 2-seed tree -> `h22` summary + title).
- Canonical per-horizon reports: `docs/reports/ladder_consistent_h{1,10,22}_2026-08-09_180326.{json,md}`.
- Combined summary: `docs/reports/ladder_consistent_multihorizon_2026-08-09_180326.md`.

## Results (3-seed mean, graph effect G1 vs P3)

Nesting verification: graph-off readout determinism = 0.0 (bit-identical) for all 3 seeds at every
horizon. Basis sizes: h1 val 14550 / test 14596; h10 val 14253 / test 14299; h22 val 13857 / test
13903 (snapshots 6482 / 6452 / 6415).

| horizon | split | G1 QLIKE<P3 seeds | QLIKE delta mean | paired-t p | DM-QLIKE all sig-neg | verdict |
|---|---|---|---|---|---|---|
| 1 | val | 2/3 | +4.343e-01 | 0.4317 | False | B |
| 1 | test | 2/3 | +3.088e-02 | 0.5305 | False | B |
| 5 | val | 3/3 | -3.899e-03 | 0.0096 | False | B |
| 5 | test | 2/3 | -5.620e-04 | 0.7913 | False | B |
| 10 | val | 3/3 | -6.532e-03 | 0.0447 | False | B |
| 10 | test | 3/3 | -1.182e-03 | 0.0669 | False | B |
| 22 | val | 3/3 | -6.170e-03 | 0.0002 | False | B |
| 22 | test | 0/3 | +4.871e-03 | 0.1425 | False | B |

The graph verdict is B (null) at all four horizons: G1 never clears the A bar (QLIKE < P3 in all
seeds AND per-seed DM-QLIKE significant-negative in all seeds).

- h1: G1 improves MSE/RMSE/R2/DirAcc over P3 and DM-MSE is significant-negative in 3/3 test seeds
  (-3.27 / -3.42 / -2.73), but QLIKE is unstable — seed 2026 inflates G1 QLIKE (VAL G1 QLIKE mean
  0.878 +/- 0.77), driven by near-floor 1-day predictions — so the QLIKE-based verdict stays B.
- h10: small G1 QLIKE improvement on VAL (paired-t p=0.0447) and TEST (3/3 seeds, p=0.0669), not
  significant under per-seed DM.
- h22: G1 QLIKE improves on VAL in 3/3 seeds (paired-t p=0.0002) but reverses on held-out TEST
  (0/3 seeds), i.e. the improvement does not generalize.

Per-horizon 5-rung tables (all 6 metrics, val + test) are in the canonical `h{1,10,22}` reports and
the combined multi-horizon file.

## Provenance note (h10/h22 run)

The original background launcher survived a harness task-kill and completed h1/h10/h22 (ladders +
dumps, all rc=0). A redundant recovery orchestrator, started while that state was unknown, raced the
launcher on h10's per-ticker `sample_manifest.json` atomic write and recorded one transient
`PermissionError` (WinError 32) -> its own h10 rc=1; the launcher's h10 completed cleanly. Runs are
fully seeded/deterministic per (horizon, seed) and `_write_json` is write-tmp-then-atomic-replace (no
torn reads), so the on-disk results are complete and identical to a single clean run. Verified: all
three seeds per horizon have `ladder_metrics.json` with the correct horizon, all 5 rungs, all-finite
metrics, nesting determinism 0.0, and consistent snapshot/observation counts; a single-process clean
re-dump of h1/h10/h22 reproduced valid 3-seed reports (seeds [42, 123, 2026]).

## Checks run

- `pytest test/test_ladder_consistent.py test/test_ladder_dump.py` -> 7 pass (RED->GREEN for both new
  horizon tests).
- `ruff check` on `ladder_consistent.py`, both test files, `ladder_consistent_dump.py` -> clean.
- Real GPU run (`.venv_gpu_encode`, cuda): 3 seeds x 5 rungs x 3 horizons; nesting determinism 0.0
  every seed/horizon; canonical reports written; clean re-dump confirmed 3-seed validity.
- Pre-push quality gate: see push section (TDD + pytest+coverage + ruff + Pandera/Evidently).

## Data-quality gate

N/A for THIS change (no data change): horizon parametrization of the driver + aggregator; trains on
the unchanged `data/processed` + `dual_group_news_panel.parquet` with a different target shift. The
pre-push hook nonetheless runs Pandera schema + Evidently drift.

## Code review

Focused adversarial self-review of the diff: (1) horizon global set/restore — guarded by `finally`
and the smoke test asserts the module default is restored; (2) obs-set invariant per horizon — the
runtime assertion in `build_basis` (graph present-nodes == pooled val/test samples) ran green in all
three real runs; (3) DM HAC lag = horizon-1 per horizon — exercised by the aggregator at h=1/10/22;
(4) concurrent-write race provenance — resolved and documented above with a clean re-dump as standing
evidence. The interactive 3-layer `/code-review` skill is unavailable in this autonomous context.

## DoD

- [x] 5-rung nested ladder at h1/h10/h22 (3 seeds, 6 metrics, val+test, same-obs-per-horizon basis).
- [x] Nesting exact (P3 = G1 graph-off), determinism 0.0 every seed/horizon; obs-set invariant asserted.
- [x] G1-vs-P3 Diebold-Mariano (QLIKE + MSE) per horizon; verdict B (null) at all horizons.
- [x] Canonical `ladder_consistent_h{1,10,22}_2026-08-09_180326.{json,md}` + combined multi-horizon MD.
- [x] TDD RED->GREEN; ruff clean.
- [ ] Pre-push gate + feature push (this section completed at push time).
