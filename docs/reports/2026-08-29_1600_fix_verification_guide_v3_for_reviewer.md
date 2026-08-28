# Fix-verification guide v3 (for the external AI reviewer) — 2026-08-29

**Repo (public):** https://github.com/ntquy9901/stock_vol_prediction01 — review the files IN the repo.
**Supersedes:** v2 (`2026-08-29_1000`). Adds: the final-review fixes F-01..F-06, the over/under-fit evidence
system (capture + gate), the coverage-gate + evidence-capture upgrades, and the authoritative paper-table
generator (F-03). **Head at write time:** `adb208d`. **Scope exclusion:** `archive/` + `deliverables_*` are
out of scope.

This maps each item to `file:line` + guard test + how to verify. Two things are still **decisions for the
owner**, flagged in §5 — not defects.

## 0. Run everything
```
python -m pytest submission/soict_lstm_gat/tests/ scripts/eda/test_*.py scripts/garch_masked/test_*.py \
  scripts/quality_gate/test_*.py scripts/paper/test_*.py tests/test_lessons_regression.py -q      # numpy-only
.venv_gpu_encode/Scripts/python.exe -m pytest \
  baselines/2026-08-21_har_anchored_residual/code/test_masked_rich.py \
  scripts/garch_masked/test_run_oos_suite.py -q                                                    # torch venv
```
The pre-push gate enforces **C0 line = 100% + C1 branch = 95%** on changed lines and writes a per-push
evidence log to `docs/reports/gate_logs/<sha>.txt` + `scripts/task_dashboard/gate_results/<sha>.json`.

## 1. Final-review findings (`code_review_final_rerun_2026-08-29.md`) — status

| ID | Sev | Verdict | Where |
|----|-----|---------|-------|
| **F-01** | High | RESOLVED — real integration test, not tautological | `scripts/garch_masked/test_garch_masked.py::test_garch_integration_alignment_on_real_purged_panel` |
| **F-02** | Med | FIXED — test-only = explicit N/A, source-no-coverage = fail-closed | `scripts/git_hooks/pre-push` step 2 |
| **F-03** | Med | FIXED — single authoritative generator + provenance | `scripts/paper/build_final_tables.py` |
| **F-04** | Low | FIXED — `nonpositive_count` in GARCH status | `submission/soict_lstm_gat/baselines.py:153-160` |
| **F-05** | Low | FIXED — per-obs DM seam documented | `submission/soict_lstm_gat/evaluate.py:91` docstring |
| **F-06** | Low | Already done (R-05/R-04) | `metrics.py::per_obs_se`, `_has_garch`, `_done` |

### F-01 — verify the GARCH alignment is genuinely proven
The old test used a `SimpleNamespace`; the new one builds a REAL panel via `MR.build_masked_rich` with the
real train/val/test purge (`horizon` anchors dropped between splits) and mocks `garch_forecast` as a ramp, then
asserts each node's k-th valid TEST observation receives forecast step `n_va+k` — at h1/h5/h10. This proves the
design is **observation-contiguous** (purge-dropped anchors are targets that exist in NO series, so there is no
observation-space drift with horizon). **To challenge:** reconstruct the anchors yourself and confirm the k-th
test target's observation index from train-end is exactly `n_va+k`; or argue the calendar interpretation is
required (a modelling choice — GARCH is a dominated benchmark, stated).

### F-03 — verify the authoritative table generator
`scripts/paper/build_final_tables.py::authoritative_cell` is the ONE place that decides the source: LEARNED
(LSTM, LSTM+GAT) → `metrics_per_seed` (mean + std); DETERMINISTIC (HAR, HAR-X, GARCH) → `metrics`. It **raises**
if a learned model has `metrics` but no `metrics_per_seed` (anti-drift guard — the exact `build_report.py` risk).
Generated `docs/paper/generated/final_tables.{md,tex}` (100 rows / 20 panels) carries SHA-256 provenance of every
input. `build_report.py` is now marked LEGACY/NOT AUTHORITATIVE. Tests:
`scripts/paper/test_build_final_tables.py` (per-seed-vs-ensemble source, fail-loud, provenance; 100% cov).
**To challenge:** confirm the generated LSTM cells equal `metrics_per_seed` (e.g. vn30 h1 = `0.7037 (0.054)`),
NOT the ensemble `0.643`; confirm deletion of a learned `metrics_per_seed` block makes the generator raise.

## 2. Over/under-fit evidence system (user mandate 2026-08-29) — audit it

An internal audit found the delivered `result.json` were TEST-ONLY (could not prove generalisation). Now:
- **Capture (training):** `run_masked_rich.py::train_masked_rich(return_splits=True)` returns train/val preds +
  per-epoch train/val learning curves; `run()` writes `train_metrics`, `val_metrics`, `fit_diagnostics`,
  `learning_curves` into `result.json`. Guard: `test_masked_rich.py::test_run_out_subdir_writes_separate_results_tree`
  asserts the blocks appear.
- **Verdict:** `scripts/quality_gate/overfit_check.py::classify_fit` — overfit if val→test QLIKE degrades >25%
  OR train→test R² drop >0.20; underfit if train & test R² both below a floor. Tests: `test_overfit_check.py`.
- **Gate:** `scripts/quality_gate/check_overfit_evidence.py` runs in pre-push over the `result.json` in the push
  diff → BLOCK if a masked-rich result lacks evidence or a learned model is over/under-fit. Tests:
  `test_check_overfit_evidence.py`. Both suites run every push (gate step 4).
- **To challenge:** verify `classify_fit` thresholds are sensible; verify the gate skips non-training result.json
  and fails-loud on unreadable ones; confirm the capture does NOT change any test metric (train/val are additive).

## 3. Coverage gate + evidence capture — audit `scripts/git_hooks/pre-push`
- Step 1 discovers the sibling `test_<module>.py` (+ `tests/`/`test/` subdir) for each changed SOURCE file and
  runs them under `.venv_gpu_encode` with `--cov-branch`; step 2 runs `diff-cover --fail-under=100` (C0 line) and
  `--branch-coverage --fail-under=95` (C1 branch), scoped to committed push files via `--include $ALLPY`.
- Test-only push → documented **N/A** (no source lines to gate); source-changed-but-no-coverage → **fail-closed**.
- Entry-driver `main()` = `# pragma: no cover`. ruff now diffs `@{upstream}` (ran silently as "na" before).
- Every push writes `gate_results/<sha>.json` (cov_c0, cov_c1, test_count, ruff, lessons, evidence_log) + a full
  log `docs/reports/gate_logs/<sha>.txt`.
- **To challenge:** confirm an uncommitted stray edit does NOT pollute the gate (the `--include` scoping);
  confirm a source edit with no test would block on C0<100.

## 4. Lessons-regression suite — `tests/test_lessons_regression.py` (runs every push)
7 documented lessons codified as invariants against the REAL shipped functions (DirAcc=sign-of-changes,
temporal-split-chronological + NaT/dup fail-loud, Parkinson=variance-not-std, QLIKE-shared-floor,
date-clustered-DM-not-overstating, normalizer-applied-round-trip). **To challenge:** confirm each imports/exercises
the shipped function (no re-implementation).

## 5. Still open — OWNER DECISIONS (not defects)
1. **Populate over/under-fit evidence for the DELIVERED panels.** The capture + gate are in place for future
   runs; the existing `results/masked_rich_floor1e2/*` predate the mandate (test-only) and are not retro-checked.
   Populating them requires a re-run (5 panels × 4 horizons × 5 seeds) that rewrites audit-linked `result.json` —
   deferred pending owner approval.
2. **Migrate the paper `.tex` to include `docs/paper/generated/final_tables.tex`.** The authoritative generator
   exists and its output matches the current hand-checked tables (reviewer verified 80/80); wiring the `.tex` to
   `\input` the generated file (and archiving legacy paper artifacts) is a mechanical follow-up.

## 6. Commit map (this review cycle)
`f49bca2` triage · `88ce8fb` hardening · `71f3c61` R-round · `ce66f70` multi-agent+gate · `9e6b74e` evidence
capture · `8c510af` F-01/F-02 · `56035b1` F-04/F-05 · `a63b3e9` over/under-fit system · `adb208d` F-03 generator.
No committed `result.json` was modified in any of them (`git show <sha> --name-only | grep result.json` = empty).

## 7. Known caveats (do not re-flag)
- Learned metrics are per-seed means (+std); DM is on the 5-seed ensemble; both stated.
- GARCH is a dominated benchmark; observation-space offset is intentional (F-01).
- VN raw OHLCV not split-adjusted (overnight winsorized); S&P 500 already adjusted.
- Coverage residual on `main()`/CLI drivers is `# pragma: no cover` by design.
