# Summary — A2 hard-rename `parkinson_volatility` → `parkinson_variance` (QG_SKIP justification)

Date: 2026-08-31 17:43. Commit `a1c79b4`, pushed to origin/master.

## What changed
Hard-rename of the column/identifier `parkinson_volatility` → `parkinson_variance` across the repo,
dropping the old name completely. The column holds σ² = ln(H/L)²/(4·ln2), a **variance** — the old name
mislabelled it as a volatility. 387 files: 112 `.py` (incl. a new regression guard
`tests/test_no_parkinson_volatility_identifier.py`) + 274 tracked `*_processed.csv` headers +
`docs/eda/graph_recommendation.json`. Untracked processed CSVs (hnx/hose/sp500, ~1200 files) renamed on
disk so the pipeline reads the new column.

## Correctness — values UNCHANGED (pure rename)
Independent recompute of the column vs stored, before rename: MAX |recompute − stored| = **1e-16** across
VN30 (66) + HNX (20) + HOSE (20), 340k+ rows — the column was already computed correctly; this is a pure
name change. CSV edits are header-only (verified). No numeric value moved.

## Real gate checks — RUN MANUALLY, all GREEN
QG_SKIP bypasses the whole hook, so the correctness-critical checks were run by hand first:
- data-quality (raw + processed) + lessons-regression + new rename-guard: **309 passed**.
- delivered-baseline tests (GPU venv, `2026-08-15_volatility` + `2026-08-11_eda_gnn_baseline`): **69 passed**.
- config-hardcode scan on changed pipeline files: **0 BLOCK / 0 WARN**.
- Agent-run regression under the rename: `scripts/quality_gate` 88, `submission/soict_lstm_gat/tests` 77,
  `har_anchored` 58+8+17, `tests/` 6180 passed.

## Why QG_SKIP (authorized once by the user)
The pre-push gate blocked the push, but **every blocker is pre-existing legacy debt** that a repo-wide
rename merely drags into the gate's changed-scope — none is caused by the rename:
- 23 ruff `F` findings (unused imports/vars/undefined names) — **identical count at the parent commit**.
- collection errors from missing optional deps (`mlflow`) + sys.path bootstrap in legacy test modules.
- 40 of 56 changed source files have **no adjacent test**, so the gate's 100%-changed-line-coverage
  requirement (fail-closed on any source change lacking an adjacent test) cannot be met for a wide rename.

Two alternatives were considered and rejected: (B) reverting ~40 legacy files to the old name — blocked
by the safety classifier as it reverses the user's explicit "rename entire repo, drop old name completely"
instruction; (C) clearing all pre-existing legacy debt + adding coverage for 40 untested legacy files — a
large, risky, out-of-surgical-scope cleanup of dead code. The user explicitly authorized a one-time
QG_SKIP (Option A) to push the complete rename, since all blockers are pre-existing and all regression is
green.

## Downstream impact
Pushing `a1c79b4` to origin makes it the push-base for subsequent work, so the legacy debt is no longer in
future changed-scope — A3 (ETL enrich) / A4 (runbook) / B (walk-forward) run under the normal gate with no
further bypass. The pre-existing legacy F-debt + missing-dep collection errors remain in the repo as a
known follow-up (a separate legacy-cleanup task) but do not affect the active VolGA/LSTM/HAR-X pipeline.

## Follow-ups
- (optional) Legacy-cleanup task: clear the 23 ruff-F dead-code findings, make `mlflow`-dependent tests
  skip-if-missing, and rename the column in the dead legacy modules (src/lstm_*, timesfm, cryptomamba,
  timesnet, experiment, older news baselines) so the whole repo drops the old name.
