# Summary of Update — Post-code quality gate

## What changed

Added a standalone post-code quality gate under `scripts/quality_gate/` that runs the project's
quality tools after code changes and returns a single PASS/FAIL with a non-zero exit code on hard
failure. Built by an independent agent, then hardened against an adversarial 3-layer code review.

## Files (path → purpose)

- `scripts/quality_gate/run_quality_gate.py` — orchestrator: LINT (ruff), TESTS (pytest + smoke),
  SCHEMA (pandera), DRIFT (evidently, informational). `--fast` skips DRIFT. Exit non-zero iff a HARD
  check (LINT/TESTS/SCHEMA) fails.
- `scripts/quality_gate/data_schemas.py` — pandera schemas + `validate_data()` for the processed
  per-ticker CSVs and the news panel, based on the actual inspected columns.
- `scripts/quality_gate/test_quality_gate.py` — 13 pytest tests (tmp fixtures + monkeypatch, no
  dependency on real training data).
- `scripts/quality_gate/README.md` — usage, checks, exit-code meaning.

## Tools installed for this (verified working on the current env)

- pandera 0.32.1, evidently 0.7.21 (both verified on pandas 3.0). ruff 0.16.1 and pytest 9.1.1 were
  already present. Deepchecks, DVC, MLflow were evaluated and NOT adopted (footprint / redundancy).

## Code review (3-layer adversarial) + actions

Ran Blind Hunter, Edge Case Hunter, and Acceptance Auditor in parallel over `scripts/quality_gate/`.
Eight distinct findings; all HIGH/MEDIUM fixed, one LOW deferred. One reported item (drift
reference/current argument order) was a false alarm — verified correct — and left unchanged.

| # | Finding | Sev | Action |
|---|---|---|---|
| 1 | Missing/empty data input reported as SCHEMA FAIL; the SKIPPED branch was dead code | HIGH | `validate_data` now returns 3-state `pass/fail/skip`; missing dir/CSVs/panel → SKIP; `check_schema` SKIPs when no artifact present |
| 2 | An all-NaN embedding column failed the whole news panel (contradicts documented all-NaN tolerance) | MED | Restricted the all-NaN check to required columns (`ticker`, `date`); embedding columns tolerated |
| 3 | Missing pandera crashed the orchestrator at import time | MED | `validate_data` imported lazily inside `check_schema`; ImportError → SCHEMA SKIPPED; data paths defined dependency-free |
| 4 | TESTS did not skip when pytest is absent | MED | Detects `No module named pytest` → SKIPPED |
| 5 | ruff `--exclude` replaced ruff's built-in default excludes (would re-lint vendored dirs) | MED | Switched to `--extend-exclude` |
| 6 | A crashing/timing-out ruff was downgraded to SKIPPED (HARD check could silently vanish) | MED | ruff invocation failure → FAIL; only ruff-not-installed → SKIPPED |
| 7 | Report filename recomputed a minute-precision timestamp → same-minute overwrite + mismatch | LOW | Reuses the passed seconds-precision run timestamp |
| 8 | News-panel `date` dtype pinned to `datetime64[us]` (false fail if stored as `ns`) | LOW | Schema no longer pins the unit; validated as datetime-any in code |
| — | Drift `run(current, reference)` argument order | — | False alarm — verified correct, no change |

Four regression tests were added for findings 1, 2, 4, 6 and the timestamped-filename fix (7).

## Commands run (real)

- `python -m pytest scripts/quality_gate/test_quality_gate.py -q` → **13 passed**.
- `python scripts/quality_gate/run_quality_gate.py --fast` → exit **1**; SCHEMA PASS 34/34.
- `python scripts/quality_gate/run_quality_gate.py` (full) → exit **1**; `drift.html` (3.6 MB) produced.
- `ruff check scripts/quality_gate/` → **All checks passed** (the gate code is lint-clean).

## Pre-existing repo debt surfaced by the gate (NOT fixed here — out of scope)

- LINT: 820 ruff errors across existing repo code (excludes `.agents .claude _bmad archive data`).
- TESTS: `tests/test_timesfm_lora.py` fails collection (`import mlflow`; mlflow not installed).

These make the gate exit non-zero today, which is the gate working as intended. Addressing them is a
separate task.

## Follow-ups

- LOW (deferred): the smoke-marker "unregistered" detection is cosmetic and only fires under pytest
  strict-markers; harmless (smoke can never hard-fail).
- Optional: clear or triage the 820 lint errors and the mlflow test-collection failure.
- Optional: wire the gate as a pre-commit/CI hook once the pre-existing debt is resolved.

## DoD checklist

- [x] Code satisfies the request (post-code gate running the new tools), isolated under `scripts/quality_gate/`.
- [x] Tests written/updated and run (13 passed); behavior changes covered by regression tests.
- [x] Lint on changed code: `ruff check scripts/quality_gate/` clean.
- [x] Code review (3-layer adversarial) run; all HIGH/MEDIUM findings fixed, LOW deferred.
- [ ] diff-cover `--fail-under=100`: Not run — diff-cover not installed (documented CLAUDE.md tooling gap).
- [x] Smoke: gate runs end-to-end in both modes with correct exit codes; drift artifact produced.
