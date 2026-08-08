# Quality Gate

Single post-code quality gate for this repository. Runs the project's quality
tools after code changes, prints a PASS/FAIL summary, and writes a timestamped
markdown report.

## Run

```bash
python scripts/quality_gate/run_quality_gate.py          # full (incl. drift)
python scripts/quality_gate/run_quality_gate.py --fast   # skip the drift step
```

## Checks

| Check | Tool | Behaviour |
| --- | --- | --- |
| LINT | `ruff check .` (excludes `.agents .claude _bmad archive data`) | HARD. `SKIPPED` if ruff is not installed. |
| TESTS | `pytest -q` (+ `pytest -q -m smoke` note) | HARD. Unregistered smoke marker or no-tests handled as a note/SKIPPED, not a hard fail. |
| SCHEMA | pandera `validate_data()` over `data/processed/*.csv` and `data/features/dual_group_news_panel.parquet` | HARD. Fails the gate if any schema check fails. |
| DRIFT | evidently `DataDriftPreset` on a temporal split of one processed ticker (rows capped for a light run) | Informational (`INFO`) — NEVER fails the gate. HTML saved to `results/quality_gate/<timestamp>/drift.html`. `--fast` skips it. |

Each check reports `(name, status, detail)` and degrades to `SKIPPED` with a
reason when its tool or input is missing, rather than faking a pass.

## Exit code

- `0` — no HARD check failed.
- non-zero — at least one HARD check (LINT / TESTS / SCHEMA) failed.

DRIFT and any SKIPPED check never cause a non-zero exit.

## Report

Each run writes `docs/reports/<YYYY-MM-DD_HHMM>_quality_gate_report.md` with the
status table and notes.

## Files

- `data_schemas.py` — pandera schemas + `validate_data()`.
- `run_quality_gate.py` — orchestrator (LINT / TESTS / SCHEMA / DRIFT).
- `test_quality_gate.py` — pytest tests for the schemas and orchestrator.
