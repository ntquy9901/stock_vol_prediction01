# Task dashboard

Records every completed task in the autonomous Track A+B campaign as an
evidence-backed entry, and aggregates all entries into one self-contained HTML
dashboard the user opens later (double-click, no server, no external deps).

## Files

| File | Purpose |
|------|---------|
| `task_ledger.json` | Canonical machine-readable ledger — a JSON array of task entries. |
| `build_dashboard.py` | Reads the ledger and writes `docs/reports/task_dashboard.html`. |
| `test_build_dashboard.py` | Pytest tests for the generator. |
| `conftest.py` | Makes the module importable when pytest runs from the repo root. |

## Regenerate the dashboard

```bash
python scripts/task_dashboard/build_dashboard.py
```

Writes `docs/reports/task_dashboard.html` (self-contained; open by
double-clicking). Light and dark themes are both supported; a toggle button in
the header persists the choice in `localStorage`, and the page otherwise follows
the OS `prefers-color-scheme`.

## Run the tests

```bash
python -m pytest scripts/task_dashboard/ -q
ruff check scripts/task_dashboard/
```

## Add a task entry

Append one object to the array in `task_ledger.json`, then regenerate. Every key
is optional at render time (missing keys degrade gracefully — a malformed entry
still renders a card rather than crashing the build), but a complete entry looks
like this:

```json
{
  "id": "task-7",
  "phase": "Phase 1 - Headline Track B experiments",
  "title": "Pooled news-GNN G0/G1 graph ablation",
  "status": "done",
  "timestamp": "2026-08-08T16:21",
  "branch": "feature/pooled-news-gnn-pilot",
  "commits": ["8055413", "702289f"],
  "skills_applied": ["systematic-debugging", "TDD-red-green", "verification-before-completion"],
  "evidence": [
    {"cmd": "python -m pytest .../test/ -q", "result": "92 passed in 15.23s"}
  ],
  "quality_gate": {"tests": "pass", "lint": "pass", "code_review": "pass", "diff_cover": "skip"},
  "code_review": {"layers": 3, "findings_fixed": 0, "findings_deferred": 0},
  "dod": [
    {"item": "pytest green (92 passed)", "ok": true}
  ],
  "result_summary": "One-line outcome of the task.",
  "report_md": "docs/reports/2026-08-08_1621_summaryOfUpdate_report.md"
}
```

## Schema

Each entry in the JSON array:

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Short unique task id. |
| `phase` | string | Roadmap phase (see `docs/reports/2026-08-08_1700_track_ab_unified_roadmap.md`). Groups cards on the dashboard. |
| `title` | string | Human-readable task title (card heading). |
| `status` | `done` \| `running` \| `blocked` \| `planned` | Drives the status badge and the overview counts/bar. Any other value folds into `planned`. |
| `timestamp` | string | ISO-ish timestamp of the entry. |
| `branch` | string | Git branch the work lives on. |
| `commits` | string[] | Commit SHAs (short) for the task. |
| `skills_applied` | string[] | Skills used (e.g. `TDD-red-green`, `systematic-debugging`, `verification-before-completion`, `3-layer-code-review`). Rendered as chips. |
| `evidence` | `{cmd, result}[]` | REAL captured command output snippets. `cmd` is the command; `result` is the pasted output. |
| `quality_gate` | `{tests, lint, code_review, diff_cover}` | Each value is `pass` / `skip` / `fail` / `n/a`; rendered with pass/skip/fail styling. |
| `code_review` | `{layers, findings_fixed, findings_deferred}` | Review depth + finding counts. |
| `dod` | `{item, ok}[]` | Definition-of-Done checklist; `ok` is a boolean. |
| `result_summary` | string | One-line outcome shown under the title. |
| `report_md` | string | Path to the task's markdown report. |

## Reporting convention for future tasks

When a campaign task reaches "done" (under the CLAUDE.md Definition of Done),
append an entry to `task_ledger.json` with **real** captured evidence — re-run
the verification command and paste its actual output into `evidence[].result`;
do not copy numbers you did not reproduce. Then run
`python scripts/task_dashboard/build_dashboard.py` to refresh the dashboard.
In-flight work can be recorded with `status: "running"` so the dashboard shows
it as in progress.
