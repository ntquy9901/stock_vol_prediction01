# Task Dashboard Quality-Gate Remediation — Summary

- Date: 2026-08-09 07:45
- Scope (edited): `scripts/task_dashboard/`, `scripts/git_hooks/`, `scripts/quality_gate/` only.
  `.worktrees/` and `archive/` untouched (other agents hold worktrees).
- Driver: `docs/reports/2026-08-09_0715_dashboard_quality_audit.md` §3 + two user
  additions (newest-first ordering; whole-card red highlight for gate-incomplete CODE tasks).
- Discipline: TDD (failing tests first, confirmed RED, then GREEN) + verification-before-completion
  (dashboard regenerated and HTML grep-verified; real pytest/ruff/diff-cover output pasted below).

## What changed

| File | Purpose of change |
|---|---|
| `scripts/task_dashboard/build_dashboard.py` | Fixed the render bug; added Data-quality column; `task_type`+`REQUIRED_GATES` validation; whole-card red `card-incomplete` + `INCOMPLETE` badge; newest-first flat ordering with per-card phase tag; "N of M code tasks fully gated" KPI; per-commit gate-result JSON overlay (`load_gate_results`, `_overlay_gate`, real run wins over hand-entry). |
| `scripts/task_dashboard/task_ledger.json` | Migrated all 19 entries to `task_type` + `{status, detail}` gate form using the audit's per-task classification (3 fully-gated, 5 N/A, 11 gaps); real gaps marked `miss` (render RED). |
| `scripts/task_dashboard/test_build_dashboard.py` | New tests: object+prose render with icon/detail; data-quality column; INCOMPLETE red card on a CODE gap; N/A task not reddened; gate-results overlay (real wins); newest-first; helper edge cases (status derivation, missing dir, invalid JSON, no-match overlay). |
| `scripts/task_dashboard/gate_results/.gitkeep` | New directory the hook writes `<short-sha>.json` into. |
| `scripts/quality_gate/run_quality_gate.py` | Added `write_gate_json()`; `main()` now also emits a per-commit gate JSON (best-effort, never fails the gate). |
| `scripts/quality_gate/test_quality_gate.py` | Tests for `write_gate_json` and the `main()` JSON emission (success + swallowed-failure paths). |
| `scripts/git_hooks/pre-push` | Captures tests/diff-cover%/ruff/pandera/evidently and writes `scripts/task_dashboard/gate_results/<short-sha>.json` so future rows come from real runs, not hand-entry. Schema-fail now sets `FAIL=1` (still blocks) instead of early-exit so the JSON is still emitted. |
| `scripts/git_hooks/README.md` | Documented the per-commit gate JSON step. |
| `docs/reports/task_dashboard.html` | Regenerated. |

## Root cause fixed
`_render_gate_row` mapped a cell to an icon only when its lowercased string was exactly
`{pass,skip,fail,n/a}`; every other value (all prose, all `{status,detail}` objects) fell through
to the muted `—` dash. So `code_review` and `diff_cover` rendered as a dash on nearly every card even
when a review ran or diff-cover hit 87%/100%. New `_resolve_gate` handles object, token, and prose
shapes and never collapses a non-empty value to a bare dash.

## Before / after (gate render)
- Before: `diff_cover` rendered `—` on all 19 cards; the Pandera/Evidently gate had no column at all.
- After: real diff-cover numbers render (`Diff-cover: 87% ...`, `Diff-cover: C0=100% ...`); a
  Data-quality column exists; every CODE task missing a required gate is a full red card.

## Ledger classification (matches audit §4)
- Fully-gated CODE (green): `t-pooled-20ep`, `b-review-masked-gnn`, `knn-sparse-adjacency` = 3.
- INCOMPLETE CODE (red): `quality-gate`, `task-7`, `t0.1-multihorizon`, `task-dashboard`,
  `t0.2-batch-graph`, `t1.1-a1-pooled-vs-commondate`, `t1.2-parsimony-pooled`,
  `data-qa-pandera-evidently`, `quality-enforcement`, `masked-gnn-rerun`, `final-architecture-svg` = 11.
- Legitimately N/A (doc/research/git-op, not reddened): `roadmap`, `consolidated-metrics`,
  `research-gnn-sparse-data`, `a-history-rewrite`, `research-har-benchmark` = 5.
- Header KPI: **Code tasks fully gated: 3 / 14 · 11 INCOMPLETE**.

## Backfill (item 4) — honest status
- `quality-enforcement` carries the one real diff-cover number that exists (87%, recorded `pass`).
- The remaining pre-tooling CODE tasks are kept `miss` (RED), not faked. Retroactive diff-cover
  was NOT run because it requires checking out historical/force-pushed commit ranges (history was
  rewritten in task `a-history-rewrite`) and the feature-branch commits live in worktrees currently
  held by other agents. Per the audit's own guidance this is recorded honestly as a gap rather than
  a fabricated number. Going forward the pre-push hook writes a real per-commit JSON that the
  dashboard overlays automatically, so new rows are sourced from real runs.

## Verification (real output)

pytest (task_dashboard + quality_gate): `40 passed`.
ruff (`scripts/task_dashboard/ scripts/quality_gate/ scripts/git_hooks/`): `All checks passed!`.
diff-cover on changed lines vs master: **Coverage: 100% (0 missing of 281)** — C0 gate met;
`build_dashboard.py` and both test files 100%, `run_quality_gate.py` 100%.

HTML grep proof (regenerated `docs/reports/task_dashboard.html`):
- Newest first: first card id = `knn-sparse-adjacency` (timestamp 2026-08-09 00:03).
- Red cards: 11 `card-incomplete"` class occurrences; 11 `INCOMPLETE` badges; KPI flag `11 INCOMPLETE`.
- `Diff-cover: 87% (hook, floor 80, target 100 per CLAUDE.md C0)` and
  `Diff-cover: C0=100% on changed lines (agent)` both render (no longer a dash).
- Overlay proof (simulated real hook JSON for commit `48a14bb` at 100%): `quality-gate` diff-cover
  rendered `hook: 100%` and the task flipped from INCOMPLETE to complete — real run wins over the
  hand-entered `skip`. Simulated artifact removed afterward.

## Not run / follow-ups
- `/code-review` (3-layer) not run by this subagent — flagged for the parent per the repo's
  code-review DoD rule (the CLAUDE.md-honest position: it is not git-automatable).
- Feature-branch CODE-task diff-cover backfill deferred (worktrees held by other agents; unsafe to
  check out here).
- Push deferred to the parent (per task brief); the pre-push hook will run the gate and emit the
  first real `gate_results/<sha>.json` on that push.
