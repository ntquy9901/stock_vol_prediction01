# Task Dashboard Quality-Gate Coverage Audit

- Date: 2026-08-09 07:15
- Scope: `scripts/task_dashboard/` (ledger + generator), `docs/reports/task_dashboard.html`,
  `scripts/git_hooks/pre-push` + `README.md`, `scripts/quality_gate/run_quality_gate.py`,
  `CLAUDE.md` (Definition of Done + Testing quality rules ENFORCED).
- Mode: read-only investigation. No source file was modified. This report is the only file written.
- Ledger size at audit time: 19 entries (verified `python -c "len(json.load(...))" == 19`).

---

## 1. Per-task gate-coverage table

Classification legend:
- **task_type**: CODE (ships/changes `.py` that is testable) · DOC (docs/report/planning) ·
  RESEARCH (web research doc) · GIT-OP (history/repo operation, no code).
- Per gate: **RUN** (evidence of a real run in the ledger) · **N/A** (legitimately not applicable
  to this task_type) · **MISS** (a code-changing task where the gate is applicable but was skipped,
  deferred, or only a weaker substitute was run).
- `review` column distinguishes **3L** (full 3-layer `/code-review`: Blind + Edge + Acceptance) from
  **focused** (a single-angle adversarial/self pass) from **N/A**.
- `data` column = Pandera + Evidently data-quality gate (a DoD-mandated gate the dashboard does not
  currently render at all — see §2).

| # | task id | task_type | tests | lint | review | diff_cover | data (pandera/evidently) |
|---|---------|-----------|-------|------|--------|------------|--------------------------|
| 1 | quality-gate | CODE | RUN (13) | RUN | 3L (4 fixed) | **MISS** (`skip`) | RUN (it is the gate itself) |
| 2 | task-7 | CODE | RUN (92) | RUN | 3L (0 fixed) | **MISS** (`skip`) | **MISS** (model/data change, not run) |
| 3 | roadmap | DOC | N/A | N/A | focused (1L) | N/A | N/A |
| 4 | t0.1-multihorizon | CODE | RUN (110) | RUN | **focused** | **MISS** (`not run (tooling gap)`) | **MISS** (manifest/horizon path) |
| 5 | task-dashboard | CODE | RUN (9) | RUN | **focused** (escaping) | **MISS** (`not run (tooling gap)`) | N/A (no data) |
| 6 | t0.2-batch-graph | CODE | RUN (117) | RUN | **focused** | **MISS** (`not run (tooling gap)`) | **MISS** (graph builder/data) |
| 7 | t1.1-a1-pooled-vs-commondate | CODE | RUN (125) | RUN | focused (adversarial-subagent, 3 LOW) | **MISS** (`not run (tooling gap)`) | **MISS** (new common-date manifest) |
| 8 | t1.2-parsimony-pooled | CODE | RUN (6) | RUN | **focused** (reproduced-analysis) | **MISS** (`not run (tooling gap)`) | N/A (reuses A1 cells) |
| 9 | data-qa-pandera-evidently | CODE | RUN | N/A | N/A | N/A | RUN (34/34 + drift) |
| 10 | consolidated-metrics | DOC | N/A | N/A | N/A | N/A | N/A |
| 11 | quality-enforcement | CODE | RUN | RUN | **focused** (self) | **RUN** (87%, below 100 target) | RUN (hook fired) |
| 12 | research-gnn-sparse-data | RESEARCH | N/A | N/A | N/A | N/A | N/A |
| 13 | t-pooled-20ep | CODE | RUN | RUN | **focused** (verification-caught) | **RUN** (hook) | RUN (hook) |
| 14 | masked-gnn-rerun | CODE | RUN (134) | RUN | **MISS** (self+tests; 3L deferred) | RUN (hook, floor 80) | RUN (34/34 + drift) |
| 15 | a-history-rewrite | GIT-OP | N/A | N/A | N/A | N/A | N/A |
| 16 | b-review-masked-gnn | CODE (review) | RUN (135) | RUN | **3L** (Blind/Edge/Acceptance) | RUN (hook) | RUN (hook) |
| 17 | research-har-benchmark | RESEARCH | N/A | N/A | N/A | N/A | N/A |
| 18 | final-architecture-svg | CODE (generator) | N/A (diagram) | RUN | **focused** (self) | N/A (plot script) | N/A |
| 19 | knn-sparse-adjacency | CODE | RUN (154) | RUN | **3L** (agent, 1 HIGH fixed) | RUN (C0=100%, agent claim) | RUN (34/34 + drift) |

Notes on individual rows verified against the ledger:
- Rows 14+16 are one logical unit: `masked-gnn-rerun` (row 14) explicitly records `"3-layer /code-review" ok:false` and `findings_deferred:1`; `b-review-masked-gnn` (row 16) is the deferred 3-layer review being completed later. So the masked-GNN code is 3-layer-reviewed **in aggregate**, but the per-row dashboard shows row 14 as review-incomplete with no link to row 16.
- Rows 9 and 10 have empty `commits: []` — no commit provenance recorded for those tasks.
- Row 18 (`final-architecture-svg`) is a matplotlib generator script; treated here as legitimately N/A for tests/diff-cover (plot-only, no logic branches), consistent with how the project treats diagram generators. Review is self-only.

---

## 2. Root-cause diagnosis: why "mostly only tests/lint show green"

Six candidate causes were named in the task brief. Evidence-checked verdicts:

**(a) Many entries are DOC/RESEARCH/GIT-OP where code gates are genuinely N/A — TRUE.**
Rows 3, 10, 12, 15, 17 (5 of 19) are DOC/RESEARCH/GIT-OP. Their `n/a` on tests/lint/diff-cover is
correct, not a gap.

**(b) Early code tasks predate the diff-cover install — TRUE.**
`quality-enforcement` (row 11) records `pip install ... diff-cover 10.4.1 installed` and is the first
task to run diff-cover live (87%). Every CODE task before it (rows 4–8) records diff_cover =
`"not run (tooling gap)"`, and the two earliest (rows 1–2) record `"skip"`. This is the single
largest source of MISS marks: 7 code tasks have no diff-cover number because the tool did not exist
when they ran.

**(c) Some code tasks got a FOCUSED adversarial review instead of full 3-layer — TRUE.**
Only 3 of the CODE tasks record a genuine 3-layer review with all three named layers:
row 16 (`b-review-masked-gnn`: Blind/EdgeCase/Acceptance) and row 19 (agent Blind/Edge/Acceptance).
Row 1 and row 2 record `layers: 3` numerically but do not name the three layers. The remaining CODE
tasks record single-angle passes: `focused-adversarial`, `adversarial-escaping`,
`reproduced-analysis`, `verification-caught`, `self`. The ledger `quality-enforcement.result_summary`
itself states: *"some earlier tasks got focused-adversarial review, not full 3-layer."*

**(d) The ledger is HAND-WRITTEN by the parent per task, so fields are inconsistent and not sourced
from real gate output — TRUE (and this is the deepest cause).** Evidence of hand-entry inconsistency:
- `code_review.layers` is typed three different ways across entries: an integer (`3`, rows 1–2), a
  list of named layers (rows 16, 19), a list with one free-text string (rows 4–8, 11, 13, 14), and an
  empty list (rows 9, 10, 12, 15, 17).
- `quality_gate.code_review` and `quality_gate.diff_cover` hold free-text prose, not status tokens —
  e.g. `"focused adversarial: horizon threaded on all paths, guard raises"`,
  `"NOW RUN: 87% Track B (debt), floor 80 enforced, target 100"`, `"C0=100% (agent)"`.
- `quality_gate.tests` varies: `"pass"`, `"pass (134)"`, `"pass (after conftest fix)"`, `"n/a (doc)"`.
- No file `*_gate_result*.json` exists anywhere in the tree (verified by `find`). The dashboard's
  gate rows are therefore not sourced from any machine artifact — they are re-typed by hand.

**This free-text hand-entry is also the direct rendering cause of "only tests/lint show."**
`build_dashboard.py:109-119` (`_render_gate_row`) maps a cell to an icon only when its lowercased
string is exactly one of `{"pass","skip","fail","n/a"}` (`GATE_ICON`, line 26); anything else falls
through to `state = "n/a"` and renders the muted `—` dash. Consequences:
- `tests` and `lint` almost always hold the canonical token `"pass"` → they render a green ✔.
- `code_review` holds prose on nearly every card → renders `—` (muted) on nearly every card, hiding
  the fact that a review happened.
- `diff_cover` holds `"skip"`/`"n/a"` (dash) on the pre-tooling tasks and prose
  (`"NOW RUN: 87%..."`, `"C0=100% (agent)"`, `"enforced via pre-push hook"`) on the later ones — so
  **diff_cover renders as a dash on all 19 cards**, even the ones where it genuinely ran at 87% or
  100%. The real numbers are invisible.
So the visual impression "mostly only tests + lint are green" is produced jointly by (b/c/d) *and* by
a renderer that silently downgrades any non-token string to a dash.

**(e) The pre-push hook runs at PUSH time and its pass/fail is NOT captured back into the per-task
ledger — TRUE.** `scripts/git_hooks/pre-push` prints results to stderr and writes only a `drift.html`
into `results/quality_gate/prepush_<ts>/` (verified: those dirs contain `drift.html` and nothing
else). It emits no per-commit JSON. `run_quality_gate.py:245-278` writes a human `*_quality_gate_report.md`
but again no JSON. Nothing links a hook run to a ledger `id`. The ledger's diff-cover / data-gate
cells are hand-copied from console output, not read from the hook.

**(f) 3-layer code-review is not git-automatable — TRUE.** Confirmed by `scripts/git_hooks/README.md`
lines 25-26 ("What it does NOT enforce ... Code review (3-layer adversarial) ... the hook cannot
perform an LLM review") and by `quality_gate.code_review` in row 11. The hook has no step for it.

**Additional finding not in the brief — the dashboard has no data-quality (Pandera/Evidently) column
at all.** `GATE_FIELDS` (`build_dashboard.py:27`) is exactly `[tests, lint, code_review, diff_cover]`.
The Pandera+Evidently gate — a DoD-mandated gate that actually ran on rows 9, 11, 13, 14, 16, 19 — is
never surfaced as a gate cell. It only appears buried in free-text `evidence`. So even when the data
gate runs, the dashboard cannot show it.

---

## 3. Remediation — guarantee 100% gate coverage + accurate reporting

Design priority order. Each item lists exact files/functions and a rough effort estimate.

### (i) Make the pre-push hook emit a machine-readable per-commit gate JSON that the dashboard reads (highest leverage — kills cause (d)+(e))
Source the gate row from real runs instead of hand-entry.

- **`scripts/quality_gate/run_quality_gate.py`**: add `write_json(results, timestamp, meta)` next to
  the existing `write_report()` (line 245). `CheckResult` is already a `@dataclass(name,status,detail)`
  (line 53-57), so serialization is trivial: dump `{tests, diff_cover_pct, ruff, pandera, evidently,
  overall, commit, branch, timestamp}` to `results/quality_gate/gate_results/<commit>.json`.
- **`scripts/git_hooks/pre-push`**: after step 4 (line 88), capture the numbers it already computes —
  pytest pass count, the `diff-cover ... --fail-under` percentage (line 53), ruff exit, and the
  Pandera/Evidently `s.status`/`d.status` (lines 80-83) — and write them to
  `results/quality_gate/gate_results/$(git rev-parse HEAD).json`. The hook already knows `HEAD` and
  `BASE`; this is ~15 lines of `printf`/`jq`-free JSON emission plus reusing the existing Python
  heredoc to append its statuses.
- **`scripts/task_dashboard/build_dashboard.py`**: add `load_gate_results()` that, for each ledger
  entry, looks up `results/quality_gate/gate_results/<commit>.json` by the entry's `commits[]` and
  overlays the real gate statuses onto the hand-entered `quality_gate` dict (real value wins;
  hand-entry is fallback only). `_render_gate_row` then renders tokens, not prose.
- Effort: ~0.5 day (hook JSON emission + a `check_diffcover`/`check_ruff` `CheckResult` so the JSON is
  complete + dashboard loader + tests).

### (ii) Add `task_type` + `required_gates` schema so the dashboard VALIDATES completeness and flags missing gates RED
Today `_render_gate_row` is a pure pass-through with no notion of "this gate was required and is
missing." Add the concept.

- **`scripts/task_dashboard/task_ledger.json`**: add a `"task_type"` field to every entry
  (`code` | `doc` | `research` | `git-op`).
- **`build_dashboard.py`**: define `REQUIRED_GATES = {"code": {"tests","lint","code_review",
  "diff_cover"}, "doc": set(), "research": set(), "git-op": set()}` (+ `data` required when the
  entry's evidence/commit touches data/manifest — reuse the hook's `DATA_TOUCHED` regex). In
  `_render_gate_row`, for each required gate whose resolved status is not `pass`/`n/a`, render a RED
  `MISSING` cell and set a card-level `incomplete` flag; propagate a new `card-incomplete` CSS class
  (mirrors existing `card-blocked`, `_CSS` line 325) so incomplete CODE tasks are visually RED.
- Add a header KPI: "N of M code tasks fully gated" computed in `build_html` (line 235).
- Effort: ~0.5 day (schema field + validation function + CSS + tests). Pairs naturally with (i).

### (iii) Policy + mechanism for the non-automatable 3-layer review (cause (f))
The hook cannot run an LLM review, so make its presence a first-class, dashboard-checked field.

- Add a required ledger field `code_review.type` ∈ {`3-layer`, `focused`, `none`} plus
  `code_review.review_log` = path to the `code_review/code_review_<date>.md` artifact
  (the baseline-folder convention already mandated by CLAUDE.md §3.F).
- **`build_dashboard.py` `_render_review`** (line 164): render `focused` and `none` on a CODE task as
  an amber/RED flag ("focused only — 3-layer required") and only `3-layer` with a resolvable
  `review_log` path as green. This turns the "focused vs 3-layer" distinction (cause (c)) into a
  visible, enforced state instead of prose.
- Lightweight cross-link: allow `code_review.superseded_by` = another task id (e.g. row 14 →
  row 16) so a deferred-then-completed review is not shown as a permanent gap.
- Effort: ~0.25 day (fields + render logic + one test).

### (iv) One-time backfill of the existing 19 entries
- For the 7 pre-tooling CODE tasks (rows 1,2,4,5,6,7,8) that record `skip`/`not run (tooling gap)`:
  check out each recorded commit range and run `diff-cover .qg_coverage.xml --compare-branch=<base>`
  to compute the real changed-line coverage retroactively, then write the number into the new
  gate-result JSON (or mark `n/a` where the task genuinely changed no testable `.py`, e.g. row 5's
  data column, row 8/row 18 already argued N/A). This converts each MISS into either a real number or
  a justified N/A.
- Re-classify each entry's `task_type` per the §1 table (5 are DOC/RESEARCH/GIT-OP → their code-gate
  cells legitimately become explicit `n/a`, not blank).
- For the focused-review CODE tasks, either (a) record `code_review.type="focused"` honestly (dashboard
  will flag), or (b) run the deferred 3-layer `/code-review` and attach a review-log — a decision per
  task, not automatable.
- Effort: ~0.5–1 day (mostly re-running diff-cover on old commit ranges; some ranges span the
  force-pushed history from row 15, so use the retained backup branches).

**Files touched by the full remediation:** `scripts/quality_gate/run_quality_gate.py`,
`scripts/git_hooks/pre-push`, `scripts/task_dashboard/build_dashboard.py`,
`scripts/task_dashboard/task_ledger.json`, plus `scripts/task_dashboard/` tests and
`scripts/git_hooks/README.md` doc update. Total rough effort: ~2–2.5 days.

---

## 4. Bottom line (of the 19 tasks)

- **Genuinely fully-gated (every applicable gate RUN, incl. real diff-cover and adequate review): 3.**
  Rows 13 (`t-pooled-20ep`), 16 (`b-review-masked-gnn`), 19 (`knn-sparse-adjacency`). Row 11
  (`quality-enforcement`) is borderline: it ran diff-cover (87%, below the C0=100 target) and only a
  self-review — counted below as a real gap on review, not here.
- **Legitimately N/A (DOC/RESEARCH/GIT-OP with no applicable code gate): 5.** Rows 3, 10, 12, 15, 17.
- **Real gaps needing backfill (CODE tasks with a missing/weaker gate): 11.** Rows 1, 2, 4, 5, 6, 7,
  8, 9, 11, 14, 18. Dominant missing gate is **diff-cover** (7 pre-tooling tasks: rows 1,2,4,5,6,7,8),
  followed by **review depth** (focused-only where 3-layer is the CLAUDE.md standard: rows 4,5,6,8,11,
  14,18) and **data-quality gate not recorded** on data-touching tasks (rows 2,4,6,7).

The gaps are concentrated, not random: everything before `quality-enforcement` (row 11) predates the
diff-cover tooling, and the review-depth gaps trace to focused reviews being recorded as-run rather
than escalated to 3-layer. The remediation in §3 removes the mechanism that let these through (hand
entry with no validation) by sourcing gate cells from the hook's real output and having the dashboard
flag any CODE task that is missing a required gate.
