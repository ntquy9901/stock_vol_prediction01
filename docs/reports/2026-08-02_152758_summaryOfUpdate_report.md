# Verification gate and evidence-capture findings

- Timestamp: `2026-08-02 15:27:58` (Asia/Bangkok, UTC+07:00)
- Purpose: handoff specification for evaluating and implementing repository verification gates and a project-specific verification skill
- Scope: verification process only; no source code, test, configuration, or existing skill was modified
- Related audit: `docs/reports/2026-08-02_152253_summaryOfUpdate_report.md`

## Conclusion

The current project constitution defines test, smoke, lint, coverage, code-review, and summary-report requirements, but it does not fully guarantee that verification evidence is complete, reproducible, or bound to the exact code and data state being evaluated. A successful verification workflow requires machine-captured command evidence, repository and data fingerprints, acceptance traceability, and clean-environment reproduction for paper results.

The existing general-purpose skills cover review and test analysis. One project-specific orchestration skill, provisionally named `verify-audit-fixes`, is recommended to bind those capabilities into a repeatable evidence-producing gate. The skill should verify fixes and capture evidence; it should not modify implementation code during the verification run.

## Findings: gaps in the current verification rules

### VER-001 — Raw command output is not mandatory

The summary-report rule requires commands and outcomes to be recorded but does not require retention of the original stdout, stderr, exit code, start time, duration, and exact invocation. A textual statement that a command passed is therefore not independently verifiable.

### VER-002 — Evidence is not obligatorily bound to repository state

Verification artifacts are not required to record Git SHA, branch, working-tree status, staged state, or diff hash. Evidence produced before a later code modification can be incorrectly associated with the current state.

### VER-003 — ML evidence is not bound to data and experiment configuration

The current rules do not require a data snapshot hash, ticker-order manifest, split boundaries, target definition, seed list, complete configuration hash, checkpoint hash, or metric-schema version. A stored metric cannot always be reconstructed from the available artifacts.

### VER-004 — The default pytest command does not currently represent the whole project

`pytest.ini` restricts discovery to `src` and `baselines`, while a substantial suite exists under root `tests/`. Repository collection also currently fails because of cross-baseline module collisions. A pass from a baseline-specific test command does not establish repository-level correctness.

### VER-005 — Branch coverage is not an automated evidence gate

The constitution requires changed-line C1 branch coverage of at least 80%, but the documented process relies on manual inspection. It lacks one deterministic command, threshold enforcement, and a retained machine-readable branch-coverage artifact.

### VER-006 — Acceptance traceability is not mandatory

No required artifact currently maps each audit finding or acceptance criterion to the implementation change, validating test, executed command, raw evidence, reviewer decision, and final status.

### VER-007 — Clean-environment reproduction is not required

Local checks can pass because of cached modules, installed packages, generated data, checkpoints, or unrecorded environment state. The rules do not require a clean checkout or isolated-environment reproduction before publication claims are accepted.

### VER-008 — Code-review evidence does not necessarily identify its exact scope

Review reports are not required to record the reviewed commit, diff hash, complete file list, reviewer layers, and disposition of every finding. A historical review can therefore be mistaken for a review of the current implementation.

### VER-009 — A summary report can be mistaken for primary evidence

A summary is an index and conclusion, not proof that commands were run against the stated state. Primary evidence must consist of raw logs, hashes, structured test reports, coverage artifacts, and result manifests.

## Required evidence directory

Each verification run should create one immutable timestamped directory:

```text
docs/reports/evidence/YYYY-MM-DD_HHMMSS/
├── manifest.json
├── git_status.txt
├── git_diff_stat.txt
├── environment.txt
├── pytest_collection.txt
├── pytest_full.txt
├── pytest_smoke.txt
├── coverage_summary.txt
├── coverage.xml
├── diff_cover.txt
├── branch_coverage.json
├── ruff.txt
├── code_review.md
├── acceptance_traceability.csv
└── result_validation.json
```

The directory should be append-only after completion. If a fix changes after evidence is captured, a new timestamped run is required.

## Required manifest schema

`manifest.json` should contain at least:

```json
{
  "verification_timestamp": "ISO-8601 timestamp with timezone",
  "git": {
    "sha": "full commit SHA",
    "branch": "branch name",
    "working_tree_clean": true,
    "diff_hash": "hash of reviewed diff"
  },
  "environment": {
    "python_version": "...",
    "platform": "...",
    "dependency_lock_hash": "..."
  },
  "data": {
    "snapshot_hash": "...",
    "ticker_manifest_hash": "...",
    "ticker_order": ["..."],
    "train_end": "YYYY-MM-DD",
    "validation_end": "YYYY-MM-DD",
    "test_end": "YYYY-MM-DD"
  },
  "experiment": {
    "config_hash": "...",
    "checkpoint_hashes": ["..."],
    "seeds": [42, 123, 2026],
    "metric_schema_version": "..."
  },
  "commands": [
    {
      "command": "python -m pytest -q",
      "started_at": "ISO-8601 timestamp",
      "duration_seconds": 0.0,
      "exit_code": 0,
      "stdout_file": "pytest_full.txt",
      "stderr_file": "pytest_full.txt"
    }
  ]
}
```

Every referenced file must exist. Hashes must be computed rather than manually copied from prose reports.

## Required traceability schema

`acceptance_traceability.csv` should contain one row per audit finding or acceptance criterion with these fields:

```text
finding_id,severity,requirement,status,changed_files,test_ids,commands,evidence_files,review_disposition,notes
```

Allowed status values:

- `Verified fixed`
- `Partially fixed`
- `Not fixed`
- `Not verifiable`
- `Not applicable`, with a required justification

A finding must not be marked `Verified fixed` unless the cited test and command both passed and their raw evidence files exist.

## Mandatory verification gates after fixes

### Gate 1 — Repository identity

- Record full Git SHA, branch, working-tree status, and reviewed diff hash.
- A dirty worktree must be rejected or every modification must be explicitly included in the verification scope.

### Gate 2 — Static repository checks

- `git diff --check` passes.
- Ruff passes over the configured non-vendored scope.
- Hardcoded path, bare-except, random-split, and duplicate module-name scans are retained as evidence.

### Gate 3 — Test discovery

- `python -m pytest --collect-only -q` exits successfully.
- Root `tests/`, `src/`, and `baselines/` are all included.
- Collected-test count and per-directory distribution are recorded.

### Gate 4 — Full tests

- `python -m pytest -q` exits successfully.
- No collection errors, failed tests, unexpected skips, or xfails are hidden.
- Machine-readable JUnit XML and raw output are retained.

### Gate 5 — Smoke tests

- `python -m pytest -m smoke -q` exits successfully.
- At least one marked smoke test boots and executes the happy path for each publication-relevant model lineage or canonical runner.
- The gate must fail when zero smoke tests are selected.

### Gate 6 — Coverage

- Changed executable lines achieve C0 line coverage of 100%.
- Changed branches achieve C1 coverage of at least 80% using an automated threshold.
- Coverage XML/JSON, diff-cover output, and uncovered changed branches are retained.

### Gate 7 — Regression evidence for audit findings

- Leakage tests demonstrate that scalers are fitted on train-only observations.
- Date-alignment tests use ticker/date keys and verify that a common sample represents one trading date across all stocks.
- Directional-accuracy tests use a hand-calculated panel fixture and reject flattened cross-ticker transitions.
- Temporal-split tests verify chronological boundaries and reject random partitioning.
- Import-isolation tests collect baseline suites in different orders and prove module identity does not change.
- JSON tests reject NaN and Infinity.

### Gate 8 — ML result provenance

- Each canonical run records Git SHA, data and ticker-manifest hashes, split dates, seed, configuration, checkpoint hash, command, and metric-schema version.
- Smoke results and full-training results use distinguishable schemas or explicit flags.
- All six mandatory metrics and panel-aware directional accuracy are present.

### Gate 9 — Statistical verification for paper claims

- Principal models use the same data snapshot, split, seed list, epoch policy, and metric implementation.
- At least three to five seeds are run for the headline model and controls.
- Mean, standard deviation, confidence intervals, and a predefined paired or block-bootstrap comparison are generated.
- Model selection and multiple-comparison policy are documented before the final table is produced.

### Gate 10 — Adversarial review

- Blind Hunter, Edge Case Hunter, and Acceptance Auditor review the exact final diff.
- The review artifact records commit SHA, diff hash, files, findings, and disposition.
- No critical or major finding remains open before verification completes.

### Gate 11 — Clean reproduction

- Publication artifacts are regenerated from a clean checkout and isolated environment.
- The canonical comparison table must match the approved results within predefined numerical tolerances.
- Any dependency on external data, GPU behavior, or unavailable infrastructure is explicitly documented.

## Existing skills that should be composed

The following available skills already provide relevant review capabilities:

- `bmad-code-review`: three-layer adversarial code review.
- `bmad-testarch-trace`: requirements-to-tests and evidence traceability.
- `bmad-testarch-test-review`: adversarial assessment of test quality.
- `bmad-testarch-nfr`: reproducibility, reliability, and non-functional verification.
- `bmad-check-implementation-readiness`: readiness gate before canonical reruns.
- `systematic-debugging`: root-cause workflow when a gate fails.
- `scientific-critical-thinking`: validity review for scientific and statistical claims.

These skills should remain independent reviewers. The proposed project-specific skill should orchestrate them and capture their outputs rather than reimplementing their logic.

## Proposed project-specific skill: `verify-audit-fixes`

### Objective

Verify a supplied set of audit findings against a specified repository state, run the required gates, retain primary evidence, and produce a traceable verdict without modifying implementation code.

### Inputs

- Audit report path.
- Target commit or explicitly scoped dirty-worktree diff.
- Optional finding IDs; default is all unresolved findings.
- Canonical test, lint, coverage, smoke, and experiment commands.
- Data and ticker manifests.
- Required evidence output directory.

### Required behavior

1. Parse finding IDs and acceptance requirements.
2. Capture repository, environment, data, and experiment fingerprints before running checks.
3. Execute commands without concealing nonzero exit codes.
4. Store raw output and structured reports under one timestamped evidence directory.
5. Validate that every manifest reference exists and matches its recorded hash.
6. Generate the acceptance traceability matrix.
7. Invoke the applicable review/test skills against the exact target state.
8. Assign one allowed verification status to every finding.
9. Refuse a successful verdict when required evidence is absent, stale, generated against another commit, or internally inconsistent.
10. Generate a concise summary that links to primary evidence rather than duplicating it.

### Explicit non-goals

- Do not implement fixes.
- Do not alter tests to make failing behavior pass during verification.
- Do not start unapproved training longer than the project training policy allows.
- Do not treat historical coverage, result JSON, or prose reports as evidence for the current commit.
- Do not mark a finding fixed based only on code inspection when executable verification is required.

### Completion criteria

The skill may report `Verification passed` only when:

- Every mandatory gate passes.
- Every critical and high-severity audit finding is `Verified fixed` or has an approved `Not applicable` justification.
- The evidence manifest validates successfully.
- Review scope matches the verified commit or diff.
- The clean-reproduction requirement is satisfied for publication results.

Otherwise, it must report `Verification failed` or `Verification incomplete` and identify the exact missing or failing evidence.

## Recommended implementation order for the next AI

1. Review this specification and the related audit report.
2. Decide whether `verify-audit-fixes` should be a repository-local skill or a reusable plugin skill.
3. Write its specification and tests before implementation.
4. Add a deterministic evidence-capture runner with no training side effects.
5. Make test discovery, smoke, lint, and coverage gates executable and machine-verifiable.
6. Add finding-to-test traceability.
7. Exercise the skill first against a deliberately failing repository state to confirm it cannot produce a false pass.
8. Exercise it again after fixes and validate all stored evidence and hashes.

## Review and verification of this document change

- Change scope: one new Markdown report only.
- Source code behavior: unchanged.
- Tests and coverage: not run because the change only records review findings and does not modify behavior.
- Lint: not run because the configured Ruff scope does not cover Markdown.
- Code review: the document consolidates findings from the completed Blind Hunter, Edge Case Hunter, and Acceptance Auditor reviews; no additional implementation diff was reviewed.
- Risk: the proposed paths and schemas are design recommendations and require specification review before implementation.

