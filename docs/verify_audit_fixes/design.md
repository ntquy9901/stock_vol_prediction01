# verify-audit-fixes — requirements & design (lightweight, per CLAUDE.md right-sizing)

## Requirements

- Bind "this audit finding is fixed" claims to machine-captured, reproducible evidence instead of
  prose assertions — full rationale in `docs/reports/2026-08-02_152758_summaryOfUpdate_report.md`.
- Must not modify implementation code, alter tests to pass, or start unapproved training.
- Must never report a gate as passed without actually running it; gates that can't run yet
  (7-11 — no per-finding regression tests, ML provenance schema, multi-seed framework,
  adversarial-review integration, or clean-room repro exist in this project) must say so
  explicitly (`Not verifiable — <reason>`), never silently or as a fake pass.

## Design

- `scripts/verify_audit_fixes/` — plain, testable Python (no Claude-specific API), so it runs the
  same from a terminal or from a Claude session: `commands.py` (subprocess wrapper capturing
  stdout/stderr/exit code/duration), `gates.py` (Gates 1-6, each a pure function returning a dict +
  writing its own evidence files), `manifest.py` (assembles `manifest.json`, documents Gates 7-11
  as `Not verifiable` with per-gate reasons), `traceability.py` (validates + writes
  `acceptance_traceability.csv`; the schema's own gate is that `Verified fixed` requires
  test_ids+commands+evidence_files present), `static_scans.py` (grep-based hardcoded-path/
  bare-except/random-split/duplicate-module-name scans), `run_verification.py` (CLI orchestrator).
- `.claude/skills/verify-audit-fixes/SKILL.md` — the session-facing process: when to invoke, how
  to call the runner, how to build the findings JSON, and the non-goals a session must hold to
  even under pressure to "just mark it verified."
- Evidence directories are immutable per run (`docs/reports/evidence/<timestamp>/`) — a changed
  fix gets a new directory, never an overwrite, so evidence can't silently go stale in place.

## Simplicity/Anti-Abstraction gate check (CLAUDE.md §5)

- No new framework/config system introduced — plain functions + argparse, reusing subprocess/git/
  pytest/ruff/diff-cover directly rather than wrapping them in a new abstraction layer.
- Gates 7-11 deliberately NOT stubbed with fake passing logic — implementing them for real requires
  infrastructure (ML provenance store, multi-seed harness) this project doesn't have yet; building
  that is out of scope here and would be exactly the kind of premature abstraction §2 warns against.

## Tests

`tests/verify_audit_fixes/` — 54/54 passing (`pytest tests/verify_audit_fixes/ -v`). Covers each
gate module in isolation plus one end-to-end CLI integration test against a minimal synthetic git
repo (`test_run_verification.py`).
