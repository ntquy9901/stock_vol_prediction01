"""CLI entrypoint: run Gates 1-6 for real and produce an evidence directory.

Usage:
    python scripts/verify_audit_fixes/run_verification.py \\
        --repo-root . \\
        --evidence-dir docs/reports/evidence/2026-08-02_180000 \\
        [--audit-report docs/reports/some_audit.md] \\
        [--target-scope "commit abc123"] \\
        [--findings-json findings.json] \\
        [--skip-gate4] [--skip-gate6]

Gates 7-11 are NOT executed here — they require infrastructure this project
does not have yet (see manifest.py's GATES_7_TO_11_NOT_VERIFIABLE). They are
recorded in manifest.json as "Not verifiable" with a specific reason each,
never silently omitted and never faked as a pass. See
.claude/skills/verify-audit-fixes/SKILL.md for how a Claude session should
handle Gates 7-11 and invoke this runner for Gates 1-6.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from verify_audit_fixes import gates, manifest as manifest_mod, traceability
else:
    from . import gates
    from . import manifest as manifest_mod
    from . import traceability


def run_verification(
    *,
    repo_root: Path,
    evidence_dir: Path,
    audit_report_path: str | None = None,
    target_scope: str | None = None,
    findings: list[dict] | None = None,
    cov_targets: tuple[str, ...] = ("src",),
    compare_branch: str | None = None,
    skip_gate4: bool = False,
    skip_gate5: bool = False,
    skip_gate6: bool = False,
    full_test_timeout: int = 3600,
    smoke_test_timeout: int = 1800,
    coverage_timeout: int = 3600,
    force: bool = False,
) -> dict:
    """Run Gates 1-6 against ``repo_root`` and write evidence to ``evidence_dir``.

    Returns the manifest dict. Raises FileExistsError if evidence_dir already
    has files in it and ``force`` is False (evidence directories are meant to
    be append-only per-run, not overwritten).
    """
    repo_root = Path(repo_root).resolve()
    evidence_dir = Path(evidence_dir).resolve()

    if evidence_dir.exists() and any(evidence_dir.iterdir()) and not force:
        raise FileExistsError(
            f"evidence_dir {evidence_dir} already contains files. Evidence "
            "directories are immutable per-run; use a new timestamped "
            "directory or pass force=True."
        )
    evidence_dir.mkdir(parents=True, exist_ok=True)

    git_identity = gates.capture_repository_identity(repo_root, evidence_dir)
    environment = gates.capture_environment(repo_root, evidence_dir)
    gate2 = gates.run_gate2_static_checks(repo_root, evidence_dir)
    gate3 = gates.run_gate3_test_discovery(repo_root, evidence_dir)

    if skip_gate4:
        gate4 = {"status": "not_run", "reason": "Skipped by caller (--skip-gate4).", "commands": []}
    else:
        gate4 = gates.run_gate4_full_tests(repo_root, evidence_dir, timeout=full_test_timeout)

    if skip_gate5:
        gate5 = {"status": "not_run", "reason": "Skipped by caller (--skip-gate5).", "commands": []}
    else:
        gate5 = gates.run_gate5_smoke_tests(repo_root, evidence_dir, timeout=smoke_test_timeout)

    if skip_gate6:
        gate6 = {"status": "not_run", "reason": "Skipped by caller (--skip-gate6).", "commands": []}
    else:
        gate6 = gates.run_gate6_coverage(
            repo_root, evidence_dir, cov_targets=cov_targets, timeout=coverage_timeout, compare_branch=compare_branch
        )

    m = manifest_mod.build_manifest(
        repo_root=repo_root,
        evidence_dir=evidence_dir,
        audit_report_path=audit_report_path,
        target_scope=target_scope,
        git_identity=git_identity,
        environment=environment,
        gate2=gate2,
        gate3=gate3,
        gate4=gate4,
        gate5=gate5,
        gate6=gate6,
    )

    if findings:
        traceability.write_traceability_csv(findings, evidence_dir)
    else:
        # Placeholder so the file exists and the manifest's file list is honest
        # about why it is empty — the skill fills this in from the audit report.
        (evidence_dir / "acceptance_traceability.csv").write_text(
            ",".join(traceability.FIELDS) + "\n"
            "# No findings supplied to this run (--findings-json was not given). "
            "Populate this file from the audit report's findings before claiming "
            "verification is complete.\n",
            encoding="utf-8",
        )

    problems = manifest_mod.validate_manifest(m, evidence_dir)
    m["manifest_validation_problems"] = problems

    manifest_mod.write_manifest(m, evidence_dir)
    return m


def _load_findings(findings_json: str | None) -> list[dict] | None:
    if not findings_json:
        return None
    data = json.loads(Path(findings_json).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"--findings-json must contain a JSON list, got {type(data)}")
    return data


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Gates 1-6 evidence capture for verify-audit-fixes.")
    p.add_argument("--repo-root", default=".", help="Repository root (default: current directory).")
    p.add_argument("--evidence-dir", required=True, help="Output evidence directory (must not already have files).")
    p.add_argument("--audit-report", default=None, help="Path to the audit report being verified.")
    p.add_argument("--target-scope", default=None, help="Description of the commit/diff scope under verification.")
    p.add_argument("--findings-json", default=None, help="JSON file: list of finding dicts for acceptance_traceability.csv.")
    p.add_argument("--cov-targets", nargs="*", default=["src"], help="Packages to measure coverage for (Gate 6).")
    p.add_argument("--compare-branch", default=None, help="diff-cover compare-branch override (default: auto-resolve origin/master/main/HEAD).")
    p.add_argument("--skip-gate4", action="store_true", help="Skip the full test run (Gate 4).")
    p.add_argument("--skip-gate5", action="store_true", help="Skip the smoke test run (Gate 5).")
    p.add_argument("--skip-gate6", action="store_true", help="Skip coverage (Gate 6).")
    p.add_argument("--full-test-timeout", type=int, default=3600)
    p.add_argument("--smoke-test-timeout", type=int, default=1800)
    p.add_argument("--coverage-timeout", type=int, default=3600)
    p.add_argument("--force", action="store_true", help="Allow writing into a non-empty evidence_dir.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    findings = _load_findings(args.findings_json)

    m = run_verification(
        repo_root=Path(args.repo_root),
        evidence_dir=Path(args.evidence_dir),
        audit_report_path=args.audit_report,
        target_scope=args.target_scope,
        findings=findings,
        cov_targets=tuple(args.cov_targets),
        compare_branch=args.compare_branch,
        skip_gate4=args.skip_gate4,
        skip_gate5=args.skip_gate5,
        skip_gate6=args.skip_gate6,
        full_test_timeout=args.full_test_timeout,
        smoke_test_timeout=args.smoke_test_timeout,
        coverage_timeout=args.coverage_timeout,
        force=args.force,
    )

    print(json.dumps({"gates": m["gates"], "manifest_validation_problems": m["manifest_validation_problems"]}, indent=2))

    gate_statuses = m["gates"]
    any_fail = any(g["status"] == "fail" for g in gate_statuses.values())
    any_problems = bool(m["manifest_validation_problems"])
    if any_fail or any_problems:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
