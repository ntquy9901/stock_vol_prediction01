"""Build manifest.json for one evidence-capture run.

The schema follows docs/reports/2026-08-02_152758_summaryOfUpdate_report.md
("Required manifest schema"), extended with a top-level ``gates`` section for
Gates 1-6 (this runner) and a ``gates_7_11`` section that records, per gate,
why it is not verifiable by this runner rather than silently omitting it.

Every file path recorded in ``commands[].stdout_file`` must exist under the
evidence directory by the time this manifest is written — that is validated
by ``validate_manifest``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# Gates 7-11 require infrastructure this project does not yet have. Listed
# explicitly (not silently skipped) per the spec's own non-goal: "Do not mark
# a finding fixed based only on code inspection when executable verification
# is required" — applied here to the verification tooling itself.
GATES_7_TO_11_NOT_VERIFIABLE = {
    "gate_7_regression_evidence": (
        "Not verifiable — no per-finding regression test suite exists yet "
        "(leakage/date-alignment/DirAcc-panel/temporal-split/import-isolation/"
        "JSON NaN-Infinity tests). Requires writing those tests against named "
        "audit findings first; this runner does not author tests."
    ),
    "gate_8_ml_result_provenance": (
        "Not verifiable — no canonical run-record schema exists that binds a "
        "metrics result to git SHA + data/ticker-manifest hash + split dates + "
        "seed + config hash + checkpoint hash + metric-schema version. "
        "Requires an ML experiment-tracking component this project does not have."
    ),
    "gate_9_statistical_verification": (
        "Not verifiable — no multi-seed (3-5 seed) run framework, paired/"
        "block-bootstrap comparison, or documented model-selection/multiple-"
        "comparison policy exists yet for this project's models."
    ),
    "gate_10_adversarial_review": (
        "Not run by this runner — requires invoking the project's adversarial "
        "code-review skill (Blind Hunter + Edge Case Hunter + Acceptance "
        "Auditor, e.g. bmad-code-review) against the exact target diff and "
        "retaining its structured disposition. This is an orchestration step "
        "for the verify-audit-fixes SKILL.md, not a mechanical command."
    ),
    "gate_11_clean_reproduction": (
        "Not verifiable — no isolated/clean-checkout reproduction environment "
        "(e.g. fresh venv + fresh clone) is wired up for this project, and no "
        "canonical comparison table with numerical tolerances is defined yet."
    ),
}


def build_manifest(
    *,
    repo_root: Path,
    evidence_dir: Path,
    audit_report_path: str | None,
    target_scope: str | None,
    git_identity: dict,
    environment: dict,
    gate2: dict,
    gate3: dict,
    gate4: dict,
    gate5: dict,
    gate6: dict,
) -> dict:
    all_commands = []
    for g in (git_identity, environment, gate2, gate3, gate4, gate5, gate6):
        all_commands.extend(g.get("commands", []))

    gates = {
        "gate_1_repository_identity": {
            "status": git_identity["status"],
            "note": git_identity["note"],
        },
        "gate_2_static_repository_checks": {"status": gate2["status"]},
        "gate_3_test_discovery": {"status": gate3["status"]},
        "gate_4_full_tests": {"status": gate4["status"]},
        "gate_5_smoke_tests": {"status": gate5["status"]},
        "gate_6_coverage": {"status": gate6["status"]},
    }

    manifest = {
        "verification_timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "audit_report_path": audit_report_path,
        "target_scope": target_scope,
        "git": {
            "sha": git_identity["sha"],
            "branch": git_identity["branch"],
            "working_tree_clean": git_identity["working_tree_clean"],
            "diff_hash": git_identity["diff_hash"],
        },
        "environment": {
            "python_version": environment["python_version"],
            "platform": environment["platform"],
            "dependency_lock_hash": environment["dependency_lock_hash"],
        },
        "data": None,
        "experiment": None,
        "gates": gates,
        "gate_details": {
            "gate_2_static_repository_checks": gate2,
            "gate_3_test_discovery": gate3,
            "gate_4_full_tests": gate4,
            "gate_5_smoke_tests": gate5,
            "gate_6_coverage": gate6,
        },
        "gates_7_11": GATES_7_TO_11_NOT_VERIFIABLE,
        "commands": all_commands,
    }
    return manifest


def validate_manifest(manifest: dict, evidence_dir: Path) -> list[str]:
    """Return a list of problems: missing stdout/stderr files referenced by manifest['commands'].

    An empty list means every referenced evidence file exists.
    """
    problems = []
    for entry in manifest.get("commands", []):
        for key in ("stdout_file", "stderr_file"):
            fname = entry.get(key)
            if fname and not (evidence_dir / fname).exists():
                problems.append(f"manifest references missing file: {fname} (command: {entry.get('command')})")
    return problems


def write_manifest(manifest: dict, evidence_dir: Path) -> Path:
    path = evidence_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
