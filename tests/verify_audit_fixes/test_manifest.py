from pathlib import Path

from scripts.verify_audit_fixes.manifest import (
    GATES_7_TO_11_NOT_VERIFIABLE,
    build_manifest,
    validate_manifest,
    write_manifest,
)


def _git_identity(clean=True):
    return {
        "sha": "a" * 40,
        "branch": "master",
        "working_tree_clean": clean,
        "diff_hash": "b" * 64,
        "status": "clean" if clean else "dirty",
        "note": "note",
        "commands": [{"command": "git status", "stdout_file": "git_status.txt", "stderr_file": "git_status.txt"}],
    }


def _environment():
    return {
        "python_version": "3.14.6",
        "platform": "Windows",
        "dependency_lock_hash": "c" * 64,
        "commands": [{"command": "pip freeze", "stdout_file": "environment.txt", "stderr_file": "environment.txt"}],
    }


def _gate(status="pass"):
    return {"status": status, "commands": []}


def test_build_manifest_includes_all_six_gates_and_gates_7_11(tmp_path):
    m = build_manifest(
        repo_root=tmp_path,
        evidence_dir=tmp_path,
        audit_report_path="docs/reports/audit.md",
        target_scope="commit abc123",
        git_identity=_git_identity(),
        environment=_environment(),
        gate2=_gate(),
        gate3=_gate(),
        gate4=_gate(),
        gate5=_gate(),
        gate6=_gate(),
    )
    assert set(m["gates"].keys()) == {
        "gate_1_repository_identity",
        "gate_2_static_repository_checks",
        "gate_3_test_discovery",
        "gate_4_full_tests",
        "gate_5_smoke_tests",
        "gate_6_coverage",
    }
    assert m["gates_7_11"] == GATES_7_TO_11_NOT_VERIFIABLE
    # Every gate 7-11 entry must explicitly say why, never silently absent.
    for reason in m["gates_7_11"].values():
        assert reason.startswith("Not verifiable") or reason.startswith("Not run")
    assert m["git"]["sha"] == "a" * 40
    assert m["audit_report_path"] == "docs/reports/audit.md"


def test_validate_manifest_detects_missing_referenced_file(tmp_path):
    m = build_manifest(
        repo_root=tmp_path,
        evidence_dir=tmp_path,
        audit_report_path=None,
        target_scope=None,
        git_identity=_git_identity(),
        environment=_environment(),
        gate2=_gate(),
        gate3=_gate(),
        gate4=_gate(),
        gate5=_gate(),
        gate6=_gate(),
    )
    # git_status.txt and environment.txt were never actually written to tmp_path.
    problems = validate_manifest(m, tmp_path)
    assert any("git_status.txt" in p for p in problems)
    assert any("environment.txt" in p for p in problems)


def test_validate_manifest_passes_when_all_files_exist(tmp_path):
    (tmp_path / "git_status.txt").write_text("x", encoding="utf-8")
    (tmp_path / "environment.txt").write_text("x", encoding="utf-8")
    m = build_manifest(
        repo_root=tmp_path,
        evidence_dir=tmp_path,
        audit_report_path=None,
        target_scope=None,
        git_identity=_git_identity(),
        environment=_environment(),
        gate2=_gate(),
        gate3=_gate(),
        gate4=_gate(),
        gate5=_gate(),
        gate6=_gate(),
    )
    assert validate_manifest(m, tmp_path) == []


def test_write_manifest_produces_valid_json(tmp_path):
    m = build_manifest(
        repo_root=tmp_path,
        evidence_dir=tmp_path,
        audit_report_path=None,
        target_scope=None,
        git_identity=_git_identity(),
        environment=_environment(),
        gate2=_gate(),
        gate3=_gate(),
        gate4=_gate(),
        gate5=_gate(),
        gate6=_gate(),
    )
    path = write_manifest(m, tmp_path)
    assert path.name == "manifest.json"
    import json

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["git"]["branch"] == "master"
