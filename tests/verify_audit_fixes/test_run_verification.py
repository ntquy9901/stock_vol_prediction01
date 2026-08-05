import json
import subprocess
from pathlib import Path

import pytest

from scripts.verify_audit_fixes.run_verification import (
    _load_findings,
    build_arg_parser,
    main,
    run_verification,
)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


def _minimal_project(repo: Path) -> None:
    (repo / "src").mkdir()
    (repo / "src" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "src" / "module.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_module.py").write_text(
        "from src.module import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "minimal project", cwd=repo)


def test_run_verification_end_to_end_produces_full_evidence_dir(git_repo, tmp_path):
    _minimal_project(git_repo)
    evidence_dir = tmp_path / "evidence_out"

    manifest = run_verification(repo_root=git_repo, evidence_dir=evidence_dir)

    for fname in (
        "manifest.json",
        "git_status.txt",
        "git_diff_stat.txt",
        "environment.txt",
        "pytest_collection.txt",
        "pytest_full.txt",
        "pytest_smoke.txt",
        "coverage_summary.txt",
        "coverage.xml",
        "diff_cover.txt",
        "branch_coverage.json",
        "ruff.txt",
        "acceptance_traceability.csv",
    ):
        assert (evidence_dir / fname).exists(), f"missing evidence file: {fname}"

    assert manifest["manifest_validation_problems"] == []
    assert manifest["gates"]["gate_1_repository_identity"]["status"] == "clean"
    assert manifest["gates"]["gate_4_full_tests"]["status"] == "pass"
    assert "gate_10_adversarial_review" in manifest["gates_7_11"]


def test_run_verification_refuses_nonempty_evidence_dir_without_force(git_repo, tmp_path):
    _minimal_project(git_repo)
    evidence_dir = tmp_path / "evidence_out"
    evidence_dir.mkdir()
    (evidence_dir / "existing.txt").write_text("x", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_verification(repo_root=git_repo, evidence_dir=evidence_dir)


def test_run_verification_skip_flags_mark_gates_not_run(git_repo, tmp_path):
    _minimal_project(git_repo)
    evidence_dir = tmp_path / "evidence_out"

    manifest = run_verification(
        repo_root=git_repo,
        evidence_dir=evidence_dir,
        skip_gate4=True,
        skip_gate5=True,
        skip_gate6=True,
    )
    assert manifest["gates"]["gate_4_full_tests"]["status"] == "not_run"
    assert manifest["gates"]["gate_5_smoke_tests"]["status"] == "not_run"
    assert manifest["gates"]["gate_6_coverage"]["status"] == "not_run"


def test_run_verification_writes_traceability_csv_from_findings(git_repo, tmp_path):
    _minimal_project(git_repo)
    evidence_dir = tmp_path / "evidence_out"
    findings = [
        {
            "finding_id": "VER-999",
            "severity": "LOW",
            "requirement": "example",
            "status": "Not applicable",
            "notes": "docs-only change",
        }
    ]

    run_verification(
        repo_root=git_repo,
        evidence_dir=evidence_dir,
        findings=findings,
        skip_gate4=True,
        skip_gate5=True,
        skip_gate6=True,
    )
    csv_text = (evidence_dir / "acceptance_traceability.csv").read_text(encoding="utf-8")
    assert "VER-999" in csv_text


def test_load_findings_returns_none_when_not_given():
    assert _load_findings(None) is None


def test_load_findings_rejects_non_list_json(tmp_path):
    path = tmp_path / "findings.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_findings(str(path))


def test_load_findings_reads_list_json(tmp_path):
    path = tmp_path / "findings.json"
    path.write_text(json.dumps([{"finding_id": "X"}]), encoding="utf-8")
    result = _load_findings(str(path))
    assert result == [{"finding_id": "X"}]


def test_main_cli_returns_zero_on_clean_pass(git_repo, tmp_path, capsys):
    _minimal_project(git_repo)
    evidence_dir = tmp_path / "evidence_out"

    exit_code = main(
        [
            "--repo-root", str(git_repo),
            "--evidence-dir", str(evidence_dir),
            "--skip-gate4",
            "--skip-gate5",
            "--skip-gate6",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "gates" in captured.out


def test_main_cli_returns_nonzero_when_a_gate_fails(git_repo, tmp_path):
    (git_repo / "test_bad.py").write_text("def test_bad():\n    assert False\n", encoding="utf-8")
    _git("add", "-A", cwd=git_repo)
    _git("commit", "-q", "-m", "add failing test", cwd=git_repo)
    evidence_dir = tmp_path / "evidence_out"

    exit_code = main(
        [
            "--repo-root", str(git_repo),
            "--evidence-dir", str(evidence_dir),
            "--skip-gate6",
        ]
    )
    assert exit_code == 1


def test_build_arg_parser_requires_evidence_dir():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
