import subprocess
import sys
from pathlib import Path

from scripts.verify_audit_fixes import gates


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Gate 1
# ---------------------------------------------------------------------------

def test_capture_repository_identity_clean_tree(git_repo, evidence_dir):
    result = gates.capture_repository_identity(git_repo, evidence_dir)
    assert result["working_tree_clean"] is True
    assert result["status"] == "clean"
    assert len(result["sha"]) == 40
    assert result["branch"] == "master"
    assert len(result["diff_hash"]) == 64
    assert (evidence_dir / "git_status.txt").exists()
    assert (evidence_dir / "git_diff_stat.txt").exists()


def test_capture_repository_identity_dirty_tree(git_repo, evidence_dir):
    (git_repo / "README.md").write_text("changed\n", encoding="utf-8")
    result = gates.capture_repository_identity(git_repo, evidence_dir)
    assert result["working_tree_clean"] is False
    assert result["status"] == "dirty"
    assert "dirty" in result["note"].lower()
    status_text = (evidence_dir / "git_status.txt").read_text(encoding="utf-8")
    assert "README.md" in status_text


def test_capture_environment_writes_file_and_matches_running_interpreter(git_repo, evidence_dir):
    import platform

    result = gates.capture_environment(git_repo, evidence_dir)
    assert result["python_version"] == platform.python_version()
    assert (evidence_dir / "environment.txt").exists()
    assert len(result["dependency_lock_hash"]) == 64


# ---------------------------------------------------------------------------
# Gate 2
# ---------------------------------------------------------------------------

def test_gate2_passes_on_clean_repo(git_repo, evidence_dir):
    (git_repo / "clean.py").write_text("x = 1\n", encoding="utf-8")
    _git("add", "clean.py", cwd=git_repo)
    _git("commit", "-q", "-m", "add clean file", cwd=git_repo)

    result = gates.run_gate2_static_checks(git_repo, evidence_dir)
    assert result["git_diff_check"]["status"] == "pass"
    assert result["ruff"]["tool_available"] is True
    assert result["ruff"]["status"] == "pass"
    assert result["status"] == "pass"
    assert (evidence_dir / "ruff.txt").exists()
    assert (evidence_dir / "static_scans.txt").exists()


def test_gate2_fails_on_git_diff_check_whitespace_error(git_repo, evidence_dir):
    (git_repo / "clean.py").write_text("x = 1\n", encoding="utf-8")
    _git("add", "clean.py", cwd=git_repo)
    _git("commit", "-q", "-m", "add clean file", cwd=git_repo)
    (git_repo / "clean.py").write_text("x = 1   \n", encoding="utf-8")  # trailing whitespace

    result = gates.run_gate2_static_checks(git_repo, evidence_dir)
    assert result["git_diff_check"]["status"] == "fail"
    assert result["status"] == "fail"


def test_gate2_fails_on_ruff_violation(git_repo, evidence_dir):
    (git_repo / "bad.py").write_text("import os\n\nx = 1\n", encoding="utf-8")  # unused import
    _git("add", "bad.py", cwd=git_repo)
    _git("commit", "-q", "-m", "add file with unused import", cwd=git_repo)

    result = gates.run_gate2_static_checks(git_repo, evidence_dir)
    assert result["ruff"]["status"] == "fail"
    assert result["status"] == "fail"
    ruff_text = (evidence_dir / "ruff.txt").read_text(encoding="utf-8")
    assert "bad.py" in ruff_text


def test_gate2_scan_evidence_reflects_mechanical_scans(git_repo, evidence_dir):
    (git_repo / "hardcoded.py").write_text('p = "C:\\\\Users\\\\me\\\\data"\n', encoding="utf-8")
    _git("add", "hardcoded.py", cwd=git_repo)
    _git("commit", "-q", "-m", "add hardcoded path", cwd=git_repo)

    result = gates.run_gate2_static_checks(git_repo, evidence_dir)
    assert result["scans"]["hardcoded_paths_count"] == 1


# ---------------------------------------------------------------------------
# Gate 3
# ---------------------------------------------------------------------------

def _make_discoverable_project(repo: Path) -> None:
    (repo / "src").mkdir()
    (repo / "src" / "dummy.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "baselines").mkdir()
    (repo / "baselines" / "dummy.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_sample.py").write_text(
        "def test_one():\n    assert True\n\ndef test_two():\n    assert True\n",
        encoding="utf-8",
    )
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "add discoverable project", cwd=repo)


def test_gate3_collects_tests_from_all_testdirs(git_repo, evidence_dir):
    _make_discoverable_project(git_repo)
    result = gates.run_gate3_test_discovery(git_repo, evidence_dir)
    assert result["status"] == "pass"
    assert result["collected_total"] == 2
    assert result["per_directory"]["tests"]["collected"] == 2
    assert result["per_directory"]["src"]["exists"] is True
    assert (evidence_dir / "pytest_collection.txt").exists()


def test_gate3_reports_nonexistent_directory(git_repo, evidence_dir):
    _make_discoverable_project(git_repo)
    result = gates.run_gate3_test_discovery(git_repo, evidence_dir, testdirs=("tests", "nonexistent_dir"))
    assert result["per_directory"]["nonexistent_dir"] == {"exists": False}


# ---------------------------------------------------------------------------
# Gate 4
# ---------------------------------------------------------------------------

def test_gate4_all_pass(git_repo, evidence_dir):
    (git_repo / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    _git("add", "-A", cwd=git_repo)
    _git("commit", "-q", "-m", "add passing test", cwd=git_repo)

    result = gates.run_gate4_full_tests(git_repo, evidence_dir)
    assert result["status"] == "pass"
    assert result["exit_code"] == 0
    assert result["counts"]["passed"] == 1
    assert result["counts"]["failed"] == 0
    assert (evidence_dir / "pytest_full.txt").exists()
    assert (evidence_dir / "junit_full.xml").exists()


def test_gate4_reports_failure(git_repo, evidence_dir):
    (git_repo / "test_bad.py").write_text("def test_bad():\n    assert False\n", encoding="utf-8")
    _git("add", "-A", cwd=git_repo)
    _git("commit", "-q", "-m", "add failing test", cwd=git_repo)

    result = gates.run_gate4_full_tests(git_repo, evidence_dir)
    assert result["status"] == "fail"
    assert result["exit_code"] != 0
    assert result["counts"]["failed"] == 1


# ---------------------------------------------------------------------------
# Gate 5
# ---------------------------------------------------------------------------

def test_gate5_smoke_pass(git_repo, evidence_dir):
    (git_repo / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    smoke: smoke test\n", encoding="utf-8"
    )
    (git_repo / "test_smoke.py").write_text(
        "import pytest\n\n@pytest.mark.smoke\ndef test_smoke_ok():\n    assert True\n",
        encoding="utf-8",
    )
    _git("add", "-A", cwd=git_repo)
    _git("commit", "-q", "-m", "add smoke test", cwd=git_repo)

    result = gates.run_gate5_smoke_tests(git_repo, evidence_dir)
    assert result["status"] == "pass"
    assert result["zero_selected"] is False
    assert result["counts"]["passed"] == 1


def test_gate5_fails_when_zero_smoke_tests_selected(git_repo, evidence_dir):
    (git_repo / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    smoke: smoke test\n", encoding="utf-8"
    )
    (git_repo / "test_nosmoke.py").write_text("def test_plain():\n    assert True\n", encoding="utf-8")
    _git("add", "-A", cwd=git_repo)
    _git("commit", "-q", "-m", "add non-smoke test", cwd=git_repo)

    result = gates.run_gate5_smoke_tests(git_repo, evidence_dir)
    assert result["status"] == "fail"
    assert result["zero_selected"] is True


# ---------------------------------------------------------------------------
# Gate 6
# ---------------------------------------------------------------------------

def _make_coverage_project(repo: Path) -> None:
    (repo / "src").mkdir()
    (repo / "src" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "src" / "module.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_module.py").write_text(
        "from src.module import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "add coverage project", cwd=repo)


def test_gate6_reports_not_run_when_tooling_missing(git_repo, evidence_dir, monkeypatch):
    monkeypatch.setattr(gates, "_tool_available", lambda module_name=None, executable=None: False)
    result = gates.run_gate6_coverage(git_repo, evidence_dir)
    assert result["status"] == "not_run"
    assert "not installed" in result["reason"]
    assert (evidence_dir / "branch_coverage.json").exists()


def test_gate6_passes_on_fully_covered_diff(git_repo, evidence_dir):
    _make_coverage_project(git_repo)
    result = gates.run_gate6_coverage(git_repo, evidence_dir, cov_targets=("src",))
    assert result["status"] == "pass"
    assert result["c0_line_diff_coverage"]["exit_code"] == 0
    assert (evidence_dir / "coverage.xml").exists()
    assert (evidence_dir / "diff_cover.txt").exists()
    assert (evidence_dir / "branch_coverage.json").exists()


def test_gate6_fails_on_uncovered_diff_lines(git_repo, evidence_dir):
    _make_coverage_project(git_repo)
    # Uncommitted addition with no covering test -> diff-cover must see it and fail C0.
    with (git_repo / "src" / "module.py").open("a", encoding="utf-8") as fh:
        fh.write("\ndef sub(a, b):\n    return a - b\n")

    result = gates.run_gate6_coverage(git_repo, evidence_dir, cov_targets=("src",))
    assert result["status"] == "fail"
    assert result["c0_line_diff_coverage"]["exit_code"] != 0
    diff_cover_text = (evidence_dir / "diff_cover.txt").read_text(encoding="utf-8")
    assert "module.py" in diff_cover_text
