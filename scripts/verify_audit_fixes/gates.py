"""Gates 1-6: repository identity, static checks, test discovery, full tests,
smoke tests, and coverage.

Every gate function:
  - actually runs the underlying command(s) (or records why it could not),
  - writes raw output to the evidence directory,
  - returns a small JSON-serializable dict for the manifest.

No gate here fabricates a pass. If required tooling is missing, the gate
reports status "not_run" with an explicit reason instead of a fake result.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from .commands import append_command_output, run_command, write_command_output
from .static_scans import format_scan_report, run_all_scans


def _tool_available(module_name: str | None = None, executable: str | None = None) -> bool:
    if module_name is not None and importlib.util.find_spec(module_name) is None:
        return False
    if executable is not None and shutil.which(executable) is None:
        return False
    return True


# ---------------------------------------------------------------------------
# Gate 1 — repository identity
# ---------------------------------------------------------------------------

def capture_repository_identity(repo_root: Path, evidence_dir: Path) -> dict:
    sha_res = run_command(["git", "rev-parse", "HEAD"], repo_root)
    branch_res = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    status_res = run_command(["git", "status", "--porcelain"], repo_root)
    diff_stat_res = run_command(["git", "diff", "--stat"], repo_root)
    diff_res = run_command(["git", "diff"], repo_root)

    write_command_output(evidence_dir, "git_status.txt", sha_res)
    append_command_output(evidence_dir, "git_status.txt", branch_res)
    append_command_output(evidence_dir, "git_status.txt", status_res)
    write_command_output(evidence_dir, "git_diff_stat.txt", diff_stat_res)
    append_command_output(evidence_dir, "git_diff_stat.txt", diff_res)

    working_tree_clean = status_res.stdout.strip() == ""
    diff_hash = hashlib.sha256(diff_res.stdout.encode("utf-8")).hexdigest()

    return {
        "sha": sha_res.stdout.strip(),
        "branch": branch_res.stdout.strip(),
        "working_tree_clean": working_tree_clean,
        "diff_hash": diff_hash,
        "status": "clean" if working_tree_clean else "dirty",
        "note": (
            "Working tree is clean."
            if working_tree_clean
            else "Working tree is dirty. Per Gate 1, this run's scope must explicitly "
            "cover every uncommitted modification listed in git_status.txt — a dirty "
            "tree is not automatically rejected by this runner; scope acknowledgement "
            "is the caller's (skill/orchestrator's) responsibility."
        ),
        "commands": [
            sha_res.to_manifest_entry("git_status.txt"),
            branch_res.to_manifest_entry("git_status.txt"),
            status_res.to_manifest_entry("git_status.txt"),
            diff_stat_res.to_manifest_entry("git_diff_stat.txt"),
            diff_res.to_manifest_entry("git_diff_stat.txt"),
        ],
    }


def capture_environment(repo_root: Path, evidence_dir: Path) -> dict:
    pip_freeze_res = run_command([sys.executable, "-m", "pip", "freeze"], repo_root)
    lock_hash = hashlib.sha256(pip_freeze_res.stdout.encode("utf-8")).hexdigest()

    env_text = (
        f"python_version: {platform.python_version()}\n"
        f"python_executable: {sys.executable}\n"
        f"platform: {platform.platform()}\n"
        f"dependency_lock_hash (sha256 of pip freeze): {lock_hash}\n"
        f"\n--- pip freeze ---\n{pip_freeze_res.stdout}\n"
    )
    (evidence_dir / "environment.txt").write_text(env_text, encoding="utf-8")

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_lock_hash": lock_hash,
        "commands": [pip_freeze_res.to_manifest_entry("environment.txt")],
    }


# ---------------------------------------------------------------------------
# Gate 2 — static repository checks
# ---------------------------------------------------------------------------

def run_gate2_static_checks(repo_root: Path, evidence_dir: Path) -> dict:
    commands = []

    diff_check_res = run_command(["git", "diff", "--check"], repo_root)
    write_command_output(evidence_dir, "static_checks.txt", diff_check_res)
    commands.append(diff_check_res.to_manifest_entry("static_checks.txt"))

    ruff_available = _tool_available(executable="ruff")
    if ruff_available:
        ruff_res = run_command(
            [
                "ruff", "check", ".",
                "--exclude", ".agents",
                "--exclude", ".claude",
                "--exclude", "_bmad",
                "--exclude", "archive",
                "--exclude", "data",
            ],
            repo_root,
        )
        write_command_output(evidence_dir, "ruff.txt", ruff_res)
        commands.append(ruff_res.to_manifest_entry("ruff.txt"))
        ruff_status = "pass" if ruff_res.exit_code == 0 else "fail"
        ruff_exit_code = ruff_res.exit_code
    else:
        (evidence_dir / "ruff.txt").write_text(
            "Not run — `ruff` executable not found on PATH.\n", encoding="utf-8"
        )
        ruff_status = "not_run"
        ruff_exit_code = None

    scan_result = run_all_scans(repo_root)
    (evidence_dir / "static_scans.txt").write_text(format_scan_report(scan_result), encoding="utf-8")

    diff_check_status = "pass" if diff_check_res.exit_code == 0 else "fail"
    overall_status = "pass" if diff_check_status == "pass" and ruff_status in ("pass", "not_run") else "fail"

    return {
        "status": overall_status,
        "git_diff_check": {"status": diff_check_status, "exit_code": diff_check_res.exit_code},
        "ruff": {"status": ruff_status, "exit_code": ruff_exit_code, "tool_available": ruff_available},
        "scans": {
            "files_scanned": scan_result["files_scanned"],
            "hardcoded_paths_count": len(scan_result["hardcoded_paths"]),
            "bare_except_count": len(scan_result["bare_except"]),
            "random_split_count": len(scan_result["random_split"]),
            "duplicate_module_names_count": len(scan_result["duplicate_module_names"]),
            "evidence_file": "static_scans.txt",
        },
        "commands": commands,
    }


# ---------------------------------------------------------------------------
# Gate 3 — test discovery
# ---------------------------------------------------------------------------

_COLLECTED_RE = re.compile(r"(\d+)\s+tests?\s+collected")
_COLLECT_ERROR_RE = re.compile(r"(\d+)\s+errors?\s+(?:during collection|in\s)")
_NO_TESTS_RE = re.compile(r"no tests ran|no tests collected", re.IGNORECASE)


def _parse_collected_count(stdout: str) -> dict:
    m = _COLLECTED_RE.search(stdout)
    collected = int(m.group(1)) if m else 0
    err_m = _COLLECT_ERROR_RE.search(stdout)
    errors = int(err_m.group(1)) if err_m else 0
    return {"collected": collected, "collection_errors": errors}


def run_gate3_test_discovery(repo_root: Path, evidence_dir: Path, testdirs: tuple[str, ...] = ("src", "baselines", "tests")) -> dict:
    overall_res = run_command([sys.executable, "-m", "pytest", "--collect-only", "-q"], repo_root, timeout=600)
    write_command_output(evidence_dir, "pytest_collection.txt", overall_res)

    per_directory = {}
    commands = [overall_res.to_manifest_entry("pytest_collection.txt")]
    for d in testdirs:
        if not (repo_root / d).exists():
            per_directory[d] = {"exists": False}
            continue
        res = run_command([sys.executable, "-m", "pytest", "--collect-only", "-q", d], repo_root, timeout=300)
        append_command_output(evidence_dir, "pytest_collection.txt", res)
        commands.append(res.to_manifest_entry("pytest_collection.txt"))
        parsed = _parse_collected_count(res.stdout)
        per_directory[d] = {"exists": True, "exit_code": res.exit_code, **parsed}

    overall_parsed = _parse_collected_count(overall_res.stdout)
    status = "pass" if overall_res.exit_code == 0 else "fail"

    return {
        "status": status,
        "exit_code": overall_res.exit_code,
        "collected_total": overall_parsed["collected"],
        "collection_errors": overall_parsed["collection_errors"],
        "per_directory": per_directory,
        "commands": commands,
    }


# ---------------------------------------------------------------------------
# Gate 4 — full tests
# ---------------------------------------------------------------------------

_SUMMARY_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|error(?:s)?|skipped|xfailed|xpassed|warnings?)"
)


def _parse_pytest_summary(stdout: str) -> dict:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    tail = stdout[-2000:]
    for n, label in _SUMMARY_COUNT_RE.findall(tail):
        n = int(n)
        label = label.rstrip("s") if label not in ("passed", "failed", "skipped") else label
        if label.startswith("error"):
            counts["errors"] += n
        elif label in counts:
            counts[label] += n
    return counts


def run_gate4_full_tests(repo_root: Path, evidence_dir: Path, timeout: int = 3600) -> dict:
    junit_path = evidence_dir / "junit_full.xml"
    res = run_command(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit_path}"],
        repo_root,
        timeout=timeout,
    )
    write_command_output(evidence_dir, "pytest_full.txt", res)
    counts = _parse_pytest_summary(res.stdout)
    status = "pass" if res.exit_code == 0 else "fail"

    return {
        "status": status,
        "exit_code": res.exit_code,
        "counts": counts,
        "junit_xml": "junit_full.xml" if junit_path.exists() else None,
        "commands": [res.to_manifest_entry("pytest_full.txt")],
    }


# ---------------------------------------------------------------------------
# Gate 5 — smoke tests
# ---------------------------------------------------------------------------

def run_gate5_smoke_tests(repo_root: Path, evidence_dir: Path, timeout: int = 1800) -> dict:
    junit_path = evidence_dir / "junit_smoke.xml"
    res = run_command(
        [sys.executable, "-m", "pytest", "-m", "smoke", "-q", f"--junitxml={junit_path}"],
        repo_root,
        timeout=timeout,
    )
    write_command_output(evidence_dir, "pytest_smoke.txt", res)
    counts = _parse_pytest_summary(res.stdout)
    zero_selected = res.exit_code == 5 or bool(_NO_TESTS_RE.search(res.stdout))

    if zero_selected:
        status = "fail"
        note = "Gate 5 fails: zero smoke tests were selected (marker 'smoke' matched nothing)."
    elif res.exit_code == 0:
        status = "pass"
        note = None
    else:
        status = "fail"
        note = "Smoke tests were selected but did not all pass."

    return {
        "status": status,
        "exit_code": res.exit_code,
        "zero_selected": zero_selected,
        "counts": counts,
        "note": note,
        "junit_xml": "junit_smoke.xml" if junit_path.exists() else None,
        "commands": [res.to_manifest_entry("pytest_smoke.txt")],
    }


# ---------------------------------------------------------------------------
# Gate 6 — coverage
# ---------------------------------------------------------------------------

_DIFF_COVER_PCT_RE = re.compile(r"Coverage[^0-9]*([0-9]+(?:\.[0-9]+)?)%")

# diff-cover's own default ("origin/main") does not exist in this repo (remote
# tracking branch is "origin/master") and does not exist at all in a fresh
# fixture repo with no remote. Resolve to the first ref that actually exists;
# "HEAD" always resolves and degrades gracefully to "only uncommitted changes".
_COMPARE_BRANCH_CANDIDATES = ("origin/master", "origin/main", "master", "main", "HEAD")


def _resolve_compare_branch(repo_root: Path, requested: str | None) -> str:
    candidates = ([requested] if requested else []) + list(_COMPARE_BRANCH_CANDIDATES)
    for ref in candidates:
        res = run_command(["git", "rev-parse", "--verify", "--quiet", ref], repo_root, timeout=30)
        if res.exit_code == 0:
            return ref
    return "HEAD"


def run_gate6_coverage(
    repo_root: Path,
    evidence_dir: Path,
    cov_targets: tuple[str, ...] = ("src",),
    timeout: int = 3600,
    compare_branch: str | None = None,
) -> dict:
    cov_cov_available = _tool_available(module_name="pytest_cov")
    diff_cover_available = _tool_available(executable="diff-cover")

    if not cov_cov_available or not diff_cover_available:
        missing = []
        if not cov_cov_available:
            missing.append("pytest-cov (python module 'pytest_cov')")
        if not diff_cover_available:
            missing.append("diff-cover (executable 'diff-cover')")
        reason = "Not run — tooling not installed: " + ", ".join(missing)
        for fname in ("coverage_summary.txt", "coverage.xml", "diff_cover.txt"):
            (evidence_dir / fname).write_text(reason + "\n", encoding="utf-8")

        (evidence_dir / "branch_coverage.json").write_text(
            json.dumps({"status": "not_run", "reason": reason}, indent=2) + "\n", encoding="utf-8"
        )
        return {"status": "not_run", "reason": reason, "commands": []}

    coverage_xml_path = evidence_dir / "coverage.xml"
    cov_args = []
    for t in cov_targets:
        cov_args += ["--cov", t]
    pytest_cov_res = run_command(
        [
            sys.executable, "-m", "pytest", "-q",
            *cov_args,
            "--cov-branch",
            f"--cov-report=xml:{coverage_xml_path}",
            "--cov-report=term-missing",
        ],
        repo_root,
        timeout=timeout,
    )
    write_command_output(evidence_dir, "coverage_summary.txt", pytest_cov_res)

    commands = [pytest_cov_res.to_manifest_entry("coverage_summary.txt")]

    if not coverage_xml_path.exists():
        (evidence_dir / "diff_cover.txt").write_text(
            "Not run — coverage.xml was not produced by the pytest-cov invocation "
            "(see coverage_summary.txt for the underlying test run's exit code/output).\n",
            encoding="utf-8",
        )
        (evidence_dir / "branch_coverage.json").write_text(
            '{"status": "not_run", "reason": "coverage.xml missing"}\n', encoding="utf-8"
        )
        return {
            "status": "not_run",
            "reason": "coverage.xml was not produced (test run likely failed before coverage was written).",
            "pytest_exit_code": pytest_cov_res.exit_code,
            "commands": commands,
        }

    resolved_compare_branch = _resolve_compare_branch(repo_root, compare_branch)

    c0_res = run_command(
        [
            "diff-cover", str(coverage_xml_path),
            "--fail-under=100",
            f"--compare-branch={resolved_compare_branch}",
        ],
        repo_root,
        timeout=300,
    )
    write_command_output(evidence_dir, "diff_cover.txt", c0_res)
    commands.append(c0_res.to_manifest_entry("diff_cover.txt"))

    c1_res = run_command(
        [
            "diff-cover", str(coverage_xml_path),
            "--branch-coverage", "--fail-under=80",
            f"--compare-branch={resolved_compare_branch}",
        ],
        repo_root,
        timeout=300,
    )
    append_command_output(evidence_dir, "diff_cover.txt", c1_res)
    commands.append(c1_res.to_manifest_entry("diff_cover.txt"))

    c0_pct_m = _DIFF_COVER_PCT_RE.search(c0_res.stdout)
    c1_pct_m = _DIFF_COVER_PCT_RE.search(c1_res.stdout)

    overall_rates = _parse_coverage_xml_rates(coverage_xml_path)

    branch_summary = {
        "status": "pass" if (c0_res.exit_code == 0 and c1_res.exit_code == 0) else "fail",
        "compare_branch": resolved_compare_branch,
        "c0_line_diff_coverage": {
            "fail_under": 100,
            "exit_code": c0_res.exit_code,
            "reported_percent": float(c0_pct_m.group(1)) if c0_pct_m else None,
        },
        "c1_branch_diff_coverage": {
            "fail_under": 80,
            "exit_code": c1_res.exit_code,
            "reported_percent": float(c1_pct_m.group(1)) if c1_pct_m else None,
        },
        "repo_wide_coverage_xml_rates": overall_rates,
        "note": (
            "c0/c1 percentages above are diff-cover's DIFF-scoped result (changed lines only, "
            "per CLAUDE.md's C0=100%/C1>=80% gate). repo_wide_coverage_xml_rates is the whole-repo "
            "rate parsed directly from coverage.xml and is NOT the diff-scoped gate; it is retained "
            "for context only."
        ),
    }
    (evidence_dir / "branch_coverage.json").write_text(json.dumps(branch_summary, indent=2), encoding="utf-8")

    return {**branch_summary, "commands": commands}


def _parse_coverage_xml_rates(coverage_xml_path: Path) -> dict:
    try:
        root = ET.parse(coverage_xml_path).getroot()
        return {
            "line_rate": float(root.get("line-rate", "nan")),
            "branch_rate": float(root.get("branch-rate", "nan")),
        }
    except (ET.ParseError, OSError, ValueError):
        return {"line_rate": None, "branch_rate": None}
