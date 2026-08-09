"""Post-code quality gate orchestrator.

Runs the project's quality tools after code changes and prints a PASS/FAIL
summary, then writes a timestamped markdown report.

Checks (each returns ``(name, status, detail)`` and skips gracefully when its
tool/input is missing):

- LINT   : ``ruff check .`` (SKIPPED if ruff not installed).
- TESTS  : ``pytest -q`` (+ ``pytest -q -m smoke``; unregistered marker or
           no-tests handled as SKIPPED, not a hard fail).
- SCHEMA : pandera ``validate_data()`` over data/processed + news panel.
- DRIFT  : evidently DataDriftPreset on a temporal split of one processed
           ticker (informational, NEVER fails the gate; ``--fast`` skips it).

Exit code is non-zero only if a HARD check (LINT real failure / TESTS failure /
SCHEMA failure) fails. DRIFT and SKIPPED never cause a non-zero exit.

Run: ``python scripts/quality_gate/run_quality_gate.py [--fast]``
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Data locations (kept dependency-free so the orchestrator imports even when
# pandera/pandas are absent; validate_data is imported lazily inside check_schema).
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NEWS_PANEL = PROJECT_ROOT / "data" / "features" / "dual_group_news_panel.parquet"

# Status constants.
PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"
INFO = "INFO"

# Statuses that make the gate exit non-zero.
HARD_FAIL_CHECKS = {"LINT", "TESTS", "SCHEMA"}

LINT_EXCLUDES = [".agents", ".claude", "_bmad", "archive", "data"]


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def _run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def check_lint() -> CheckResult:
    if shutil.which("ruff") is None:
        return CheckResult("LINT", SKIPPED, "ruff not installed")
    cmd = ["ruff", "check", "."]
    for exc in LINT_EXCLUDES:
        # --extend-exclude ADDS to ruff's built-in defaults (.venv, build, etc.);
        # plain --exclude would REPLACE them and re-lint vendored dirs.
        cmd += ["--extend-exclude", exc]
    try:
        proc = _run(cmd)
    except Exception as exc:  # noqa: BLE001 - a crash/timeout must not hide a HARD check
        return CheckResult("LINT", FAIL, f"ruff invocation failed: {exc}")
    if proc.returncode == 0:
        return CheckResult("LINT", PASS, "no lint errors")
    lines = [ln.strip() for ln in (proc.stdout or proc.stderr).splitlines() if ln.strip()]
    found = [ln for ln in lines if ln.startswith("Found") and "error" in ln]
    detail = found[-1] if found else (lines[-1] if lines else f"exit {proc.returncode}")
    return CheckResult("LINT", FAIL, detail)


def check_tests() -> CheckResult:
    try:
        proc = _run([sys.executable, "-m", "pytest", "-q"])
    except Exception as exc:  # noqa: BLE001
        return CheckResult("TESTS", FAIL, f"pytest invocation failed: {exc}")

    # pytest not installed -> SKIPPED (tool missing, not a real test failure).
    if "No module named pytest" in ((proc.stdout or "") + (proc.stderr or "")):
        return CheckResult("TESTS", SKIPPED, "pytest not installed")

    # exit 5 = no tests collected -> treat as SKIPPED (nothing to run).
    if proc.returncode == 5:
        unit = CheckResult("TESTS", SKIPPED, "no tests collected")
        return unit

    summary = _pytest_summary(proc.stdout)
    if proc.returncode == 0:
        smoke = _smoke_note()
        return CheckResult("TESTS", PASS, f"{summary}{smoke}")
    return CheckResult("TESTS", FAIL, summary or f"exit {proc.returncode}")


def _smoke_note() -> str:
    """Try the smoke marker; report as note only (never a hard fail here)."""
    try:
        proc = _run([sys.executable, "-m", "pytest", "-q", "-m", "smoke"])
    except Exception:  # noqa: BLE001
        return " | smoke: skipped (invocation failed)"
    if proc.returncode == 5:
        return " | smoke: no smoke tests collected"
    combined = proc.stdout + proc.stderr
    if "ERROR" in combined and "not found in `markers`" in combined:
        return " | smoke: marker unregistered (skipped)"
    if proc.returncode == 0:
        return f" | smoke: {_pytest_summary(proc.stdout)}"
    return f" | smoke: {_pytest_summary(proc.stdout)}"


def _pytest_summary(stdout: str) -> str:
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "passed" in ln or "failed" in ln or "error" in ln:
            return ln.strip("= ")
    return lines[-1] if lines else ""


def check_schema() -> CheckResult:
    # Lazy import so a missing pandera/pandas skips SCHEMA instead of crashing
    # the whole orchestrator at import time.
    try:
        from scripts.quality_gate.data_schemas import (
            INVALID,
            MISSING,
            validate_data,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("SCHEMA", SKIPPED, f"pandera/pandas unavailable: {exc}")
    try:
        results = validate_data()
    except Exception as exc:  # noqa: BLE001
        return CheckResult("SCHEMA", FAIL, f"validate_data crashed: {exc}")

    failures = [(n, d) for n, status, d in results if status == INVALID]
    present = [r for r in results if r[1] != MISSING]
    if failures:
        head = "; ".join(f"{n}: {d}" for n, d in failures[:3])
        more = f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""
        return CheckResult(
            "SCHEMA", FAIL, f"{len(failures)}/{len(present)} failed -> {head}{more}"
        )
    if not present:
        skips = [d for _, status, d in results if status == MISSING]
        return CheckResult("SCHEMA", SKIPPED, skips[0] if skips else "no data inputs")
    return CheckResult("SCHEMA", PASS, f"{len(present)}/{len(present)} artifacts valid")


def check_drift(out_dir: Path, max_rows: int = 4000) -> CheckResult:
    """Temporal train-vs-test data-drift report (informational)."""
    try:
        import pandas as pd
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset
    except Exception as exc:  # noqa: BLE001
        return CheckResult("DRIFT", SKIPPED, f"evidently import failed: {exc}")

    # Pick one representative processed ticker with a usable OHLCV/vol series.
    csvs = sorted(p for p in PROCESSED_DIR.glob("*.csv") if p.name != "processing_summary.csv")
    if not csvs:
        return CheckResult("DRIFT", SKIPPED, "no processed ticker files")

    try:
        df = pd.read_csv(csvs[0])
        df = df.select_dtypes(include=["number"]).dropna()
        if len(df) > max_rows:
            df = df.iloc[-max_rows:]
        if len(df) < 40:
            return CheckResult("DRIFT", SKIPPED, f"too few rows in {csvs[0].name}")
        split = int(len(df) * 0.7)
        ref = df.iloc[:split].reset_index(drop=True)
        cur = df.iloc[split:].reset_index(drop=True)

        data_def = DataDefinition()
        report = Report([DataDriftPreset()])
        snapshot = report.run(
            Dataset.from_pandas(cur, data_definition=data_def),
            Dataset.from_pandas(ref, data_definition=data_def),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / "drift.html"
        snapshot.save_html(str(html_path))
    except Exception as exc:  # noqa: BLE001
        return CheckResult("DRIFT", SKIPPED, f"drift report failed: {exc}")

    return CheckResult(
        "DRIFT",
        INFO,
        f"{csvs[0].name}: ref={len(ref)}/cur={len(cur)} rows -> {html_path}",
    )


def run_gate(fast: bool = False, timestamp: str | None = None) -> list[CheckResult]:
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results = [check_lint(), check_tests(), check_schema()]
    if fast:
        results.append(CheckResult("DRIFT", SKIPPED, "--fast (drift skipped)"))
    else:
        out_dir = PROJECT_ROOT / "results" / "quality_gate" / timestamp
        results.append(check_drift(out_dir))
    return results


def gate_failed(results: list[CheckResult]) -> bool:
    return any(
        r.name in HARD_FAIL_CHECKS and r.status == FAIL for r in results
    )


def print_summary(results: list[CheckResult]) -> None:
    name_w = max(len(r.name) for r in results)
    stat_w = max(len(r.status) for r in results)
    print("\n" + "=" * 72)
    print("QUALITY GATE SUMMARY")
    print("=" * 72)
    print(f"{'CHECK'.ljust(name_w)}  {'STATUS'.ljust(stat_w)}  DETAIL")
    print("-" * 72)
    for r in results:
        print(f"{r.name.ljust(name_w)}  {r.status.ljust(stat_w)}  {r.detail}")
    print("=" * 72)
    overall = "FAIL" if gate_failed(results) else "PASS"
    print(f"OVERALL: {overall}")
    print("=" * 72)


def write_report(results: list[CheckResult], timestamp: str) -> Path:
    reports_dir = PROJECT_ROOT / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Reuse the run timestamp (seconds precision) so the filename matches the
    # body/drift dir and two runs in the same minute do not overwrite.
    path = reports_dir / f"{timestamp}_quality_gate_report.md"
    overall = "FAIL" if gate_failed(results) else "PASS"

    lines = [
        "# Quality Gate Report",
        "",
        f"- Timestamp: {timestamp}",
        f"- Project root: {PROJECT_ROOT.name}",
        f"- Overall: {overall}",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for r in results:
        detail = r.detail.replace("|", "\\|")
        lines.append(f"| {r.name} | {r.status} | {detail} |")
    lines += [
        "",
        "## Notes",
        "",
        "- HARD checks (LINT, TESTS, SCHEMA) determine the exit code. "
        "DRIFT is informational and SKIPPED checks do not fail the gate.",
        f"- Data validated: {PROCESSED_DIR} (per-ticker CSVs) and "
        f"{NEWS_PANEL.name}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_gate_json(
    results: list[CheckResult],
    out_dir: Path,
    commit: str,
    branch: str | None = None,
    diff_cover_pct: float | None = None,
    timestamp: str | None = None,
) -> Path:
    """Write a machine-readable per-commit gate result the dashboard can overlay.

    Emits ``<out_dir>/<commit>.json`` with a flat schema (tests_passed,
    diff_cover_pct, ruff, pandera, evidently, overall) so future dashboard rows
    are sourced from real runs, not hand-entry.
    """
    import json

    by_name = {r.name: r for r in results}

    def _status(name: str) -> str | None:
        r = by_name.get(name)
        return r.status if r else None

    tests = _status("TESTS")
    lint = _status("LINT")
    schema = _status("SCHEMA")
    drift = _status("DRIFT")

    payload = {
        "commit": commit,
        "branch": branch,
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "tests_passed": (tests == PASS) if tests is not None else None,
        "diff_cover_pct": diff_cover_pct,
        "ruff": "pass" if lint == PASS else ("fail" if lint == FAIL else "na"),
        "pandera": "pass" if schema == PASS else ("fail" if schema == FAIL else "na"),
        "evidently": (drift or "na").lower(),
        "overall": "fail" if gate_failed(results) else "pass",
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{commit}.json"
    # Do not clobber a richer coverage value already written for this commit
    # (the pre-push hook measures diff-cover; this entry point cannot). If this
    # run has no coverage number, keep the existing one.
    if payload["diff_cover_pct"] is None and path.exists():
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior = {}
        if isinstance(prior, dict) and prior.get("diff_cover_pct") is not None:
            payload["diff_cover_pct"] = prior["diff_cover_pct"]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-code quality gate.")
    parser.add_argument(
        "--fast", action="store_true", help="skip the DRIFT step for a quick run"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results = run_gate(fast=args.fast, timestamp=timestamp)
    print_summary(results)
    report_path = write_report(results, timestamp)
    print(f"Report written: {report_path}")

    # Also emit a per-commit gate-result JSON the task dashboard overlays. Best
    # effort: skip silently if HEAD is unavailable. diff-cover % is not measured
    # here (the pre-push hook is the primary source for that), so it is left null.
    try:
        import subprocess

        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if commit:
            gate_dir = PROJECT_ROOT / "scripts" / "task_dashboard" / "gate_results"
            json_path = write_gate_json(results, gate_dir, commit=commit, branch=branch or None)
            print(f"Gate result written: {json_path}")
    except Exception as exc:  # noqa: BLE001 - JSON emission must never fail the gate
        print(f"(gate-result JSON not written: {exc})")

    return 1 if gate_failed(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
