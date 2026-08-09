"""Tests for the quality gate orchestrator and data schemas.

Uses tiny in-memory / tmp fixtures and monkeypatching so tests never depend on
the full training data and never recursively invoke pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.quality_gate import data_schemas as ds  # noqa: E402
from scripts.quality_gate import run_quality_gate as rqg  # noqa: E402


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def _write_processed(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def test_processed_schema_passes_valid(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "parkinson_volatility": [0.01, 0.02, 0.015],
        }
    )
    csv = tmp_path / "AAA_processed.csv"
    _write_processed(csv, df)
    name, status, detail = ds._check_processed_file(csv)
    assert status == ds.VALID, detail


def test_processed_schema_fails_negative_volatility(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "parkinson_volatility": [0.01, -0.5],
        }
    )
    csv = tmp_path / "BAD_processed.csv"
    _write_processed(csv, df)
    _, status, _ = ds._check_processed_file(csv)
    assert status == ds.INVALID


def test_processed_schema_fails_high_lt_low(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "open": [10.0, 10.0],
            "high": [9.0, 11.0],  # first row: high < low
            "low": [10.0, 10.0],
            "close": [9.5, 10.5],
            "volume": [100, 200],
            "parkinson_volatility": [0.01, 0.02],
        }
    )
    csv = tmp_path / "OHLC_processed.csv"
    _write_processed(csv, df)
    _, status, detail = ds._check_processed_file(csv)
    assert status == ds.INVALID
    assert "high < low" in detail


def test_processed_schema_fails_nonmonotonic_dates(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "date": ["2020-01-03", "2020-01-01"],
            "parkinson_volatility": [0.01, 0.02],
        }
    )
    csv = tmp_path / "ORDER_processed.csv"
    _write_processed(csv, df)
    _, status, detail = ds._check_processed_file(csv)
    assert status == ds.INVALID
    assert "monotonic" in detail


def test_validate_data_on_tiny_fixtures(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    _write_processed(
        proc / "AAA_processed.csv",
        pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "parkinson_volatility": [0.01, 0.02],
            }
        ),
    )
    panel = tmp_path / "panel.parquet"
    pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]).astype(
                "datetime64[us]"
            ),
            "kq_emb_0": [0.1, -0.2],
            "kq_emb_norm": [1.0, 2.0],
            "kq_topic_earnings_count": [0.0, 1.0],
        }
    ).to_parquet(panel)

    results = ds.validate_data(processed_dir=proc, news_panel=panel)
    assert all(s == ds.VALID for _, s, _ in results), results


def test_validate_data_flags_bad_news_norm(tmp_path: Path) -> None:
    proc = tmp_path / "processed"
    proc.mkdir()
    _write_processed(
        proc / "AAA_processed.csv",
        pd.DataFrame(
            {"date": ["2020-01-01"], "parkinson_volatility": [0.01]}
        ),
    )
    panel = tmp_path / "panel.parquet"
    pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": pd.to_datetime(["2020-01-01"]).astype("datetime64[us]"),
            "kq_emb_norm": [-5.0],  # negative norm -> invalid
        }
    ).to_parquet(panel)

    results = ds.validate_data(processed_dir=proc, news_panel=panel)
    news = [r for r in results if r[0] == "news_panel"][0]
    assert news[1] == ds.INVALID


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


def test_gate_failed_logic() -> None:
    ok = [rqg.CheckResult("SCHEMA", rqg.PASS, "")]
    assert not rqg.gate_failed(ok)
    hard = [rqg.CheckResult("TESTS", rqg.FAIL, "")]
    assert rqg.gate_failed(hard)
    # DRIFT failing/INFO never fails the gate.
    soft = [rqg.CheckResult("DRIFT", rqg.FAIL, "")]
    assert not rqg.gate_failed(soft)


def test_run_gate_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    # Monkeypatch subprocess-backed checks to avoid recursion and keep it light.
    monkeypatch.setattr(
        rqg, "check_lint", lambda: rqg.CheckResult("LINT", rqg.SKIPPED, "stub")
    )
    monkeypatch.setattr(
        rqg, "check_tests", lambda: rqg.CheckResult("TESTS", rqg.PASS, "stub")
    )
    monkeypatch.setattr(
        rqg,
        "check_schema",
        lambda: rqg.CheckResult("SCHEMA", rqg.PASS, "stub"),
    )
    results = rqg.run_gate(fast=True)
    assert isinstance(results, list)
    names = {r.name for r in results}
    assert {"LINT", "TESTS", "SCHEMA", "DRIFT"} <= names
    assert not rqg.gate_failed(results)


def test_write_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rqg, "PROJECT_ROOT", tmp_path)
    results = [
        rqg.CheckResult("LINT", rqg.SKIPPED, "ruff not installed"),
        rqg.CheckResult("SCHEMA", rqg.PASS, "34/34 valid"),
    ]
    path = rqg.write_report(results, "2020-01-01_000000")
    assert path.exists()
    # Filename reuses the passed run timestamp (seconds precision), so two runs
    # in the same minute do not overwrite and it matches the body/drift dir.
    assert path.name == "2020-01-01_000000_quality_gate_report.md"
    text = path.read_text(encoding="utf-8")
    assert "Quality Gate Report" in text
    assert "SCHEMA" in text


def test_write_gate_json(tmp_path: Path) -> None:
    import json

    results = [
        rqg.CheckResult("TESTS", rqg.PASS, "13 passed"),
        rqg.CheckResult("LINT", rqg.PASS, "no lint errors"),
        rqg.CheckResult("SCHEMA", rqg.PASS, "34/34 valid"),
        rqg.CheckResult("DRIFT", rqg.INFO, "ACB ref=2800/cur=1200"),
    ]
    out_dir = tmp_path / "gate_results"
    path = rqg.write_gate_json(
        results,
        out_dir,
        commit="abc1234",
        branch="master",
        diff_cover_pct=87.0,
        timestamp="2020-01-01T00:00",
    )
    assert path.exists()
    assert path.name == "abc1234.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["commit"] == "abc1234"
    assert data["tests_passed"] is True
    assert data["diff_cover_pct"] == 87.0
    assert data["ruff"] == "pass"
    assert data["pandera"] == "pass"
    assert data["overall"] == "pass"


def test_main_writes_gate_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import subprocess as _sp

    monkeypatch.setattr(rqg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rqg, "check_lint", lambda: rqg.CheckResult("LINT", rqg.PASS, "ok"))
    monkeypatch.setattr(rqg, "check_tests", lambda: rqg.CheckResult("TESTS", rqg.PASS, "ok"))
    monkeypatch.setattr(rqg, "check_schema", lambda: rqg.CheckResult("SCHEMA", rqg.PASS, "ok"))
    monkeypatch.setattr(sys, "argv", ["run_quality_gate.py", "--fast"])

    def _fake_git(cmd, *a, **k):
        val = "deadbee" if "--short" in cmd else "master"
        return _sp.CompletedProcess(cmd, 0, stdout=val + "\n", stderr="")

    monkeypatch.setattr(_sp, "run", _fake_git)

    rc = rqg.main()
    assert rc == 0
    gate_json = tmp_path / "scripts" / "task_dashboard" / "gate_results" / "deadbee.json"
    assert gate_json.exists()
    data = json.loads(gate_json.read_text(encoding="utf-8"))
    assert data["commit"] == "deadbee"
    assert data["branch"] == "master"
    assert data["overall"] == "pass"


def test_main_gate_json_failure_is_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A crash while emitting the gate JSON must never fail the gate itself.
    import subprocess as _sp

    monkeypatch.setattr(rqg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(rqg, "check_lint", lambda: rqg.CheckResult("LINT", rqg.PASS, "ok"))
    monkeypatch.setattr(rqg, "check_tests", lambda: rqg.CheckResult("TESTS", rqg.PASS, "ok"))
    monkeypatch.setattr(rqg, "check_schema", lambda: rqg.CheckResult("SCHEMA", rqg.PASS, "ok"))
    monkeypatch.setattr(sys, "argv", ["run_quality_gate.py", "--fast"])

    def _boom(*a, **k):
        raise RuntimeError("git exploded")

    monkeypatch.setattr(_sp, "run", _boom)
    assert rqg.main() == 0  # gate still passes despite JSON emission crashing


def test_write_gate_json_preserves_existing_coverage_when_null(tmp_path: Path) -> None:
    # The pre-push hook writes a real diff_cover_pct; a later run_quality_gate
    # run (which cannot measure coverage) must NOT clobber it with null.
    import json

    out_dir = tmp_path / "gate_results"
    results = [
        rqg.CheckResult("TESTS", rqg.PASS, "ok"),
        rqg.CheckResult("LINT", rqg.PASS, "ok"),
        rqg.CheckResult("SCHEMA", rqg.PASS, "ok"),
    ]
    rqg.write_gate_json(results, out_dir, commit="abc1234",
                        diff_cover_pct=87.0, timestamp="2020-01-01T00:00")
    rqg.write_gate_json(results, out_dir, commit="abc1234",
                        diff_cover_pct=None, timestamp="2020-01-02T00:00")
    data = json.loads((out_dir / "abc1234.json").read_text(encoding="utf-8"))
    assert data["diff_cover_pct"] == 87.0


def test_write_gate_json_tolerates_malformed_existing_file(tmp_path: Path) -> None:
    # A corrupt pre-existing gate file must not crash the null-coverage merge;
    # there is simply nothing to preserve.
    import json

    out_dir = tmp_path / "gate_results"
    out_dir.mkdir()
    (out_dir / "abc1234.json").write_text("{not json", encoding="utf-8")
    results = [
        rqg.CheckResult("TESTS", rqg.PASS, "ok"),
        rqg.CheckResult("LINT", rqg.PASS, "ok"),
        rqg.CheckResult("SCHEMA", rqg.PASS, "ok"),
    ]
    path = rqg.write_gate_json(results, out_dir, commit="abc1234",
                               diff_cover_pct=None, timestamp="2020-01-03T00:00")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["diff_cover_pct"] is None


# ---------------------------------------------------------------------------
# Fixed-behaviour regression tests (review findings 2026-08-08)
# ---------------------------------------------------------------------------


def test_missing_inputs_are_skipped_not_failed(tmp_path: Path) -> None:
    # No processed dir and no news panel -> MISSING, and SCHEMA must SKIP (never
    # a hard FAIL) so a fresh checkout without data does not break the gate.
    missing_dir = tmp_path / "nope"
    missing_panel = tmp_path / "nope.parquet"
    results = ds.validate_data(processed_dir=missing_dir, news_panel=missing_panel)
    assert all(s == ds.MISSING for _, s, _ in results), results

    monkey_results = results
    import scripts.quality_gate.data_schemas as ds_mod

    def _fake_validate() -> list:
        return monkey_results

    orig = ds_mod.validate_data
    ds_mod.validate_data = _fake_validate  # type: ignore[assignment]
    try:
        result = rqg.check_schema()
    finally:
        ds_mod.validate_data = orig  # type: ignore[assignment]
    assert result.status == rqg.SKIPPED


def test_news_panel_tolerates_all_nan_embedding(tmp_path: Path) -> None:
    panel = tmp_path / "panel.parquet"
    pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "kq_emb_0": [float("nan"), float("nan")],  # all-NaN embedding col
            "kq_topic_x_count": [0.0, 1.0],
        }
    ).to_parquet(panel)
    name, status, detail = ds._check_news_panel(panel)
    assert status == ds.VALID, detail


def test_check_tests_skips_when_pytest_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    def _fake_run(cmd, timeout: int = 900):  # noqa: ANN001, ARG001
        return types.SimpleNamespace(
            returncode=1, stdout="", stderr="No module named pytest"
        )

    monkeypatch.setattr(rqg, "_run", _fake_run)
    result = rqg.check_tests()
    assert result.status == rqg.SKIPPED


def test_check_lint_crash_is_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rqg.shutil, "which", lambda _: "ruff")

    def _boom(cmd, timeout: int = 900):  # noqa: ANN001, ARG001
        raise TimeoutError("ruff hung")

    monkeypatch.setattr(rqg, "_run", _boom)
    result = rqg.check_lint()
    # A crashing/timing-out ruff is a HARD FAIL, not a silent SKIP.
    assert result.status == rqg.FAIL
