import csv

import pytest

from scripts.verify_audit_fixes.traceability import (
    TraceabilityError,
    build_rows,
    validate_finding,
    write_traceability_csv,
)


def _finding(**overrides):
    base = {
        "finding_id": "VER-001",
        "severity": "HIGH",
        "requirement": "Raw command output retained",
        "status": "Not fixed",
        "changed_files": ["scripts/verify_audit_fixes/commands.py"],
        "test_ids": [],
        "commands": [],
        "evidence_files": [],
        "review_disposition": "",
        "notes": "",
    }
    base.update(overrides)
    return base


def test_validate_finding_rejects_missing_required_fields():
    with pytest.raises(TraceabilityError):
        validate_finding({"finding_id": "X"})


def test_validate_finding_rejects_disallowed_status():
    with pytest.raises(TraceabilityError):
        validate_finding(_finding(status="Fixed"))


def test_validate_finding_allows_all_documented_statuses():
    for status in (
        "Verified fixed",
        "Partially fixed",
        "Not fixed",
        "Not verifiable",
        "Not applicable",
    ):
        kwargs = {"status": status, "notes": "justification"}
        if status == "Verified fixed":
            kwargs.update(test_ids=["t1"], commands=["cmd"], evidence_files=["ev.txt"])
        validate_finding(_finding(**kwargs))  # must not raise


def test_validate_finding_not_applicable_requires_justification():
    with pytest.raises(TraceabilityError):
        validate_finding(_finding(status="Not applicable", notes=""))


def test_validate_finding_verified_fixed_requires_evidence():
    with pytest.raises(TraceabilityError):
        validate_finding(_finding(status="Verified fixed", test_ids=[], commands=[], evidence_files=[]))


def test_validate_finding_verified_fixed_passes_with_evidence():
    validate_finding(
        _finding(
            status="Verified fixed",
            test_ids=["tests/test_x.py::test_y"],
            commands=["python -m pytest"],
            evidence_files=["pytest_full.txt"],
        )
    )  # must not raise


def test_build_rows_joins_list_fields():
    rows = build_rows([_finding(changed_files=["a.py", "b.py"])])
    assert rows[0]["changed_files"] == "a.py;b.py"


def test_write_traceability_csv_writes_header_and_rows(tmp_path):
    findings = [_finding(finding_id="VER-001"), _finding(finding_id="VER-002", status="Not applicable", notes="n/a: docs-only")]
    path = write_traceability_csv(findings, tmp_path)
    assert path.exists()
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["finding_id"] for r in rows] == ["VER-001", "VER-002"]


def test_write_traceability_csv_raises_on_invalid_finding(tmp_path):
    with pytest.raises(TraceabilityError):
        write_traceability_csv([_finding(status="bogus-status")], tmp_path)
