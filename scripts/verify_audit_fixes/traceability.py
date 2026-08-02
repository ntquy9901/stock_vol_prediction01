"""Build acceptance_traceability.csv (one row per audit finding).

Schema and allowed status values come from
docs/reports/2026-08-02_152758_summaryOfUpdate_report.md ("Required
traceability schema"). This module only validates and serializes rows — it
does not parse audit-report prose into findings. The caller (the
verify-audit-fixes skill, run by a Claude session that has read the audit
report) supplies structured finding dicts.
"""
from __future__ import annotations

import csv
from pathlib import Path

FIELDS = [
    "finding_id",
    "severity",
    "requirement",
    "status",
    "changed_files",
    "test_ids",
    "commands",
    "evidence_files",
    "review_disposition",
    "notes",
]

ALLOWED_STATUSES = {
    "Verified fixed",
    "Partially fixed",
    "Not fixed",
    "Not verifiable",
    "Not applicable",
}


class TraceabilityError(ValueError):
    pass


def _join(value) -> str:
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return "" if value is None else str(value)


def validate_finding(finding: dict) -> None:
    missing = [f for f in ("finding_id", "severity", "requirement", "status") if not finding.get(f)]
    if missing:
        raise TraceabilityError(f"finding missing required field(s) {missing}: {finding}")

    status = finding["status"]
    if status not in ALLOWED_STATUSES:
        raise TraceabilityError(
            f"finding {finding['finding_id']!r} has disallowed status {status!r}; "
            f"must be one of {sorted(ALLOWED_STATUSES)}"
        )
    if status == "Not applicable" and not finding.get("notes"):
        raise TraceabilityError(
            f"finding {finding['finding_id']!r} is 'Not applicable' but has no justification in 'notes'"
        )
    if status == "Verified fixed":
        if not finding.get("test_ids") or not finding.get("commands") or not finding.get("evidence_files"):
            raise TraceabilityError(
                f"finding {finding['finding_id']!r} is 'Verified fixed' but is missing "
                "test_ids/commands/evidence_files — a finding must not be marked "
                "'Verified fixed' unless the cited test and command both passed and "
                "their raw evidence files exist."
            )


def build_rows(findings: list[dict]) -> list[dict]:
    rows = []
    for finding in findings:
        validate_finding(finding)
        rows.append({field: _join(finding.get(field)) for field in FIELDS})
    return rows


def write_traceability_csv(findings: list[dict], evidence_dir: Path) -> Path:
    path = evidence_dir / "acceptance_traceability.csv"
    rows = build_rows(findings)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
