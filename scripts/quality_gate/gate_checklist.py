"""Quality-gate checklist evidence + completeness verifier (per user 2026-09-05).

The pre-push gate emits a per-commit result JSON. This module turns that into a human-readable CHECKLIST
evidence file (one row per required check + its recorded result) and, as the FINAL gate step, VERIFIES the
checklist is complete: every required check must have a recorded result (no silent gaps) and an
``overall: pass`` must not hide a failed blocking check. If the evidence is incomplete or inconsistent the
verifier exits non-zero so the push is blocked — i.e. the gate proves it actually checked everything.

Run (final pre-push step): python scripts/quality_gate/gate_checklist.py <gate_results/<sha>.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Every required check must be PRESENT as a key (a silently-dropped check = incomplete evidence).
REQUIRED_KEYS = [
    "commit", "timestamp", "tests_passed", "cov_c0_line_pct", "cov_c1_branch_pct",
    "ruff", "lessons_regression", "pandera", "evidently", "overall",
]
# ...and these must also be non-None. Coverage % is deliberately None on a docs-only / no-source push
# ("na" -> no source lines to gate), so cov_c0/c1 are present-but-may-be-None and are NOT listed here.
REQUIRED_NONNULL = [
    "commit", "timestamp", "tests_passed", "ruff", "lessons_regression", "pandera", "evidently", "overall",
]
# Fields whose value must not be a failure when the run is declared overall-pass.
BLOCKING_FIELDS = ["lessons_regression", "pandera", "evidently"]


def verify_checklist(payload: dict) -> tuple[bool, list[str]]:
    """Return (ok, problems). Flags missing required fields (silent gaps) and any overall-pass that hides a
    failed blocking check or failed tests -- so a 'pass' can only stand on complete, consistent evidence."""
    problems: list[str] = []
    for k in REQUIRED_KEYS:
        if k not in payload:
            problems.append(f"missing required check: {k}")
    for k in REQUIRED_NONNULL:
        if k in payload and payload.get(k) is None:
            problems.append(f"empty required check: {k}")
    if payload.get("overall") == "pass":
        if payload.get("tests_passed") is False:
            problems.append("overall=pass but tests_passed=false")
        for k in BLOCKING_FIELDS:
            if payload.get(k) == "fail":
                problems.append(f"overall=pass but {k}=fail")
    return (not problems, problems)


def _mark(payload: dict, key: str) -> str:
    if key not in payload or payload.get(key) is None:
        return "MISSING"
    return str(payload[key])


def build_checklist(payload: dict) -> str:
    """Render the gate result as a markdown checklist (evidence artifact)."""
    ok, problems = verify_checklist(payload)
    rows = "\n".join(f"| {k} | {_mark(payload, k)} |" for k in REQUIRED_KEYS)
    head = (f"# Quality-gate checklist — {payload.get('commit', '?')} "
            f"({payload.get('timestamp', '?')})\n\n"
            f"Evidence-complete: **{'YES' if ok else 'NO'}**\n\n"
            "| check | result |\n|---|---|\n")
    tail = "" if ok else ("\n**Problems:**\n" + "\n".join(f"- {p}" for p in problems) + "\n")
    return head + rows + "\n" + tail


def main(argv=None) -> int:  # pragma: no cover - entry driver (file I/O)
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("[gate-checklist] usage: gate_checklist.py <gate_result.json>")
        return 0
    src = Path(argv[0])
    if not src.exists():
        print(f"[gate-checklist] BLOCK: gate result JSON not found: {src}")
        return 1
    payload = json.loads(src.read_text(encoding="utf-8"))
    out = src.with_name(src.stem + "_checklist.md")
    out.write_text(build_checklist(payload), encoding="utf-8")
    ok, problems = verify_checklist(payload)
    print(f"[gate-checklist] wrote {out}")
    if ok:
        print("[gate-checklist] evidence complete: all required checks recorded + consistent.")
        return 0
    print("[gate-checklist] BLOCK: evidence incomplete/inconsistent:")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
