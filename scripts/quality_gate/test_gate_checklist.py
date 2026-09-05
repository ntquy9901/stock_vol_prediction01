"""Unit tests for the gate checklist evidence + completeness verifier."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gate_checklist as gc  # noqa: E402


def _full_pass():
    return {
        "commit": "abc1234", "timestamp": "2026-09-05T23:00:00", "tests_passed": True,
        "cov_c0_line_pct": 100.0, "cov_c1_branch_pct": 100.0, "ruff": "pass",
        "lessons_regression": "pass", "pandera": "pass", "evidently": "pass", "overall": "pass",
    }


def test_verify_ok_on_complete_pass():
    ok, problems = gc.verify_checklist(_full_pass())
    assert ok and problems == []


def test_verify_flags_missing_field():
    p = _full_pass(); del p["cov_c1_branch_pct"]
    ok, problems = gc.verify_checklist(p)
    assert not ok
    assert any("cov_c1_branch_pct" in x for x in problems)


def test_verify_flags_none_field():
    p = _full_pass(); p["pandera"] = None
    ok, problems = gc.verify_checklist(p)
    assert not ok and any("pandera" in x for x in problems)


def test_verify_accepts_none_coverage_docs_only():
    # cov None = 'na' (docs-only / no source to gate) is acceptable, not a gap
    p = _full_pass(); p["cov_c0_line_pct"] = None; p["cov_c1_branch_pct"] = None
    ok, problems = gc.verify_checklist(p)
    assert ok and problems == []


def test_verify_flags_missing_coverage_key():
    p = _full_pass(); del p["cov_c0_line_pct"]
    ok, problems = gc.verify_checklist(p)
    assert not ok and any("cov_c0_line_pct" in x for x in problems)


def test_verify_flags_pass_hiding_failed_tests():
    p = _full_pass(); p["tests_passed"] = False
    ok, problems = gc.verify_checklist(p)
    assert not ok and any("tests_passed" in x for x in problems)


def test_verify_flags_pass_hiding_blocking_fail():
    p = _full_pass(); p["evidently"] = "fail"
    ok, problems = gc.verify_checklist(p)
    assert not ok and any("evidently=fail" in x for x in problems)


def test_verify_skips_consistency_when_not_pass():
    # overall != pass -> consistency block skipped; still complete if all fields present
    p = _full_pass(); p["overall"] = "fail"; p["tests_passed"] = False
    ok, problems = gc.verify_checklist(p)
    assert ok and problems == []


def test_build_checklist_marks_complete_and_missing():
    md = gc.build_checklist(_full_pass())
    assert "Evidence-complete: **YES**" in md
    assert "| tests_passed | True |" in md
    p = _full_pass(); del p["ruff"]
    md2 = gc.build_checklist(p)
    assert "Evidence-complete: **NO**" in md2
    assert "MISSING" in md2 and "Problems:" in md2
