"""Tests for the task-dashboard generator (RED-first, per TDD)."""
import json
from pathlib import Path

import build_dashboard as bd


def _seed_task() -> dict:
    return {
        "id": "quality-gate",
        "phase": "Phase 0 - Foundations",
        "title": "Post-code quality gate",
        "status": "done",
        "timestamp": "2026-08-08T15:42:00",
        "branch": "master",
        "commits": ["48a14bb"],
        "skills_applied": ["3-layer-code-review", "verification-before-completion"],
        "evidence": [{"cmd": "pytest -q", "result": "13 passed in 1.28s"}],
        "quality_gate": {"tests": "pass", "lint": "pass", "code_review": "pass", "diff_cover": "skip"},
        "code_review": {"layers": 3, "findings_fixed": 4, "findings_deferred": 0},
        "dod": [{"item": "tests green", "ok": True}],
        "result_summary": "Quality gate built and hardened.",
        "report_md": "docs/reports/2026-08-08_1542_summaryOfUpdate_report.md",
    }


def test_build_html_contains_seeded_task_title_and_status():
    html = bd.build_html([_seed_task()])
    assert "<html" in html and "</html>" in html
    assert "Post-code quality gate" in html
    # status badge present for 'done'
    assert "done" in html
    assert "status-badge" in html


def test_build_html_shows_evidence_output():
    html = bd.build_html([_seed_task()])
    assert "13 passed in 1.28s" in html
    assert "pytest -q" in html


def test_build_html_shows_commit_and_report_link():
    html = bd.build_html([_seed_task()])
    assert "48a14bb" in html
    assert "2026-08-08_1542_summaryOfUpdate_report.md" in html


def test_empty_ledger_produces_valid_empty_state():
    html = bd.build_html([])
    assert "<html" in html and "</html>" in html
    assert "empty-state" in html


def test_malformed_entry_handled_gracefully():
    # Entry missing every optional key must not crash the generator.
    html = bd.build_html([{}])
    assert "<html" in html and "</html>" in html
    # A malformed entry still renders a card (with fallback text).
    assert "task-card" in html


def test_status_counts_computed():
    tasks = [
        {"status": "done"},
        {"status": "done"},
        {"status": "running"},
        {"status": "blocked"},
    ]
    counts = bd.status_counts(tasks)
    assert counts["done"] == 2
    assert counts["running"] == 1
    assert counts["blocked"] == 1
    assert counts["planned"] == 0


def test_load_ledger_reads_array(tmp_path: Path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps([_seed_task()]), encoding="utf-8")
    tasks = bd.load_ledger(p)
    assert len(tasks) == 1
    assert tasks[0]["id"] == "quality-gate"


def test_load_ledger_missing_file_returns_empty(tmp_path: Path):
    tasks = bd.load_ledger(tmp_path / "does_not_exist.json")
    assert tasks == []


def test_main_writes_html_file(tmp_path: Path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([_seed_task()]), encoding="utf-8")
    out = tmp_path / "out" / "dashboard.html"
    bd.main(ledger_path=ledger, out_path=out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Post-code quality gate" in text
    assert "status-badge" in text


# ---------------------------------------------------------------------------
# Remediation: gate rendering (object + prose), data-quality column
# ---------------------------------------------------------------------------


def _code_task(**over) -> dict:
    t = _seed_task()
    t["task_type"] = "code"
    t["quality_gate"] = {
        "tests": "pass",
        "lint": "pass",
        "code_review": "pass",
        "diff_cover": "pass",
        "data_quality": "pass",
    }
    t.update(over)
    return t


def test_gate_object_renders_icon_and_detail():
    # A {status, detail} object must render the detail text AND a status icon,
    # never collapse to a bare dash.
    t = _code_task(
        quality_gate={
            "tests": {"status": "pass", "detail": "13 passed"},
            "lint": {"status": "pass", "detail": "clean"},
            "code_review": {"status": "pass", "detail": "3-layer"},
            "diff_cover": {"status": "pass", "detail": "100%"},
            "data_quality": {"status": "pass", "detail": "34/34"},
        }
    )
    html = bd.build_html([t])
    assert "13 passed" in html
    assert "gate-pass" in html


def test_gate_prose_renders_with_icon_not_dashed():
    # Free-text prose (legacy hand-entry) must still surface its text, not the
    # muted "—" dash the old renderer produced.
    t = _code_task(
        quality_gate={
            "tests": "pass",
            "lint": "pass",
            "code_review": "pass",
            "diff_cover": "NOW RUN: 87% Track B, floor 80 enforced",
            "data_quality": "n/a",
        }
    )
    html = bd.build_html([t])
    assert "87%" in html
    # prose gets a real status icon (warn), not the muted "—" dash of the old renderer
    assert "gate-warn" in html


def test_data_quality_column_present():
    html = bd.build_html([_code_task()])
    assert "Data-quality" in html


def test_code_task_missing_gate_gets_incomplete_red_card():
    t = _code_task(
        id="gap",
        quality_gate={
            "tests": "pass",
            "lint": "pass",
            "code_review": "pass",
            "diff_cover": {"status": "miss", "detail": "skip (pre-tooling)"},
            "data_quality": "n/a",
        },
    )
    html = bd.build_html([t])
    assert "INCOMPLETE" in html
    assert 'card-incomplete"' in html


def test_na_task_not_reddened():
    doc = _seed_task()
    doc["id"] = "adoc"
    doc["task_type"] = "doc"
    doc["quality_gate"] = {
        "tests": "n/a",
        "lint": "n/a",
        "code_review": "n/a",
        "diff_cover": "n/a",
        "data_quality": "n/a",
    }
    html = bd.build_html([doc])
    assert 'card-incomplete"' not in html
    assert "INCOMPLETE" not in html


def test_fully_gated_code_task_not_reddened():
    html = bd.build_html([_code_task(id="ok")])
    assert 'card-incomplete"' not in html


# ---------------------------------------------------------------------------
# Remediation: per-commit gate-result JSON overlay (real run wins)
# ---------------------------------------------------------------------------


def test_load_gate_results_reads_dir(tmp_path: Path):
    d = tmp_path / "gate_results"
    d.mkdir()
    (d / "abc1234.json").write_text(
        json.dumps({"commit": "abc1234", "diff_cover_pct": 88.0}), encoding="utf-8"
    )
    gr = bd.load_gate_results(d)
    assert "abc1234" in gr
    assert gr["abc1234"]["diff_cover_pct"] == 88.0


def test_gate_results_overlay_real_value_wins():
    # Hand-entry says diff_cover missing; the real per-commit gate JSON says 91%.
    t = _code_task(
        id="ov",
        commits=["deadbee"],
        quality_gate={
            "tests": "pass",
            "lint": "pass",
            "code_review": "pass",
            "diff_cover": {"status": "miss", "detail": "skip"},
            "data_quality": "pass",
        },
    )
    gr = {
        "deadbee": {
            "commit": "deadbee",
            "tests_passed": True,
            "diff_cover_pct": 91.0,
            "ruff": "pass",
            "pandera": "pass",
            "timestamp": "2026-08-09T01:00",
        }
    }
    html = bd.build_html([t], gate_results=gr)
    assert "91" in html
    # Overlaid diff_cover now passes -> the task is no longer incomplete.
    assert 'card-incomplete"' not in html


# ---------------------------------------------------------------------------
# Remediation: newest-first ordering
# ---------------------------------------------------------------------------


def test_newest_task_renders_first():
    old = _code_task(id="old", title="OLDEST TASK", timestamp="2026-08-01T10:00")
    new = _code_task(id="new", title="NEWEST TASK", timestamp="2026-08-09T10:00")
    html = bd.build_html([old, new])
    assert html.index("NEWEST TASK") < html.index("OLDEST TASK")


# ---------------------------------------------------------------------------
# Helper-level coverage: gate resolution, validation, overlay edge cases
# ---------------------------------------------------------------------------


def test_resolve_gate_prose_status_derivation():
    # Empty -> na; explicit n/a prose -> na; failure prose -> fail;
    # pass prose -> pass; anything else (a number/note) -> warn (never dropped).
    assert bd._resolve_gate("")[0] == "na"
    assert bd._resolve_gate("n/a (doc)")[0] == "na"
    assert bd._resolve_gate("not applicable here")[0] == "na"
    assert bd._resolve_gate("SCHEMA failed on ACB")[0] == "fail"
    assert bd._resolve_gate("passed 13 tests")[0] == "pass"
    assert bd._resolve_gate("87% coverage")[0] == "warn"


def test_resolve_gate_object_with_detail_only():
    status, detail = bd._resolve_gate({"detail": "34/34 valid"})
    assert detail == "34/34 valid"
    assert status == "warn"  # no explicit status, non-na detail -> warn


def test_validate_task_data_gate_miss_flags_incomplete():
    # All four core gates pass, but the data gate is an explicit miss on a
    # data-touching code task -> incomplete via the data gate.
    t = _code_task(
        id="datamiss",
        quality_gate={
            "tests": "pass",
            "lint": "pass",
            "code_review": "pass",
            "diff_cover": "pass",
            "data_quality": {"status": "miss", "detail": "data changed, gate not run"},
        },
    )
    incomplete, missing = bd.validate_task(t)
    assert incomplete
    assert "data_quality" in missing


def test_load_gate_results_missing_dir_returns_empty(tmp_path: Path):
    assert bd.load_gate_results(tmp_path / "nope") == {}


def test_load_gate_results_skips_invalid_and_nondict(tmp_path: Path):
    d = tmp_path / "gate_results"
    d.mkdir()
    (d / "bad.json").write_text("{not valid json", encoding="utf-8")
    (d / "list.json").write_text("[1, 2, 3]", encoding="utf-8")
    (d / "ok.json").write_text(json.dumps({"commit": "ok"}), encoding="utf-8")
    gr = bd.load_gate_results(d)
    assert "bad" not in gr and "list" not in gr
    assert "ok" in gr


def test_overlay_gate_no_matching_commit_returns_unchanged():
    t = _code_task(id="nomatch", commits=["zzzzzzz"])
    out = bd._overlay_gate(t, {"other": {"diff_cover_pct": 50.0}})
    assert out is t  # untouched when no commit matches


def test_overlay_gate_malformed_diff_cover_does_not_crash_build():
    # A gate-result JSON whose diff_cover_pct is non-numeric must NOT crash the
    # whole dashboard build; the bad value is skipped, the rest still renders.
    t = _code_task(id="badpct", commits=["cafef00"])
    gr = {"cafef00": {"commit": "cafef00", "diff_cover_pct": "N/A",
                      "timestamp": "2026-08-09T02:00"}}
    html = bd.build_html([t], gate_results=gr)  # must not raise
    assert "badpct" in html


def test_overlay_gate_ruff_fail_renders_fail_not_warn():
    # A real ruff FAILURE from the hook must surface as fail (red), not a soft warn.
    t = _code_task(id="rufffail", commits=["beadfab"])
    gr = {"beadfab": {"commit": "beadfab", "ruff": "fail",
                      "timestamp": "2026-08-09T02:00"}}
    out = bd._overlay_gate(t, gr)
    assert bd._resolve_gate(out["quality_gate"]["lint"])[0] == "fail"


def test_overlay_gate_ruff_na_does_not_override_handentry():
    # ruff "na" (not measured on the changed files) must NOT downgrade a
    # hand-entered lint pass to warn.
    t = _code_task(id="ruffna", commits=["beadfab"])
    gr = {"beadfab": {"commit": "beadfab", "ruff": "na",
                      "timestamp": "2026-08-09T02:00"}}
    out = bd._overlay_gate(t, gr)
    assert bd._resolve_gate(out["quality_gate"]["lint"])[0] == "pass"


# ---------------------------------------------------------------------------
# Guard-branch coverage (empty/blank/non-dict inputs)
# ---------------------------------------------------------------------------


def test_load_ledger_blank_file_returns_empty(tmp_path: Path):
    p = tmp_path / "blank.json"
    p.write_text("   \n", encoding="utf-8")
    assert bd.load_ledger(p) == []


def test_load_ledger_non_array_raises(tmp_path: Path):
    p = tmp_path / "obj.json"
    p.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    try:
        bd.load_ledger(p)
    except ValueError as exc:
        assert "JSON array" in str(exc)
    else:  # pragma: no cover - the raise above is the contract
        raise AssertionError("expected ValueError for non-array ledger")


def test_status_counts_unknown_status_falls_back_to_planned():
    counts = bd.status_counts([{"status": "mystery"}])
    assert counts["planned"] == 1


def test_render_overview_zero_total_has_no_stackbar():
    html = bd._render_overview({"done": 0, "running": 0, "blocked": 0, "planned": 0})
    assert "stackbar" not in html


def test_render_evidence_skips_non_dict_items():
    # A non-dict evidence entry is skipped; an all-non-dict list yields "".
    assert bd._render_evidence(["not-a-dict"]) == ""
    html = bd._render_evidence([{"cmd": "x", "result": "y"}, 42])
    assert "x" in html and "y" in html


def test_render_dod_skips_non_dict_items():
    assert bd._render_dod([123, "nope"]) == ""
    html = bd._render_dod([{"item": "real", "ok": True}, 7])
    assert "real" in html


def test_render_task_card_non_dict_input():
    # A non-dict task must not crash; it renders a fallback card.
    html = bd._render_task_card("not-a-dict")
    assert "task-card" in html
