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
