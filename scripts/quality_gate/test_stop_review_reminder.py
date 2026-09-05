"""Unit tests for the Stop-hook code-review reminder."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stop_review_reminder as sr  # noqa: E402


def test_changed_py_sources_selects_source_only():
    porcelain = "\n".join([
        " M baselines/x/code/run.py",       # source -> keep
        "?? scripts/quality_gate/new.py",   # untracked source -> keep
        " M baselines/x/test/test_run.py",  # test dir -> skip
        "?? scripts/test_helper.py",        # test_ prefix -> skip
        " M archive/old.py",                # excluded tree -> skip
        " M data/x.py",                     # excluded tree -> skip
        " M docs/readme.md",                # not .py -> skip
    ])
    assert sr.changed_py_sources(porcelain) == [
        "baselines/x/code/run.py", "scripts/quality_gate/new.py"]


def test_changed_py_sources_empty():
    assert sr.changed_py_sources("") == []
    assert sr.changed_py_sources(" M docs/a.md\n?? b.txt") == []


def test_reminder_none_when_clean():
    assert sr.reminder([]) is None


def test_reminder_lists_paths():
    msg = sr.reminder(["a/b.py", "c/d.py"])
    assert "systemMessage" in msg
    assert "a/b.py" in msg["systemMessage"] and "c/d.py" in msg["systemMessage"]
    assert "/code-review" in msg["systemMessage"]
