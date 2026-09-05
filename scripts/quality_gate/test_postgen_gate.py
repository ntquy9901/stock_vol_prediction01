"""Unit tests for the post-generate hook gate (pure logic: scope filter + decision assembly)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

import postgen_gate as pg  # noqa: E402


def test_rel_in_scope_accepts_repo_py(tmp_path):
    f = REPO / "scripts" / "quality_gate" / "postgen_gate.py"   # a real in-scope .py
    assert pg.rel_in_scope(str(f)) == "scripts/quality_gate/postgen_gate.py"


def test_rel_in_scope_rejects_non_py():
    assert pg.rel_in_scope(str(REPO / "README.md")) is None


def test_rel_in_scope_rejects_excluded_trees():
    # a path under an excluded tree (need not exist — the prefix check rejects it first)
    assert pg.rel_in_scope(str(REPO / "archive" / "x.py")) is None
    assert pg.rel_in_scope(str(REPO / "data" / "x.py")) is None


def test_rel_in_scope_rejects_outside_repo(tmp_path):
    outside = tmp_path / "stray.py"
    outside.write_text("x = 1\n")
    assert pg.rel_in_scope(str(outside)) is None


def test_rel_in_scope_rejects_missing_file():
    assert pg.rel_in_scope(str(REPO / "scripts" / "does_not_exist_xyz.py")) is None


def test_is_test_file():
    assert pg.is_test_file("scripts/quality_gate/test_postgen_gate.py")
    assert pg.is_test_file("baselines/x/test/thing.py")
    assert not pg.is_test_file("baselines/x/code/run.py")


def test_build_decision_none_when_clean():
    assert pg.build_decision("x/run.py", None, None) is None


def test_build_decision_blocks_on_ruff():
    d = pg.build_decision("x/run.py", "F401 unused import", None)
    assert d["decision"] == "block"
    assert "ruff pyflakes" in d["reason"]
    assert "x/run.py" in d["reason"]


def test_build_decision_blocks_on_hardcode():
    d = pg.build_decision("x/run.py", None, "    x/run.py:5: lookback=22 hardcoded")
    assert d["decision"] == "block"
    assert "config-hardcode" in d["reason"]


def test_build_decision_combines_both():
    d = pg.build_decision("x/run.py", "F811 redef", "    x/run.py:5: top_k=5 hardcoded")
    assert d["decision"] == "block"
    assert "ruff pyflakes" in d["reason"] and "config-hardcode" in d["reason"]


def test_file_from_hook_json_tool_input():
    import io
    s = io.StringIO('{"tool_name":"Edit","tool_input":{"file_path":"/a/b.py"}}')
    assert pg.file_from_hook_json(s) == "/a/b.py"


def test_file_from_hook_json_tool_response_fallback():
    import io
    s = io.StringIO('{"tool_response":{"filePath":"/c/d.py"}}')
    assert pg.file_from_hook_json(s) == "/c/d.py"


def test_file_from_hook_json_malformed_returns_none():
    import io
    assert pg.file_from_hook_json(io.StringIO("not json")) is None
    assert pg.file_from_hook_json(io.StringIO('{"tool_input":{}}')) is None
