"""Tests for the Track-B reproducibility entry point.

The `view` path must work with NO dataset and NO training, using only the bundled
result JSONs. These tests exercise that path end-to-end plus the g1_final guards
that do not require the dataset.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_BUNDLE = Path(__file__).resolve().parents[1]
_CODE = _BUNDLE / "trackb_code"
for _path in (str(_BUNDLE), str(_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load_reproduce():
    spec = importlib.util.spec_from_file_location("reproduce", _BUNDLE / "reproduce.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


reproduce = _load_reproduce()


def test_fmt_metric_formatting():
    assert reproduce._fmt("rmse", 0.00148) == "1.480e-03"
    assert reproduce._fmt("r2", 0.7351) == "0.7351"
    assert reproduce._fmt("directional_accuracy", 48.54) == "48.54"
    assert reproduce._fmt("qlike", None) == "n/a"


def test_load_validation_metrics_has_all_configs_and_metrics():
    metrics = reproduce._load_validation_metrics()
    assert set(metrics) == {"P0", "P1", "P2", "P3", "G1"}
    for config, values in metrics.items():
        assert set(values) == set(reproduce._METRICS), config
        for value in values.values():
            assert isinstance(value, float)


def test_no_separate_g0_row():
    # The ladder collapsed G0: P3 is exactly G1 with the graph disabled.
    assert "G0" not in dict((c, r) for c, _, r in reproduce._LADDER)
    assert [c for c, _, _ in reproduce._LADDER] == ["P0", "P1", "P2", "P3", "G1"]


def test_classical_baselines_loaded():
    classical = reproduce._load_classical_test()
    assert {"HAR", "HARQ", "GARCH"}.issubset(set(classical))
    # HAR/HARQ tie the deep models near ~0.57-0.58; GARCH is far worse (>1).
    assert 0.5 < classical["HAR"]["qlike"] < 0.65
    assert classical["GARCH"]["qlike"] > 1.0


def test_g1_test_matches_ladder_json():
    # Full-precision consistency: view's G1 test QLIKE must equal the canonical ladder JSON.
    g1 = reproduce._load_g1_test_metrics()
    assert abs(g1["qlike"] - 0.5759260895226245) < 1e-12
    assert abs(g1["rmse"] - 0.0023052718907185933) < 1e-15


def test_table_lines_label_g1_as_final():
    val = reproduce._load_validation_metrics()
    lines = reproduce._table_lines(val, None)
    text = "\n".join(lines)
    assert "G1" in text
    assert "FINAL / PROPOSED" in text
    # every ladder config appears as a row
    for config, _, _ in reproduce._LADDER:
        assert any(line.startswith(config) for line in lines)


@pytest.mark.smoke
def test_cmd_view_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduce, "_OUTPUT", tmp_path / "output")
    assert reproduce.cmd_view() == 0
    md = tmp_path / "output" / "results_table.md"
    png = tmp_path / "output" / "summary.png"
    assert md.exists() and md.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0
    body = md.read_text(encoding="utf-8")
    assert "P0" in body and "G1" in body and "FINAL / PROPOSED" in body


def test_g1_test_row_from_canonical_json():
    # The bundle ships the paper's held-out test 3-seed mean; view shows it with no data.
    test_metrics = reproduce._load_g1_test_metrics()
    assert test_metrics is not None
    assert set(test_metrics) == set(reproduce._METRICS)
    lines = reproduce._table_lines(reproduce._load_validation_metrics(), test_metrics)
    assert any("TEST (paper 3-seed)" in line for line in lines)


def test_parsimony_note_present():
    note = reproduce._graph_significance_note()
    assert "parsimony" in note.lower() or "significant" in note.lower()
    lines = reproduce._table_lines(reproduce._load_validation_metrics(), None)
    assert any("Parsimony finding" in line for line in lines)


def test_infer_missing_checkpoint_raises(tmp_path):
    import g1_final
    with pytest.raises(FileNotFoundError):
        g1_final.infer_g1(tmp_path / "does_not_exist.pt")


def test_g1_final_imports():
    import g1_final
    assert hasattr(g1_final, "train_g1")
    assert hasattr(g1_final, "infer_g1")
    assert hasattr(g1_final, "build_graph_context")


def test_cmd_infer_missing_checkpoint_returns_one(tmp_path, monkeypatch):
    # The offline guard: no checkpoint -> print guidance, return 1 (no data needed).
    monkeypatch.setattr(reproduce, "_CHECKPOINTS", tmp_path / "checkpoints")
    assert reproduce.cmd_infer() == 1


def test_main_view_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduce, "_OUTPUT", tmp_path / "output")
    assert reproduce.main(["view"]) == 0
    assert reproduce.main(["results"]) == 0


def test_main_infer_dispatch_no_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduce, "_CHECKPOINTS", tmp_path / "checkpoints")
    assert reproduce.main(["infer"]) == 1


def test_main_unrecognised_returns_two():
    assert reproduce.main(["bogus"]) == 2


def test_main_empty_enters_menu_and_quits(monkeypatch):
    # main([]) -> _menu(); feed "0" to quit immediately.
    monkeypatch.setattr("builtins.input", lambda *_: "0")
    assert reproduce.main([]) == 0


def test_menu_view_then_quit(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduce, "_OUTPUT", tmp_path / "output")
    responses = iter(["1", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(responses))
    assert reproduce._menu() == 0


def test_menu_unrecognised_then_quit(monkeypatch):
    responses = iter(["99", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(responses))
    assert reproduce._menu() == 0


def test_menu_eof_returns_zero(monkeypatch):
    def _raise_eof(*_):
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    assert reproduce._menu() == 0
