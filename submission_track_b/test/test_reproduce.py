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
    assert set(metrics) == {"P0", "P1", "P2", "P3", "G0", "G1"}
    for config, values in metrics.items():
        assert set(values) == set(reproduce._METRICS), config
        for value in values.values():
            assert isinstance(value, float)


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
