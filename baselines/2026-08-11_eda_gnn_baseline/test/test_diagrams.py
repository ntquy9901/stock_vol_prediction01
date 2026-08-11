"""Smoke test: the architecture-diagram generator renders SVG + PNG without error."""

import importlib.util
from pathlib import Path

_DIAGRAMS = Path(__file__).resolve().parents[1] / "design" / "diagrams" / "generate_eda_gnn_diagrams.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_eda_gnn_diagrams", _DIAGRAMS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_diagrams_writes_svg_and_png(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "HERE", tmp_path)
    module.draw_ladder()
    module.draw_e3_detail()
    for name in ("eda_gnn_ladder", "eda_gnn_e3_detail"):
        for ext in ("svg", "png"):
            out = tmp_path / f"{name}.{ext}"
            assert out.exists() and out.stat().st_size > 0
