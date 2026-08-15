"""Smoke test for the trackA architecture/ablation SVG generator.

Runs both diagram functions into a tmp dir and asserts valid, non-empty SVGs are produced.
Skips if matplotlib is unavailable in the running interpreter.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import generate_trackA_diagrams as g  # noqa: E402


def test_diagrams_generate_valid_svgs(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "OUT_DIR", tmp_path)
    g.diagram_full_model()
    g.diagram_ablation()
    for name in ("trackA_gat_architecture.svg", "trackA_gat_ablation.svg"):
        svg = tmp_path / name
        assert svg.exists() and svg.stat().st_size > 1000
        assert "<svg" in svg.read_text(encoding="utf-8")[:2000]
