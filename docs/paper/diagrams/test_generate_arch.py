"""Smoke + label-correctness test for the architecture-figure generator.

Guards against regressing the figure to the old (wrong) description: the paper model uses a directed
volume->Parkinson weighted graph and five node features, NOT a graphical-lasso / partial-correlation graph.
"""
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent


def test_generate_arch_writes_all_formats():
    runpy.run_path(str(HERE / "generate_arch.py"), run_name="__main__")
    for ext in ("svg", "pdf", "png"):
        f = HERE / f"soict_harlstmgat.{ext}"
        assert f.exists() and f.stat().st_size > 1000, f"{ext} missing or too small"


def test_figure_labels_match_paper_model():
    runpy.run_path(str(HERE / "generate_arch.py"), run_name="__main__")
    svg = (HERE / "soict_harlstmgat.svg").read_text(encoding="utf-8")
    # correct model description present
    assert "5 node features" in svg
    assert ("vol-&gt;Parkinson" in svg) or ("vol->Parkinson" in svg)  # ">" is XML-escaped in SVG
    assert "2-hop" in svg
    # stale/wrong graph description absent
    assert "graphical-lasso" not in svg
    assert "partial-corr" not in svg
