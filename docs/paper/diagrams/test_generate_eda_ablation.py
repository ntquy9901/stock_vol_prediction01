"""Smoke test for the EDA-GNN E0->E1->E2->E3 ablation SVG generator.

Runs the real render (exercises every box/arrow) and asserts the SVG is non-empty
and carries the load-bearing, searchable labels + reported values of the actual
ablation, so a broken or drifted generator fails the gate rather than review.
Every asserted number traces to the ground-truth sources cited in
generate_eda_ablation.py (2026-08-11_1631_eda_gnn_results.md +
graph_recommendation.json).
"""

import pathlib
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import generate_eda_ablation as gen  # noqa: E402

# Task-required searchable labels: each names a real ablation element.
_REQUIRED_LABELS = (
    "E0", "E1", "E2", "E3",
    "MarketPK",
    "volume_zscore",
    "QLIKE",
    "HAR",
    "graph",
)

# Reported test-QLIKE values + DM-vs-HAR p-values that must render verbatim.
_REQUIRED_VALUES = (
    "0.5735", "0.5686", "0.5681", "0.5709",  # E0..E3 test QLIKE
    "0.5760", "0.5708",                       # E3off / G1corr comparison rungs
    "p=0.017", "p=0.012", "p=0.116", "p=0.044",  # DM verdicts
)


@pytest.mark.smoke
def test_build_returns_figure_with_ablation_labels() -> None:
    fig = gen.build()
    assert fig is not None
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    for label in _REQUIRED_LABELS:
        assert label in texts, f"missing ablation label: {label}"
    for value in _REQUIRED_VALUES:
        assert value in texts, f"missing reported value: {value}"


@pytest.mark.smoke
def test_main_writes_searchable_svg(tmp_path, monkeypatch) -> None:
    written = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def spy(self, path, *a, **k):
        written["path"] = str(path)
        return real_savefig(self, tmp_path / "out.svg", *a, **k)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    gen.main()

    assert written["path"].endswith("eda_ablation_E0_E3.svg")
    out = tmp_path / "out.svg"
    assert out.exists() and out.stat().st_size > 0
    svg = out.read_text(encoding="utf-8")
    assert "<svg" in svg
    # svg.fonttype='none' keeps labels as searchable <text> in the output.
    for label in ("E0", "E1", "E2", "E3", "MarketPK", "volume_zscore",
                  "QLIKE", "HAR", "graph"):
        assert label in svg, f"label not searchable in SVG output: {label}"
