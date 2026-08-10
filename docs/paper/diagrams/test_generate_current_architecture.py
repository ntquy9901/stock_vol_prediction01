"""Smoke test for the CURRENT G1 (Track B) architecture SVG generator.

Runs the real render (exercises every box/arrow) and asserts the SVG is non-empty
and carries the load-bearing, searchable labels of the actual architecture, so a
broken or drifted generator fails the gate rather than review. Every asserted label
traces to the ground-truth sources cited in generate_current_architecture.py.
"""

import pathlib
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import generate_current_architecture as gen  # noqa: E402

# Labels that must stay in the diagram: each names a real architecture element.
_REQUIRED_LABELS = (
    "HAR",                 # the 3 HAR node features (task-required label)
    "k-NN",                # masked k-NN-8 correlation adjacency (task-required label)
    "news",                # PhoBERT news branch (task-required label)
    "G1",                  # the proposed model / headline (task-required label)
    "P3",                  # graph-off nested rung (task-required label)
    "P0",                  # HAR rung of the ladder
    "PhoBERT",             # news encoder input, dual-group 146-dim
    "gate",                # per-ticker sigmoid gate
    "message passing",     # the graph layer operation (NOT multi-head attention)
    "Parkinson",           # target = 5-day-ahead Parkinson variance
    "present-node",        # present-node masking of the graph
    "parsimony",           # the honest null-graph finding note
)


@pytest.mark.smoke
def test_build_returns_figure_with_architecture_labels() -> None:
    fig = gen.build()
    assert fig is not None
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    for label in _REQUIRED_LABELS:
        assert label in texts, f"missing architecture label: {label}"


@pytest.mark.smoke
def test_main_writes_searchable_svg(tmp_path, monkeypatch) -> None:
    written = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def spy(self, path, *a, **k):
        written["path"] = str(path)
        return real_savefig(self, tmp_path / "out.svg", *a, **k)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    gen.main()

    assert written["path"].endswith("current_g1_architecture.svg")
    out = tmp_path / "out.svg"
    assert out.exists() and out.stat().st_size > 0
    svg = out.read_text(encoding="utf-8")
    assert "<svg" in svg
    # svg.fonttype='none' keeps labels as searchable <text> in the output.
    for label in ("HAR", "k-NN", "news", "G1", "P3"):
        assert label in svg, f"label not searchable in SVG output: {label}"
