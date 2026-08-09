"""Smoke test for the PROPOSED-UPDATES architecture SVG generator.

Runs the drawing code (exercises every box/arrow) and asserts the key
proposed-update labels and their status tags are present, so a broken or
incomplete generator fails the gate rather than review. Each label traces to a
motivating report cited in generate_proposed_updates.py.
"""

import pathlib
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import generate_proposed_updates as gen  # noqa: E402

# The proposed-update labels the review page depends on. If any is dropped from
# the diagram the HTML table and the SVG fall out of sync -> fail here.
_REQUIRED_LABELS = (
    "k-NN top-8 sparse",          # graph adjacency option (DONE impl / PENDING DM)
    "presence mask",              # masked availability-aware message passing (DONE)
    "availability-aware",         # the news-on-graph novelty (DONE)
    "cache frozen-encoder",       # training speedup ~15x (PROPOSED)
    "encode present nodes only",  # training speedup ~2.1x (PROPOSED)
    "ReduceLROnPlateau",          # LR scheduler robustness (PROPOSED)
    "HARQ",                       # expanded baseline suite (PROPOSED)
    "log-HAR",                    # expanded baseline suite (PROPOSED)
    "GARCH",                      # expanded baseline suite (PROPOSED)
    "EWMA",                       # expanded baseline suite (PROPOSED)
    "persistence",                # naive floor baseline (PROPOSED)
    "Diebold-Mariano",            # significance test (PROPOSED)
    "Model Confidence Set",       # significance test (PROPOSED)
)

# The three status tags must all be visible so a reviewer can tell apart what is
# already built from what is only proposed.
_REQUIRED_TAGS = ("[DONE]", "[PENDING DM]", "[PROPOSED]")


@pytest.mark.smoke
def test_build_returns_figure_with_proposed_labels() -> None:
    fig = gen.build()
    assert fig is not None
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    for label in _REQUIRED_LABELS:
        assert label in texts, f"missing proposed-update label: {label}"
    for tag in _REQUIRED_TAGS:
        assert tag in texts, f"missing status tag: {tag}"


@pytest.mark.smoke
def test_main_writes_searchable_svg(tmp_path, monkeypatch) -> None:
    written = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def spy(self, path, *a, **k):
        written["path"] = str(path)
        return real_savefig(self, tmp_path / "out.svg", *a, **k)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    gen.main()

    assert written["path"].endswith("proposed_updates_architecture.svg")
    out = tmp_path / "out.svg"
    assert out.exists() and out.stat().st_size > 0
    svg = out.read_text(encoding="utf-8")
    assert "<svg" in svg
    # svg.fonttype='none' keeps labels as searchable <text> in the output.
    for label in ("k-NN top-8 sparse", "ReduceLROnPlateau", "Diebold-Mariano"):
        assert label in svg, f"label not searchable in SVG output: {label}"
