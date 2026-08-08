"""Smoke test for the final-combined-architecture SVG generator.

Runs the drawing code (exercises every box/arrow) and asserts the key
architecture labels are present, so a broken generator fails CI, not review.
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import generate_final_combined as gen  # noqa: E402


def test_build_returns_figure_with_key_labels() -> None:
    fig = gen.build()
    assert fig is not None
    texts = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    for label in ("P0: HAR", "presence_mask", "Positivity floor", "LEAKAGE BARRIER"):
        assert label in texts, f"missing architecture label: {label}"


def test_main_writes_svg(tmp_path, monkeypatch) -> None:
    written = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def spy(self, path, *a, **k):
        written["path"] = str(path)
        return real_savefig(self, tmp_path / "out.svg", *a, **k)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    gen.main()
    assert written["path"].endswith("final_combined_architecture.svg")
    assert (tmp_path / "out.svg").exists()
