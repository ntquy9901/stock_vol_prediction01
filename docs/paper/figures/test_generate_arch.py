"""Coverage for the VolTree architecture-figure generator: the box/arrow/glyph helpers add patches, and
main() renders the embedded PNG/PDF without error. generate_arch.py is now the authoritative source for
fig_architecture.{png,pdf} (no draw.io CLI available), so this test keeps it inside the pre-push gate."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_arch as GA  # noqa: E402


def test_box_and_arr_add_patches():
    fig, ax = plt.subplots()
    n0 = len(ax.patches)
    GA.box(ax, 0, 0, 1, 1, "t", "#ffffff")
    GA.arr(ax, 0, 0, 1, 1, "lbl")     # with label -> exercises the txt branch
    GA.arr(ax, 0, 0, 1, 1)            # without label -> exercises the no-txt branch
    assert len(ax.patches) > n0                       # box + arrow patches added
    plt.close(fig)


def test_graph_glyph_adds_nodes():
    fig, ax = plt.subplots()
    n0 = len(ax.patches)
    GA.graph_glyph(ax, 3.0, 2.0)
    assert len(ax.patches) == n0 + 6                  # 4 low-vol + 2 high-vol nodes
    plt.close(fig)


def test_main_writes_nonempty_png_and_pdf(monkeypatch):
    # main() writes the embedded figure fig_architecture.{png,pdf} (this script is the single source).
    root = Path(__file__).resolve().parents[3]        # repo root; main() saves to docs/paper/figures/<...>
    monkeypatch.chdir(root)
    GA.main()
    for ext in ("png", "pdf"):
        p = root / "docs" / "paper" / "figures" / f"fig_architecture.{ext}"
        assert p.exists() and p.stat().st_size > 10000


def test_architecture_layout_is_portrait(monkeypatch):
    import matplotlib.image as mpimg
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    GA.main()
    img = mpimg.imread(root / "docs" / "paper" / "figures" / "fig_architecture.png")
    height, width = img.shape[0], img.shape[1]
    assert height > width                              # vertical top-to-bottom layout is taller than wide


def test_voltree_only_no_variant_summary():
    # Advisor: the figure describes the full VolTree model only; the ablation variants live in the text.
    import inspect
    src = inspect.getsource(GA)
    assert "VolTree" in src
    assert "optional" not in src.lower()               # every stage is mandatory
    assert "XGB+E" not in src and "XGB+LG" not in src  # no ablation-variant summary in the figure
