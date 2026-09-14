"""Coverage for the architecture-figure generator: the box/arrow helpers add patches, and main() renders
the PNG/PDF without error. Keeps the figure script inside the pre-push gate (a changed .py needs a test)."""
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
    GA.arr(ax, 0, 0, 1, 1, "lbl")
    assert len(ax.patches) > n0                       # box + arrow patches added
    plt.close(fig)


def test_main_writes_nonempty_png_and_pdf(monkeypatch):
    root = Path(__file__).resolve().parents[3]        # repo root; main() saves to docs/paper/figures/<...>
    monkeypatch.chdir(root)
    GA.main()
    for ext in ("png", "pdf"):
        p = root / "docs" / "paper" / "figures" / f"fig_architecture.{ext}"
        assert p.exists() and p.stat().st_size > 10000
