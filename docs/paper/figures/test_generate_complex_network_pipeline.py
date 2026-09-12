"""Smoke test for the complex-network pipeline figure generator (docs artifact).

Exercises `main()` end to end with `Figure.savefig` monkeypatched to a no-op, so the drawing code
(every box/arrow, the `arr` txt / no-txt branches) runs without touching the tracked PNG/PDF. This is
the adjacent test the pre-push coverage gate discovers for `generate_complex_network_pipeline.py`.
"""
import importlib
import sys
from pathlib import Path

import matplotlib
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
gen = importlib.import_module("generate_complex_network_pipeline")


@pytest.mark.smoke
def test_main_runs_without_saving(monkeypatch):
    saved = []
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig",
                        lambda self, *a, **k: saved.append(a[0] if a else None))
    gen.main()
    # png + pdf writes attempted, both routed to the same base name
    assert len(saved) == 2
    assert all("fig_complex_network_pipeline" in str(p) for p in saved)


def test_arr_branches_and_box(monkeypatch):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    gen.box(ax, 0, 0, 1, 1, "t", "#eee")          # box path
    gen.arr(ax, 0, 0, 1, 1)                        # arr with txt=None (no label branch)
    gen.arr(ax, 0, 0, 1, 1, "label")              # arr with txt set (label branch)
    plt.close(fig)
