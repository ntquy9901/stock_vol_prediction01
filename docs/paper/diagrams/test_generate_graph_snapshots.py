"""Smoke test for the masked graph-snapshot SVG generator (rendering path only).

The heavy real-data manifest build is exercised separately when the figure is
regenerated; this test drives the pure rendering function on a tiny synthetic
panel so it runs fast and still asserts the SVG carries searchable node/mode
labels (svg.fonttype='none').
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_graph_snapshots import render_graph_figure  # noqa: E402


def _tiny_panel() -> dict:
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    present = [0, 1, 2]  # DDD absent
    dense = np.zeros((4, 4), dtype=float)
    for i in present:
        dense[i, i] = 1.0
    dense[0, 1] = dense[1, 0] = 0.8
    dense[0, 2] = dense[2, 0] = -0.5
    dense[1, 2] = dense[2, 1] = 0.3
    knn = dense.copy()
    knn[1, 2] = knn[2, 1] = 0.0  # sparsified
    return {
        "date": "2020-01-02",
        "split": "train",
        "tickers_all": tickers,
        "present_ids": present,
        "adjacency_dense": dense,
        "adjacency_knn": knn,
    }


@pytest.mark.smoke
def test_render_produces_svg_with_labels(tmp_path: Path) -> None:
    out = tmp_path / "snap.svg"
    render_graph_figure([_tiny_panel()], out)
    assert out.exists() and out.stat().st_size > 0
    text = out.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "AAA" in text  # a present ticker label survives as searchable text
    assert "dense" in text  # dense mode label
    assert "k-NN" in text  # sparse mode label


@pytest.mark.smoke
def test_render_rejects_empty_panels(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        render_graph_figure([], tmp_path / "empty.svg")
