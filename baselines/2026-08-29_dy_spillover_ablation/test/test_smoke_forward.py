"""CPU-only smoke: the DY (2014) connectedness adjacency builds for a small HNX slice, aligns to
D.tickers, has the self-loop/Top-K convention, and MaskedRichNet(use_graph=True) yields a finite output.
NO training loop, NO GPU. Uses a tiny real-data slice per CLAUDE.md's real-data-sample smoke rule.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import run_dy_ablation as RDA  # noqa: E402
from config import SMOKE       # noqa: E402

PANEL = "hnx"


def _tiny_keep(n=8):
    keep = RDA.EFA.screened_tickers(PANEL)
    if keep is None:
        pytest.skip(f"{PANEL} panel not available")
    return set(sorted(keep)[:n])


@pytest.mark.smoke
def test_dy_adjacency_aligns_and_forward_pass_is_finite():
    keep = _tiny_keep(8)
    with tempfile.TemporaryDirectory() as td:
        D, files = RDA.build_panel_masked(PANEL, SMOKE, horizon=1, out_dir=td, keep_tickers=keep)
        adj, stats = RDA.dy_adj_for(D, files, p=1, H=10, top_k=5)
        assert adj.shape == (D.N, D.N)
        assert adj.dtype == np.float32
        assert np.array_equal(np.diag(adj), np.ones(D.N, dtype=np.float32))   # self-loops
        assert abs(stats["row_sum_mean"] - 1.0) < 1e-5                        # DY normalisation
        assert stats["total_connectedness_index"] >= 0.0
        off = adj.copy(); np.fill_diagonal(off, 0.0)
        assert (off > 0).sum(axis=1).max() <= 5                               # Top-K per row
        out = RDA.forward_pass_smoke(D, adj, batch=2)
        assert out.shape[1] == D.N
        assert np.isfinite(out).all()
