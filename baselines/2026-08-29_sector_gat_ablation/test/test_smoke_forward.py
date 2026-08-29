"""CPU-only smoke: sector adjacency builds for a small HNX slice, aligns to D.tickers, and
MaskedRichNet(use_graph=True) yields a finite output on a tiny batch. NO training loop, NO GPU.

Uses a tiny real-data slice (a handful of screened HNX tickers) per CLAUDE.md's real-data-sample
smoke rule, so it exercises the true processed-file writer + panel builder, not a synthetic fixture.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import run_sector_ablation as RSA  # noqa: E402
from config import SMOKE           # noqa: E402

PANEL = "hnx"


def _tiny_keep(n=8):
    keep = RSA.EFA.screened_tickers(PANEL)
    if keep is None:
        pytest.skip(f"{PANEL} panel not available")
    return set(sorted(keep)[:n])


@pytest.mark.smoke
def test_sector_adjacency_aligns_and_forward_pass_is_finite():
    keep = _tiny_keep(8)
    csv = RSA.default_sector_csv(PANEL)
    with tempfile.TemporaryDirectory() as td:
        D, files = RSA.build_panel_masked(PANEL, SMOKE, horizon=1, out_dir=td, keep_tickers=keep)
        adj, cov = RSA.sector_adj_for(D.tickers, csv)
        # adjacency aligned to node order
        assert adj.shape == (D.N, D.N)
        assert adj.dtype == np.float32
        assert np.array_equal(np.diag(adj), np.ones(D.N, dtype=np.float32))   # self-loops
        assert np.array_equal(adj, adj.T)                                     # undirected default
        assert cov["n_sectors"] >= 1
        # one CPU forward pass consumes the sector edge and stays finite
        out = RSA.forward_pass_smoke(D, adj, batch=2)
        assert out.shape[1] == D.N
        assert np.isfinite(out).all()


@pytest.mark.smoke
def test_full_hnx_sector_coverage_reasonable():
    """The full screened HNX universe maps to ICB sectors with high coverage (metadata sanity)."""
    keep = RSA.EFA.screened_tickers(PANEL)
    if keep is None:
        pytest.skip(f"{PANEL} panel not available")
    _, cov = RSA.sector_adj_for(sorted(keep), RSA.default_sector_csv(PANEL))
    assert cov["coverage_frac"] >= 0.9
    assert cov["n_sectors"] >= 8              # HNX spans many ICB industries
    assert cov["avg_off_degree"] > 0          # sectors actually create edges
