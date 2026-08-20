"""TDD tests for edges.glasso_adjacency — TRAIN-frozen graphical-LASSO partial-corr Top-K graph."""
from __future__ import annotations

import numpy as np
import pandas as pd

import edges


def _block_panel(t: int = 400, seed: int = 0) -> pd.DataFrame:
    """Two clean clusters of 3 tickers each (shared latent factor + idiosyncratic noise).

    Within a cluster the columns share a factor -> strong positive partial correlation; across
    clusters partial correlation is ~0. With top_k=2 and cluster size 3, each node's Top-2 |pcorr|
    off-diagonal neighbours are exactly its two cluster-mates (mutual), so the undirected Top-K
    graph is a pair of 3-cliques and every row has exactly top_k+1 nonzeros.
    """
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal(t)
    f2 = rng.standard_normal(t)
    cols = {}
    for i in range(3):
        cols[f"A{i}"] = f1 + 0.3 * rng.standard_normal(t)
    for i in range(3):
        cols[f"B{i}"] = f2 + 0.3 * rng.standard_normal(t)
    return pd.DataFrame(cols)


def test_shape_diagonal_and_sparsity():
    panel = _block_panel()
    top_k = 2
    adj = edges.glasso_adjacency(panel, top_k=top_k)

    n = panel.shape[1]
    assert adj.shape == (n, n)
    assert adj.dtype == np.float32
    # diagonal (self-loops) all 1.0
    assert np.allclose(np.diag(adj), 1.0)
    # every node keeps at most top_k off-diagonal edges (+1 self-loop)
    for i in range(n):
        assert np.count_nonzero(adj[i]) <= top_k + 1
    # symmetric (undirected partial-corr graph)
    assert np.allclose(adj, adj.T)


def test_determinism():
    panel = _block_panel()
    a1 = edges.glasso_adjacency(panel, top_k=3)
    a2 = edges.glasso_adjacency(panel, top_k=3)
    assert np.allclose(a1, a2)


def test_non_convergence_returns_finite():
    """Near-singular / collinear panel must still yield a finite adjacency (no NaN/inf)."""
    rng = np.random.default_rng(1)
    t = 300
    base = rng.standard_normal(t)
    panel = pd.DataFrame(
        {
            "a": base,
            "b": base,                                   # exact duplicate -> singular covariance
            "c": base + 1e-12 * rng.standard_normal(t),  # near-duplicate
            "d": rng.standard_normal(t),
        }
    )
    adj = edges.glasso_adjacency(panel, top_k=2, alpha=0.01)
    assert adj.shape == (4, 4)
    assert np.all(np.isfinite(adj))
    assert np.allclose(np.diag(adj), 1.0)
