"""Tests for graph_spillover.construct_directed_spillover_graph (design.md §2.1)."""
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import numpy as np
import pytest

from graph_spillover import construct_directed_spillover_graph


def test_shape_is_num_stocks_square():
    rng = np.random.default_rng(0)
    vol = rng.random((30, 5))
    adj = construct_directed_spillover_graph(vol, k=2)
    assert adj.shape == (5, 5)


def test_asymmetric_on_lead_lag_data():
    """Stock 1 LEADS stock 0 (vol_0[t] = vol_1[t-1] + noise): edge 0<-1 (0 receives from 1)
    should be much stronger than edge 1<-0, and the matrix overall should NOT be symmetric —
    this is the whole point vs. the sibling's `construct_correlation_graph`, which always
    symmetrizes."""
    rng = np.random.default_rng(42)
    n = 200
    driver = rng.normal(size=n)
    vol = np.zeros((n, 3))
    vol[:, 1] = driver + rng.normal(scale=0.05, size=n)          # stock 1 = the leader
    vol[:, 0] = np.roll(driver, 1) + rng.normal(scale=0.05, size=n)  # stock 0 lags stock 1 by 1 day
    vol[:, 2] = rng.normal(size=n)                                # unrelated

    adj = construct_directed_spillover_graph(vol, k=2)

    assert adj[0, 1] > adj[1, 0], "stock 0 should receive strongly from leader stock 1"
    assert not np.allclose(adj, adj.T), "spillover graph must be directed (asymmetric)"


def test_no_self_loops():
    rng = np.random.default_rng(1)
    vol = rng.random((50, 6))
    adj = construct_directed_spillover_graph(vol, k=3)
    assert np.all(np.diag(adj) == 0)


def test_top_k_respected():
    rng = np.random.default_rng(2)
    vol = rng.random((100, 10))
    k = 3
    adj = construct_directed_spillover_graph(vol, k=k)
    for i in range(10):
        assert np.count_nonzero(adj[i]) <= k


@pytest.mark.parametrize("seq_length", [0, 1, 2])
def test_degenerate_short_window_returns_zero_graph(seq_length):
    vol = np.zeros((seq_length, 4))
    adj = construct_directed_spillover_graph(vol, k=2)
    assert adj.shape == (4, 4)
    assert np.all(adj == 0)


def test_zero_variance_stock_gets_no_incoming_edges():
    rng = np.random.default_rng(3)
    vol = rng.random((60, 4))
    vol[:, 2] = 5.0  # constant volatility -> zero variance -> can't correlate
    adj = construct_directed_spillover_graph(vol, k=2)
    assert np.all(adj[2, :] == 0), "constant-volatility receiver should have no incoming edges"


def test_weights_are_nonnegative():
    rng = np.random.default_rng(4)
    vol = rng.random((80, 8))
    adj = construct_directed_spillover_graph(vol, k=4)
    assert np.all(adj >= 0)
