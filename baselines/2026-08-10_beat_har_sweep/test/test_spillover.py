"""Tests for the directed Diebold-Yilmaz spillover adjacency (C3/C5)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from spillover import directed_spillover_adjacency  # noqa: E402


def test_shape_finite_rows_normalized():
    rng = np.random.default_rng(1)
    panel = rng.normal(size=(400, 5))
    adjacency = directed_spillover_adjacency(panel, var_lag=1, fevd_horizon=10)
    assert adjacency.shape == (5, 5)
    assert np.isfinite(adjacency).all()
    assert (adjacency >= 0).all()
    assert np.allclose(adjacency.sum(axis=1), 1.0, atol=1e-8)  # row-normalized shares


def test_directionality_lead_lag():
    """If series 1 leads series 0 (y0[t] = 0.8*y1[t-1]), then 0's error variance is explained
    more by 1 than vice-versa: adjacency[0,1] > adjacency[1,0]."""

    rng = np.random.default_rng(2)
    n = 1200
    y1 = np.cumsum(rng.normal(scale=0.1, size=n)) + rng.normal(scale=0.05, size=n)
    y0 = np.empty(n)
    y0[0] = 0.0
    y0[1:] = 0.8 * y1[:-1] + rng.normal(scale=0.02, size=n - 1)
    panel = np.column_stack([y0, y1])
    adjacency = directed_spillover_adjacency(panel, var_lag=1, fevd_horizon=10)
    assert adjacency[0, 1] > adjacency[1, 0]


def test_asymmetric_in_general():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(500, 4))
    y[1:, 0] += 0.5 * y[:-1, 3]  # 3 -> 0
    adjacency = directed_spillover_adjacency(y, var_lag=1, fevd_horizon=8)
    assert not np.allclose(adjacency, adjacency.T)


def test_rejects_degenerate_panel():
    with pytest.raises(ValueError):
        directed_spillover_adjacency(np.ones((10, 1)), var_lag=1, fevd_horizon=5)
    with pytest.raises(ValueError):
        directed_spillover_adjacency(np.zeros((3, 4)), var_lag=1, fevd_horizon=5)
