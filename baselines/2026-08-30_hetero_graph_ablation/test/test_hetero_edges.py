"""Unit tests for the two per-relation adjacencies + per-relation Min-Max normalization (hetero_edges.py).

Independently verifies (per the task): (a) the per-relation Min-Max maps each relation's weights into [0,1]
with TRAIN-ONLY min/max, (c) edge counts match the thresholds on a fixture. UNIQUE basename
(test_hetero_edges.py) to avoid the pytest prepend-import duplicate-basename shadowing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import hetero_edges as HE  # noqa: E402


# ------------------------- fit_minmax / apply_minmax (per-relation Min-Max) -------------------------

def test_fit_minmax_empty_is_noop():
    assert HE.fit_minmax(np.array([])) == (0.0, 1.0)


def test_fit_minmax_returns_train_min_max():
    w = np.array([0.3, 0.9, 0.5, 0.7])
    assert HE.fit_minmax(w) == (0.3, 0.9)


def test_apply_minmax_maps_into_unit_interval_independent_recompute():
    w = np.array([0.25, 0.5, 0.75, 1.0])
    lo, hi = HE.fit_minmax(w)
    out = HE.apply_minmax(w, lo, hi)
    # INDEPENDENT recompute of Min-Max (w - lo)/(hi - lo), then the eps lower-clip
    expected = np.clip((w - lo) / (hi - lo), HE.EPS, 1.0)
    assert np.allclose(out, expected)
    assert (out >= 0.0).all() and (out <= 1.0).all()      # into [0,1]
    assert out.max() == pytest.approx(1.0)                 # the largest edge maps to 1
    assert out.min() == pytest.approx(HE.EPS)              # the smallest edge is eps-clipped (stays present)
    assert out.dtype == np.float32


def test_apply_minmax_degenerate_all_equal_maps_to_one():
    w = np.array([0.4, 0.4, 0.4])
    out = HE.apply_minmax(w, 0.4, 0.4)                      # hi <= lo -> flat relation
    assert np.allclose(out, 1.0)


# ------------------------- _relation_adjacency (hand-built fires + weights) -------------------------

def test_relation_adjacency_counts_selfloop_symmetry_and_normalization():
    # 4 nodes; fire edges (0,1) w=0.3 and (2,3) w=0.9 (symmetric); diagonal fires ignored
    n = 4
    fires = np.zeros((n, n), dtype=bool)
    raw = np.zeros((n, n))
    for i, j, w in [(0, 1, 0.3), (2, 3, 0.9)]:
        fires[i, j] = fires[j, i] = True
        raw[i, j] = raw[j, i] = w
    np.fill_diagonal(fires, True)                          # diagonal fires MUST be dropped by the builder
    adj, stats = HE._relation_adjacency(fires, raw)
    assert adj.shape == (n, n) and adj.dtype == np.float32
    assert np.allclose(np.diag(adj), 1.0)                  # self-loop
    assert np.allclose(adj, adj.T)                         # symmetric
    assert stats["n_edges"] == 2                           # off-diagonal fired pairs only
    # Min-Max over fired {0.3,0.9}: 0.3 -> eps, 0.9 -> 1.0
    assert adj[2, 3] == pytest.approx(1.0)
    assert adj[0, 1] == pytest.approx(HE.EPS)
    assert stats["minmax"] == [0.3, 0.9]
    assert stats["max_off_degree"] == 1
    assert stats["n_singletons"] == 0
    assert (adj[np.triu_indices(n, 1)] >= 0).all() and (adj <= 1.0 + 1e-6).all()


def test_relation_adjacency_empty_is_all_selfloops():
    n = 3
    fires = np.zeros((n, n), dtype=bool)
    adj, stats = HE._relation_adjacency(fires, np.zeros((n, n)))
    assert stats["n_edges"] == 0
    assert stats["n_singletons"] == 3
    off = adj.copy(); np.fill_diagonal(off, 0.0)
    assert (off == 0).all() and np.allclose(np.diag(adj), 1.0)


# ------------------------- build_relation_adjacencies (returns fixture) -------------------------

def _returns_wide(seed=1, n_days=400):
    """4-ticker close panel: A,B highly correlated + co-moving; C,D independent noise."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.02, size=n_days)
    ra = base + rng.normal(0, 0.001, size=n_days)
    rb = base + rng.normal(0, 0.001, size=n_days)
    rc = rng.normal(0, 0.02, size=n_days)
    rd = rng.normal(0, 0.02, size=n_days)
    idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
    px = lambda r: 10.0 * np.cumprod(1 + r)
    return pd.DataFrame({"A": px(ra), "B": px(rb), "C": px(rc), "D": px(rd)}, index=idx)


def test_build_relation_adjacencies_shapes_selfloop_symmetry():
    wide = _returns_wide()
    cutoff = wide.index[-1] + pd.Timedelta(days=1)          # all rows train
    adj_lin, adj_nl, diag = HE.build_relation_adjacencies(wide, cutoff)
    for adj in (adj_lin, adj_nl):
        assert adj.shape == (4, 4) and adj.dtype == np.float32
        assert np.allclose(np.diag(adj), 1.0)
        assert np.allclose(adj, adj.T)
        assert np.isfinite(adj).all()
        off = adj.copy(); np.fill_diagonal(off, 0.0)
        assert (off >= 0).all() and (off <= 1.0 + 1e-6).all()   # per-relation weights in [0,1]
    assert diag["n_nodes"] == 4 and diag["n_pairs"] == 6
    assert {"linear_corr", "nonlinear_assoc"} <= set(diag)
    assert adj_lin[0, 1] > 0                                # A-B correlation edge present
    assert diag["linear_corr"]["thresh"] == HE.CORR_THRESH
    assert diag["nonlinear_assoc"]["thresh"] == HE.LIFT_THRESH


def test_edge_counts_match_thresholds_on_fixture():
    """(c) edge counts match the thresholds: independently recount |rho|>thresh / lift>thresh from the corr /
    lift primitives and assert the diag agrees with the built adjacencies (nonzero off-diagonal)."""
    import corrlift_edge as CL
    wide = _returns_wide(seed=3)
    cutoff = wide.index[-1] + pd.Timedelta(days=1)
    returns = CL.daily_returns(wide)
    corr = CL.pearson_corr(returns, HE.MIN_OVERLAP)
    event, valid = CL.move_events(returns)
    lift = CL.pairwise_lift(event, valid, HE.MIN_PAIRS)
    iu = np.triu_indices(4, 1)
    exp_lin = int((np.isfinite(corr[iu]) & (np.abs(corr[iu]) > HE.CORR_THRESH)).sum())
    exp_nl = int((np.isfinite(lift[iu]) & (lift[iu] > HE.LIFT_THRESH)).sum())
    adj_lin, adj_nl, diag = HE.build_relation_adjacencies(wide, cutoff)
    assert diag["linear_corr"]["n_edges"] == exp_lin
    assert diag["nonlinear_assoc"]["n_edges"] == exp_nl
    # off-diagonal nonzero count (each undirected edge = 2 directed entries) matches 2*n_edges
    assert int((adj_lin != 0).sum()) - 4 == 2 * exp_lin
    assert int((adj_nl != 0).sum()) - 4 == 2 * exp_nl


def test_build_relation_adjacencies_train_only_leakage_frozen():
    """(a) train-only min/max: poisoning POST-cutoff rows must NOT change either frozen adjacency (both the
    thresholded edges AND the Min-Max min/max are computed on train rows only)."""
    wide = _returns_wide(seed=5)
    cutoff = wide.index[len(wide) // 2]
    lin_ref, nl_ref, _ = HE.build_relation_adjacencies(wide, cutoff)
    poisoned = wide.copy()
    poisoned.loc[poisoned.index >= cutoff] = 1e6
    lin_p, nl_p, _ = HE.build_relation_adjacencies(poisoned, cutoff)
    assert np.array_equal(lin_ref, lin_p)
    assert np.array_equal(nl_ref, nl_p)


def test_build_relation_adjacencies_high_thresholds_empty():
    wide = _returns_wide()
    cutoff = wide.index[-1] + pd.Timedelta(days=1)
    adj_lin, adj_nl, diag = HE.build_relation_adjacencies(wide, cutoff, corr_thresh=0.999, lift_thresh=100.0)
    assert diag["linear_corr"]["n_edges"] == 0
    assert diag["nonlinear_assoc"]["n_edges"] == 0
    for adj in (adj_lin, adj_nl):
        off = adj.copy(); np.fill_diagonal(off, 0.0)
        assert (off == 0).all()


def test_load_close_wide_reexport_identity():
    import corrlift_edge as CL
    assert HE.load_close_wide is CL.load_close_wide
