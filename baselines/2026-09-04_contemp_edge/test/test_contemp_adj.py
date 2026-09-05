"""Unit tests for the contemporaneous-edge builder (run_contemp.build_contemp_adj / _fold_adj).

Covers the only new pure logic in this probe baseline; run() itself is a GPU training driver
(marked no-cover) exercised by the --smoke path.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

import run_contemp as RC  # noqa: E402


def _make_sqrt_pk(n_days=40, n_nodes=6, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_days, 1))
    # nodes 1,2 strongly co-move with node 0; nodes 3,4,5 are independent noise
    x = np.zeros((n_days, n_nodes), dtype=np.float64)
    x[:, 0] = base[:, 0]
    x[:, 1] = base[:, 0] + 0.05 * rng.standard_normal(n_days)
    x[:, 2] = base[:, 0] + 0.05 * rng.standard_normal(n_days)
    x[:, 3] = rng.standard_normal(n_days)
    x[:, 4] = rng.standard_normal(n_days)
    x[:, 5] = rng.standard_normal(n_days)
    return np.abs(x) + 0.1  # sqrt-vol is non-negative


def test_shape_and_self_loop():
    sqrt_pk = _make_sqrt_pk()
    A = RC.build_contemp_adj(sqrt_pk, last_row=30, top_k=3)
    assert A.shape == (6, 6)
    assert A.dtype == np.float32
    assert np.allclose(np.diag(A), 1.0)  # self-loop = 1 on every node


def test_topk_count_per_target():
    sqrt_pk = _make_sqrt_pk()
    k = 3
    A = RC.build_contemp_adj(sqrt_pk, last_row=30, top_k=k)
    off = A.copy()
    np.fill_diagonal(off, 0.0)
    # each target row keeps exactly top_k non-zero off-diagonal sources (generic data -> no ties)
    for j in range(A.shape[0]):
        assert np.count_nonzero(off[j]) == k


def test_picks_the_correlated_sources():
    sqrt_pk = _make_sqrt_pk()
    A = RC.build_contemp_adj(sqrt_pk, last_row=35, top_k=2)
    off = A.copy()
    np.fill_diagonal(off, 0.0)
    # node 0's strongest partners are the co-moving nodes 1 and 2
    top2 = set(np.argsort(-np.abs(off[0]))[:2].tolist())
    assert top2 == {1, 2}


def test_train_only_no_lookahead():
    """Edge weights must be computed from train rows only: changing post-cutoff rows leaves A fixed."""
    sqrt_pk = _make_sqrt_pk()
    A1 = RC.build_contemp_adj(sqrt_pk, last_row=25, top_k=3)
    perturbed = sqrt_pk.copy()
    perturbed[26:] = perturbed[26:] * 5.0 + 3.0  # scramble future rows
    A2 = RC.build_contemp_adj(perturbed, last_row=25, top_k=3)
    assert np.array_equal(A1, A2)


def test_nan_safe():
    sqrt_pk = _make_sqrt_pk()
    sqrt_pk[:, 5] = np.nan  # a node with all-NaN history
    A = RC.build_contemp_adj(sqrt_pk, last_row=30, top_k=3)
    assert np.isfinite(A).all()
    assert A[5, 5] == 1.0  # self-loop still set even for the degenerate node


def test_fold_adj_vol2pk_branch_returns_stored_adj():
    sentinel = np.eye(4, dtype=np.float32) * 7.0
    D = SimpleNamespace(adj_vol2pk=sentinel)
    out = RC._fold_adj(panel=None, fold=None, wf=None, edge="vol2pk", D=D)
    assert out is sentinel


def test_fold_adj_contemp_branch_builds_from_panel():
    sqrt_pk = _make_sqrt_pk()
    panel = SimpleNamespace(pk=sqrt_pk ** 2, anchors={"train": np.array([10, 20])})
    fold = SimpleNamespace(train="train")
    wf = SimpleNamespace(horizon=1)
    A = RC._fold_adj(panel=panel, fold=fold, wf=wf, edge="contemp", D=None)
    assert A.shape == (6, 6)
    assert np.allclose(np.diag(A), 1.0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
