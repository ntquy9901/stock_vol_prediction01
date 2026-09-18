"""Leaf-cooccurrence graph that smooths the champion GBME's predictions.

Each observation's *leaf-vector* is its row of the HGBR leaf-index matrix (see ``hgbr_leaf.leaf_matrix``) — the
integer leaf it lands in for each of the ``T`` trees. Two same-day stocks are neighbours if their leaf-vectors
overlap a lot (fraction of trees landing in the SAME leaf = Hamming similarity). A per-day kNN graph over that
similarity smooths the base prediction::

    y_smooth_i = (1 - alpha) * y_i + alpha * mean_{j in kNN(i)} y_j

with ``alpha`` fit on a validation slice per fold and frozen for test. The leaf-graph is a deterministic
re-encoding of the SAME own-history features the GBME already used, so it carries no new information; this
module measures — honestly — whether smoothing over it helps per-stock QLIKE.

Causality: the graph on any day uses only that day's cross-section (same-day leaf-vectors); each date is
smoothed independently (no cross-day mixing). Pure numpy/scipy — no XGBoost dependency (the leaf-vectors come
from the real GBME trees). The graph functions mirror the verified helpers in the sibling
``2026-09-18_gbm_leaf_graph``; the per-obs QLIKE for alpha selection is the tested one in ``metrics``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402  (import first so it registers the submission `metrics` path)
import metrics as M  # noqa: E402

FL = FM.FL


def day_similarity(leaf_rows):
    """Pairwise leaf-Hamming similarity for one day's cross-section.

    ``leaf_rows`` is (m x T) integer leaf indices for ``m`` stocks over ``T`` trees. Returns an (m x m) matrix
    whose (i, j) entry is the fraction of trees where stock i and stock j land in the SAME leaf. Identical
    leaf-vectors -> 1.0; fully disjoint -> 0.0. Diagonal is 1.0.

    Implemented as a sparse one-hot inner product (S @ S.T) / T over globally-unique (tree, leaf) ids, which is
    exact and far cheaper than an m x m x T broadcast."""
    leaf_rows = np.asarray(leaf_rows, np.int64)
    m, t = leaf_rows.shape
    off = int(leaf_rows.max()) + 1                                 # per-tree id offset -> globally unique ids
    ids = (leaf_rows + np.arange(t, dtype=np.int64) * off).ravel()
    rows = np.repeat(np.arange(m), t)
    s = sp.csr_matrix((np.ones(m * t, np.float32), (rows, ids)), shape=(m, int(ids.max()) + 1))
    return (s @ s.T).toarray() / float(t)


def knn_neighbour_mean(pred, sim, k):
    """Mean base prediction over each stock's top-``k`` leaf-similar neighbours (self excluded).

    A singleton cross-section (m == 1) has no neighbours, so it returns the base prediction unchanged."""
    pred = np.asarray(pred, float)
    m = len(pred)
    if m == 1:
        return pred.copy()
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)                                    # never a neighbour of itself
    kk = min(int(k), m - 1)
    idx = np.argpartition(-s, kk - 1, axis=1)[:, :kk]              # top-kk by similarity (unordered is fine)
    return pred[idx].mean(axis=1)


def smooth_day(pred, leaf_rows, k, alpha):
    """Graph-smooth one day's base predictions: (1-alpha)*pred + alpha*neighbour_mean.

    ``alpha == 0`` (or a singleton day) returns the base unchanged; ``alpha == 1`` returns the neighbour-mean."""
    pred = np.asarray(pred, float)
    if alpha == 0.0 or len(pred) == 1:
        return pred.copy()
    nbr = knn_neighbour_mean(pred, day_similarity(leaf_rows), k)
    return (1.0 - alpha) * pred + alpha * nbr


def smooth_all(pred, leaves, dates, k, alpha):
    """Apply per-day leaf-graph smoothing to every row, grouping by ``dates`` (each date's cross-section is
    smoothed independently -> strictly causal, no cross-day mixing). ``alpha == 0`` is an identity fast path
    that skips building any graph (exact, and the expected NO-GO case)."""
    pred = np.asarray(pred, float)
    out = pred.copy()
    if alpha == 0.0:
        return out
    dates = np.asarray(dates)
    for d in np.unique(dates):
        mask = dates == d
        out[mask] = smooth_day(pred[mask], leaves[mask], k, alpha)
    return out


def fit_alpha(y, pred, leaves, dates, k, grid, floor=FL):
    """Pick the smoothing weight in ``grid`` that minimises validation QLIKE (frozen for test). Returns
    ``(best_alpha, best_qlike)``; ties keep the smaller alpha (grid is ascending, strict ``<`` update)."""
    best_a, best_q = grid[0], float("inf")
    for a in grid:
        q = float(np.mean(M.per_obs_qlike(y, smooth_all(pred, leaves, dates, k, a), floor=floor)))
        if q < best_q:
            best_a, best_q = a, q
    return best_a, best_q
