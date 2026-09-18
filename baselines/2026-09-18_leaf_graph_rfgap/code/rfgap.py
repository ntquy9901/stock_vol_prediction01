"""XGBoost gamma booster + leaf-cooccurrence proximity that smooths its predictions (v2: RF-GAP / KeRF).

v1 (``baselines/2026-09-18_gbm_leaf_graph``) built a hard top-k leaf-Hamming kNN graph. v2 keeps that as one
selectable scheme and adds two literature-grounded *weighted* proximities:

- **RF-GAP** (arXiv 2307.01077): treat the boosted-tree leaves as a proper regression proximity. For one day's
  cross-section, each tree ``t`` puts stock ``i`` in a leaf; the other same-day stocks sharing that leaf,
  ``C_i(t)``, each receive weight ``1/|C_i(t)|`` from tree ``t`` (leave-one-out: self excluded). Averaging over
  the trees that give ``i`` at least one co-member and renormalising yields a row-stochastic proximity
  ``W`` (``sum_j W[i,j]=1``, ``W[i,i]=0``). ``neighbour_mean_i = sum_j W[i,j] y_j`` is exactly the tree-ensemble's
  leave-one-out same-day prediction — a stock never dominates its own smoothing.
- **KeRF large-leaf down-weighting** (arXiv 2601.02735): scale each tree's contribution by
  ``KERF_FUNC(leaf_population)`` (``1/pop`` or ``1/sqrt(pop)``). A giant storm-day leaf (hundreds of trivially
  "similar" stocks) then contributes little, directly targeting HOSE spike-robustness. Still row-stochastic.

The proximity is strictly causal: it uses only that day's cross-section (same-day leaf-vectors) and a booster fit
on the past train window. No future rows, no cross-day mixing (each date's cross-section smoothed independently).

Reuses the champion floor from ``full_matrix`` (single source) and the tested per-obs QLIKE from ``metrics``.
All tunables come from ``rfgap_config``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import xgboost as xgb

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402  (import first so it registers the submission `metrics` path)
import metrics as M  # noqa: E402
import rfgap_config as C  # noqa: E402

FL = FM.FL


# --------------------------------------------------------------------------- XGBoost gamma booster
def _params(seed):
    """XGBoost ``reg:gamma`` params mirroring the champion HGBR gamma capacity (single-sourced from config)."""
    return {"objective": "reg:gamma", "eta": C.XGB_LR, "max_leaves": C.XGB_MAX_LEAVES,
            "max_depth": C.XGB_MAX_DEPTH, "grow_policy": "lossguide", "lambda": C.XGB_L2,
            "min_child_weight": C.XGB_MIN_CHILD_WEIGHT, "seed": int(seed), "tree_method": "hist",
            "verbosity": 0}


def fit_booster(trf, cols, seed, floor=FL):
    """Fit one XGBoost gamma booster on the causal train rows ``trf`` (floored target). Returns the booster so
    its per-tree leaf indices can be read for the cooccurrence proximity."""
    y_tr = np.maximum(trf["y"].to_numpy(float), floor)
    d_tr = xgb.DMatrix(trf[cols].to_numpy(float), label=y_tr)
    return xgb.train(_params(seed), d_tr, num_boost_round=C.XGB_N_ESTIMATORS)


def predict_booster(bst, X, floor=FL):
    """Floored gamma prediction of one booster on feature matrix ``X`` (clipped to [floor, PRED_CAP])."""
    return np.clip(bst.predict(xgb.DMatrix(np.asarray(X, float))), floor, C.PRED_CAP)


def predict_xgb(trf, combo, cols, seeds, floor=FL):
    """Seed-ensembled plain XGBoost gamma base prediction over the ``combo`` rows (this is the SHARED base for
    every smoothing variant, so the graph effect is isolated)."""
    X = combo[cols].to_numpy(float)
    return np.mean([predict_booster(fit_booster(trf, cols, s, floor), X, floor) for s in seeds], 0)


def leaf_matrix(bst, X):
    """Integer (n_samples x n_trees) leaf-index matrix from a fitted booster (``pred_leaf=True``)."""
    return bst.predict(xgb.DMatrix(np.asarray(X, float)), pred_leaf=True).astype(np.int32)


# --------------------------------------------------------------------------- leaf-Hamming (v1 'knn' scheme)
def day_similarity(leaf_rows):
    """Pairwise leaf-Hamming similarity for one day's cross-section (v1 scheme).

    ``leaf_rows`` is (m x T) integer leaf indices for ``m`` stocks over ``T`` trees. Returns an (m x m) matrix
    whose (i, j) entry is the fraction of trees where stock i and stock j land in the SAME leaf. Diagonal 1.0.
    Sparse one-hot inner product (S @ S.T)/T over globally-unique (tree, leaf) ids — exact and cheap."""
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


# --------------------------------------------------------------------------- RF-GAP / KeRF proximity
def _kerf_weight(pop, kerf_func):
    """KeRF large-leaf down-weight of a leaf given its (same-day) population ``pop`` (array or scalar)."""
    pop = np.asarray(pop, float)
    if kerf_func == "invsqrt":
        return 1.0 / np.sqrt(pop)
    if kerf_func == "inv":
        return 1.0 / pop
    raise ValueError(f"unknown KERF_FUNC {kerf_func!r} (expected 'inv' or 'invsqrt')")


def day_weights(leaf_rows, scheme, k=None, kerf_func=None):
    """Reference (m x m) row-stochastic proximity matrix for one day's cross-section (self weight 0).

    - ``knn``        : hard top-``k`` leaf-Hamming neighbours, each weight ``1/kk``.
    - ``rfgap``      : each tree contributes a leave-one-out distribution over same-leaf co-members (weight
                       ``1/|co-members|``), averaged over trees with >=1 co-member and renormalised to sum 1.
    - ``rfgap_kerf`` : same, but each tree's contribution scaled by ``KERF_FUNC(leaf_population)``.

    Rows with no co-member anywhere (fully isolated stocks) are returned all-zero; the caller treats an all-zero
    row as "no smoothing" (neighbour mean = own base prediction). This function is the exact reference used by
    the tests and mirrored by the vectorised ``neighbour_mean`` hot path (``test_*_matches_reference``)."""
    lr = np.asarray(leaf_rows, np.int64)
    m, t = lr.shape
    W = np.zeros((m, m), float)
    if m == 1:
        return W
    if scheme == "knn":
        sim = day_similarity(lr)
        np.fill_diagonal(sim, -np.inf)
        kk = min(int(k if k is not None else C.K_NEIGHBOURS), m - 1)
        idx = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        np.put_along_axis(W, idx, 1.0 / kk, axis=1)
        return W
    if scheme not in ("rfgap", "rfgap_kerf"):
        raise ValueError(f"unknown scheme {scheme!r}")
    kf = kerf_func if kerf_func is not None else C.KERF_FUNC
    num = np.zeros((m, m), float)
    den = np.zeros(m, float)
    for col in range(t):
        leaves = lr[:, col]
        same = leaves[:, None] == leaves[None, :]
        np.fill_diagonal(same, False)                              # leave-one-out: self is never a co-member
        co = same.sum(1).astype(float)                            # number of same-day co-members in this tree
        pos = co > 0
        if not pos.any():
            continue
        wt = np.ones(m) if scheme == "rfgap" else _kerf_weight(co + 1.0, kf)   # pop = co-members + self
        num[pos] += (wt[pos, None] * same[pos]) / co[pos, None]   # per-tree LOO distribution, scaled by wt
        den[pos] += wt[pos]
    nz = den > 0
    W[nz] = num[nz] / den[nz, None]
    return W


def _rfgap_neighbour_mean(pred, leaf_rows, kerf_func):
    """Vectorised RF-GAP(+KeRF) leave-one-out neighbour mean for one day (matches ``day_weights @ pred``).

    ``kerf_func`` is ``None`` for plain RF-GAP, else a KeRF down-weight name. Computes, per tree, each stock's
    same-leaf group sum/size via a single bincount over globally-unique (tree, leaf) ids, forms the leave-one-out
    mean ``(group_sum - self)/(group_size - 1)`` for trees where the stock has >=1 co-member, and averages those
    per-tree LOO means with weights ``1`` (RF-GAP) or ``KERF_FUNC(group_size)`` (KeRF). Fully vectorised, O(m*T)
    memory, no (m x m) matrix — the hot path for the large train/val/test panels."""
    pred = np.asarray(pred, float)
    m, t = leaf_rows.shape
    if m == 1:
        return pred.copy()
    lr = np.asarray(leaf_rows, np.int64)
    off = int(lr.max()) + 1
    gid = lr + np.arange(t, dtype=np.int64) * off                 # (m, T) globally-unique (tree, leaf) id
    nbins = int(gid.max()) + 1
    flat = gid.ravel()
    gc = np.bincount(flat, minlength=nbins).astype(float)         # population per (tree, leaf)
    gs = np.bincount(flat, weights=np.repeat(pred, t), minlength=nbins)   # base-pred sum per (tree, leaf)
    GC = gc[gid]                                                  # (m, T) leaf population for each stock/tree
    GS = gs[gid]                                                  # (m, T) leaf pred-sum for each stock/tree
    valid = GC > 1.0                                              # tree gives the stock >=1 co-member
    denom_loo = np.where(valid, GC - 1.0, 1.0)
    loo = np.where(valid, (GS - pred[:, None]) / denom_loo, 0.0)  # per-tree leave-one-out neighbour mean
    w = valid.astype(float) if kerf_func is None else np.where(valid, _kerf_weight(GC, kerf_func), 0.0)
    den = w.sum(1)
    num = (w * loo).sum(1)
    has = den > 0
    return np.where(has, num / np.where(has, den, 1.0), pred)     # isolated stock -> own base prediction


def neighbour_mean(pred, leaf_rows, scheme, k=None, kerf_func=None):
    """Dispatch to the scheme's neighbour mean: hard kNN (v1) or vectorised RF-GAP(+KeRF)."""
    pred = np.asarray(pred, float)
    if scheme == "knn":
        if len(pred) == 1:
            return pred.copy()
        return knn_neighbour_mean(pred, day_similarity(leaf_rows), k if k is not None else C.K_NEIGHBOURS)
    if scheme == "rfgap":
        return _rfgap_neighbour_mean(pred, leaf_rows, None)
    if scheme == "rfgap_kerf":
        return _rfgap_neighbour_mean(pred, leaf_rows, kerf_func if kerf_func is not None else C.KERF_FUNC)
    raise ValueError(f"unknown scheme {scheme!r}")


# --------------------------------------------------------------------------- smoothing + alpha fit
def smooth_day(pred, leaf_rows, scheme, k, alpha, kerf_func=None):
    """Graph-smooth one day's base predictions: ``(1-alpha)*pred + alpha*neighbour_mean``.

    ``alpha == 0`` (or a singleton day) returns the base unchanged; ``alpha == 1`` returns the neighbour mean."""
    pred = np.asarray(pred, float)
    if alpha == 0.0 or len(pred) == 1:
        return pred.copy()
    nbr = neighbour_mean(pred, leaf_rows, scheme, k, kerf_func)
    return (1.0 - alpha) * pred + alpha * nbr


def smooth_all(pred, leaves, dates, scheme, k, alpha, kerf_func=None):
    """Apply per-day smoothing to every row, grouping by ``dates`` (each date's cross-section smoothed
    independently -> strictly causal, no cross-day mixing). ``alpha == 0`` is an identity fast path that builds
    no graph (exact, and the expected NO-GO case)."""
    pred = np.asarray(pred, float)
    out = pred.copy()
    if alpha == 0.0:
        return out
    dates = np.asarray(dates)
    for d in np.unique(dates):
        mask = dates == d
        out[mask] = smooth_day(pred[mask], leaves[mask], scheme, k, alpha, kerf_func)
    return out


def fit_alpha(y, pred, leaves, dates, scheme, k, grid, kerf_func=None, floor=FL):
    """Pick the smoothing weight in ``grid`` that minimises validation QLIKE (frozen for test). Returns
    ``(best_alpha, best_qlike)``; ties keep the smaller alpha (grid ascending, strict ``<`` update)."""
    best_a, best_q = grid[0], float("inf")
    for a in grid:
        q = float(np.mean(M.per_obs_qlike(y, smooth_all(pred, leaves, dates, scheme, k, a, kerf_func), floor=floor)))
        if q < best_q:
            best_a, best_q = a, q
    return best_a, best_q
