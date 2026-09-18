"""Fit the champion gamma-GBME (sklearn ``HistGradientBoostingRegressor``) and extract its per-tree leaf-index
matrix — the LITERAL Option A: the leaf-cooccurrence graph is built from GBME's OWN trees, no XGBoost proxy.

HGBR has no public ``.apply()``. Per-tree leaf indices come from ``model._predictors`` (a list per boosting
iteration, each a list-per-output of length 1 for regression). Each ``TreePredictor.nodes`` is a structured
numpy array whose fields include ``is_leaf, feature_idx, num_threshold, left, right, value``. We descend each
tree on the RAW feature path (compare ``x[feature_idx] <= num_threshold``, go ``left`` else ``right``), which
avoids the private ``_bin_mapper`` entirely, until ``is_leaf``.

Fragility (``_predictors`` and the ``nodes`` dtype are sklearn-private) is bounded by the **reconstruction
guard**: for a fitted model, ``baseline_prediction + sum(leaf value over trees) == model._raw_predict(X)`` in
link (log) space (shrinkage is baked into ``value``). ``leaf_matrix(..., check=True)`` asserts this each fold
and raises if a future sklearn upgrade changes the layout (fail-loud, not silent). Verified against the design
doc ``gbme_leafgraph_integration.html`` (reconstructed to ~1e-15).

Hyperparameters are single-sourced from ``gbme_lg_config.GBME_PARAMS`` and matched to the champion
``full_matrix.gbm`` (a test asserts identical predictions on a fixture).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402  (import first so it registers the submission `metrics` path)
import gbme_lg_config as C  # noqa: E402

FL = FM.FL


def fit_gbme(trf, cols, seed, floor=FL):
    """Fit one gamma-GBME (HGBR) on the causal train rows ``trf`` (floored target). Returns the FITTED model so
    its per-tree leaf indices can be read for the cooccurrence graph. Params single-sourced from
    ``GBME_PARAMS`` and matched to ``full_matrix.gbm``."""
    m = HistGradientBoostingRegressor(random_state=int(seed), **C.GBME_PARAMS)
    m.fit(trf[cols].to_numpy(float), np.maximum(trf["y"].to_numpy(float), floor))
    return m


def predict_gbme(model, X, floor=FL):
    """Floored gamma prediction of one fitted GBME (matches ``full_matrix.gbm``: ``np.maximum(predict, floor)``,
    no upper cap on the base -> exact parity with the deployed champion)."""
    return np.maximum(model.predict(np.asarray(X, float)), floor)


def _tree_predictors(model):
    """The per-tree ``TreePredictor`` objects (one output for regression), from the private ``_predictors``."""
    return [stage[0] for stage in model._predictors]


def _traverse_tree(nodes, X):
    """Vectorised single-tree descent. ``nodes`` is a HGBR ``TreePredictor.nodes`` structured array; ``X`` is
    an (n x n_features) float array. Returns ``(leaf_node_idx[n], leaf_value[n])``.

    All samples descend in lockstep: an ``active = ~is_leaf[node]`` mask advances only the samples still at an
    internal node (``left`` if ``x[feature_idx] <= num_threshold`` else ``right``) until every sample is at a
    leaf. Bounded by tree depth — no per-sample Python loop (the perf-critical path on SP500).

    Precondition: ``X`` is NaN-free (the panel drops rows with NaN own-history features). HGBR routes missing
    values per-node via ``missing_go_to_left``; this raw ``<=`` path sends NaN right instead, so a stray NaN
    would mis-route and trip the reconstruction guard in ``leaf_matrix`` (fail-loud, never silent)."""
    n = X.shape[0]
    is_leaf = nodes["is_leaf"].astype(bool)
    fidx = nodes["feature_idx"]
    thr = nodes["num_threshold"]
    left = nodes["left"]
    right = nodes["right"]
    value = nodes["value"]
    node = np.zeros(n, dtype=np.int64)
    active = ~is_leaf[node]
    while active.any():
        cur = node[active]
        go_left = X[active, fidx[cur]] <= thr[cur]
        node[active] = np.where(go_left, left[cur], right[cur])
        active = ~is_leaf[node]
    return node, value[node]


def leaf_matrix(model, X, check=True, tol=C.RECON_TOL):
    """(n_samples x n_trees) int32 leaf-node-index matrix from a fitted HGBR, plus a fail-loud reconstruction
    guard. Traverses every tree on the RAW ``num_threshold`` path (no ``_bin_mapper``).

    When ``check`` is True, asserts ``max| baseline_prediction + sum(leaf value) - _raw_predict(X) | <= tol`` in
    link space — pinning against sklearn layout drift. Raises ``RuntimeError`` on failure (never silently
    degrades). The guard verifies the leaf-VALUE sum; the graph consumes the leaf INDICES. The raw
    ``num_threshold`` traversal is exact, so index-correctness is guaranteed by construction; the guard is a
    defensive cross-check that additionally catches any layout change (leaf values are effectively unique floats,
    so a wrong-leaf landing that preserved the value sum is not realistic)."""
    X = np.asarray(X, float)
    preds = _tree_predictors(model)
    n, t = X.shape[0], len(preds)
    out = np.empty((n, t), dtype=np.int32)
    value_sum = np.zeros(n, dtype=float)
    for ti, tp in enumerate(preds):
        node, leaf_val = _traverse_tree(tp.nodes, X)
        out[:, ti] = node
        value_sum += leaf_val
    if check and n:                                               # _raw_predict rejects 0-row input
        recon = float(model._baseline_prediction) + value_sum
        raw = np.asarray(model._raw_predict(X), float).ravel()
        max_err = float(np.max(np.abs(recon - raw)))
        if not (max_err <= tol):
            raise RuntimeError(
                f"HGBR leaf reconstruction failed: max|recon - _raw_predict| = {max_err:.3e} > tol {tol:.1e}. "
                "sklearn tree layout may have changed (private _predictors/.nodes) — extraction is unsafe.")
    return out
