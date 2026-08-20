"""TRAIN-frozen graphical-LASSO (partial-correlation) Top-K adjacency for the HAR-LSTM-GAT graph.

A raw correlation edge mostly re-encodes the common market factor. The graphical LASSO estimates a
sparse precision (inverse-covariance) matrix whose partial correlations measure conditional
dependence between two tickers GIVEN the others (common factor removed) — the principled, market-
factor-robust edge (Zhang et al., 2308.01419). The graph is estimated on TRAIN rows only (the caller
guarantees ``pk_wide_train`` holds train rows) and frozen, so it is leakage-safe.

Distilled from ``baselines/2026-08-15_volatility/code/edges_glasso.py``: keeps
``precision_to_partial_corr``, the alpha-raise graphical-LASSO loop with a pinv fallback, and Top-K
selection; drops the augmented-frame / ticker-id plumbing (caller passes a plain wide panel).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso

# Ascending regularization ladder tried until graphical_lasso converges (stronger alpha = sparser,
# better conditioned). A caller-supplied ``alpha`` is prepended.
_ALPHA_LADDER: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8)


def precision_to_partial_corr(precision: np.ndarray) -> np.ndarray:
    """Partial-correlation matrix from a precision (inverse-covariance) matrix.

    ``pcorr_ij = -P_ij / sqrt(P_ii * P_jj)`` for i != j; diagonal set to 1. Non-finite entries
    (from zero/negative pivots on a degenerate precision) are mapped to 0 so the result is finite.
    """
    precision = np.asarray(precision, dtype=float)
    d = np.sqrt(np.abs(np.diag(precision)))
    denom = np.outer(d, d)
    denom[denom == 0.0] = np.nan
    pcorr = -precision / denom
    np.fill_diagonal(pcorr, 1.0)
    return np.nan_to_num(pcorr, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_precision(corr: np.ndarray, alpha_ladder: tuple[float, ...]) -> np.ndarray:
    """Graphical-LASSO precision, raising alpha until it converges; pinv fallback if all fail."""
    for a in alpha_ladder:
        try:
            _cov, precision = graphical_lasso(corr, alpha=a, max_iter=200)
        except (FloatingPointError, ValueError):
            continue
        if np.all(np.isfinite(precision)):
            return precision
    return np.linalg.pinv(corr)  # fallback: plain (pseudo-)inverse — always finite


def glasso_adjacency(
    pk_wide_train: pd.DataFrame, top_k: int = 5, alpha: float | None = None
) -> np.ndarray:
    """Undirected graphical-LASSO partial-correlation Top-K adjacency.

    Parameters
    ----------
    pk_wide_train : one column per ticker, TRAIN rows only (caller-guaranteed leakage-safe).
    top_k : per node, keep the ``top_k`` strongest ``|partial-corr|`` off-diagonal edges.
    alpha : optional starting graphical-LASSO penalty (prepended to the auto-raise ladder);
            ``None`` starts the ladder at 0.05.

    Returns
    -------
    ``float32`` ``[N, N]`` adjacency with unit self-loops (diagonal = 1.0). The graph is made
    **undirected** by keeping, for each pair, the larger-magnitude of the two directed Top-K entries
    (magnitude-max symmetrization — preserves the sign of negative partial correlations, unlike a
    plain ``np.maximum``). Always finite even when the covariance is singular (pinv fallback).
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    n = pk_wide_train.shape[1]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # 1. Correlation of the panel (pairwise-complete), forced symmetric with unit diagonal.
    corr = pk_wide_train.corr().to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    corr = 0.5 * (corr + corr.T)

    # 2. Graphical LASSO -> 3. partial correlations.
    ladder = ((alpha,) + _ALPHA_LADDER) if alpha is not None else _ALPHA_LADDER
    precision = _fit_precision(corr, ladder)
    pcorr = precision_to_partial_corr(precision)

    # 4. Top-K off-diagonal edges per (target) node.
    off = pcorr.copy()
    np.fill_diagonal(off, 0.0)
    k = min(top_k, n - 1)
    directed = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        strengths = np.abs(off[i])
        # Deterministic descending order; ties broken by index (stable via argsort on -strength).
        order = np.argsort(-strengths, kind="stable")[:k]
        for j in order:
            if strengths[j] > 0.0:
                directed[i, j] = pcorr[i, j]

    # Symmetrize (undirected): keep the larger-magnitude directed entry of each pair, sign intact.
    keep_ij = np.abs(directed) >= np.abs(directed.T)
    adjacency = np.where(keep_ij, directed, directed.T).astype(np.float32)

    # 5. Self-loops.
    np.fill_diagonal(adjacency, 1.0)
    return adjacency
