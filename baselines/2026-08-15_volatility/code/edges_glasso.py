"""P5: TRAIN-frozen graphical-LASSO (partial-correlation) adjacency — the market-factor-robust edge.

A raw correlation edge mostly re-encodes the common market factor already in the ``market_pk`` node
feature (EDA plan section 41). The graphical LASSO estimates a sparse precision matrix; its partial
correlations measure conditional dependence between two tickers GIVEN the others (the common factor
removed), which is the principled alternative to both raw correlation and the vol->PK edge, and the
edge family used by Zhang et al. (2308.01419). Estimated on TRAIN sqrt(PK) only and frozen (leakage-safe,
mirrors ``edges.build_vol2pk_adjacency``); pair to ``edges.swap_adjacency`` to build the graph.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT / "baselines" / "2026-08-11_eda_gnn_baseline" / "code",
           _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from edges import _train_panels  # noqa: E402  read-only: wide train (vshock, sqrt(PK)) panels


def precision_to_partial_corr(precision: np.ndarray) -> np.ndarray:
    """Partial-correlation matrix from a precision (inverse-covariance) matrix.

    ``pcorr_ij = -P_ij / sqrt(P_ii P_jj)`` for i != j; diagonal set to 1.
    """
    precision = np.asarray(precision, dtype=float)
    d = np.sqrt(np.diag(precision))
    denom = np.outer(d, d)
    denom[denom == 0.0] = np.nan
    pcorr = -precision / denom
    np.fill_diagonal(pcorr, 1.0)
    return np.nan_to_num(pcorr, nan=0.0)


def _pairwise_corr(panel: pd.DataFrame, min_pairs: int = 30) -> np.ndarray:
    """Pairwise-complete Pearson correlation matrix (columns may have different NaN patterns)."""
    corr = panel.corr(min_periods=min_pairs).to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _nearest_psd(matrix: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD (clip eigenvalues at eps), keep unit diagonal."""
    sym = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(sym)
    psd = (vecs * np.clip(vals, eps, None)) @ vecs.T
    d = np.sqrt(np.diag(psd))
    psd = psd / np.outer(d, d)                       # renormalize to a correlation matrix (unit diag)
    np.fill_diagonal(psd, 1.0)
    return psd


def glasso_partial_corr(panel: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Partial-correlation matrix (labelled) via graphical LASSO on the panel's correlation matrix.

    Uses the pairwise-complete correlation projected to PSD as the empirical covariance, so tickers
    with only partly overlapping train windows still contribute (mirrors the vol->PK pairwise estimate).
    """
    cols = list(panel.columns)
    corr = _nearest_psd(_pairwise_corr(panel))
    a = alpha
    for _ in range(6):                               # raise alpha until glasso converges
        try:
            _cov, precision = graphical_lasso(corr, alpha=a, max_iter=200)
            break
        except (FloatingPointError, ValueError):
            a *= 2.0
    else:
        precision = np.linalg.pinv(corr)             # fallback: plain precision
    pcorr = precision_to_partial_corr(precision)
    return pd.DataFrame(pcorr, index=cols, columns=cols)


def topk_adjacency(pcorr: pd.DataFrame, ticker_to_id: Mapping[str, int], top_k: int = 5) -> np.ndarray:
    """Directed adjacency A[target, source] = pcorr for each target's Top-K |partial-corr| sources
    (partial correlation is symmetric, so this yields a symmetric-support graph), plus self-loops.
    Matches the message-passing convention of ``edges.build_vol2pk_adjacency``."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    node_count = len(ticker_to_id)
    adjacency = np.zeros((node_count, node_count), dtype=np.float32)
    for target in pcorr.columns:
        if target not in ticker_to_id:
            continue
        col = pcorr[target].drop(labels=[target], errors="ignore")
        col = col[[c for c in col.index if c in ticker_to_id]].dropna()
        if col.empty:
            continue
        top_sources = col.abs().sort_values(ascending=False).head(top_k).index
        for source in top_sources:
            adjacency[ticker_to_id[target], ticker_to_id[source]] = float(pcorr.loc[source, target])
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def build_glasso_adjacency(augmented_frames, ticker_to_id: Mapping[str, int], top_k: int = 5,
                           alpha: float = 0.05) -> np.ndarray:
    """Fixed graphical-LASSO partial-correlation adjacency: each target's Top-K conditional-dependence
    neighbours, estimated on TRAIN sqrt(PK) only and frozen."""
    _vshock_wide, pk_vol_wide = _train_panels(augmented_frames)
    pcorr = glasso_partial_corr(pk_vol_wide, alpha=alpha)
    return topk_adjacency(pcorr, ticker_to_id, top_k=top_k)
