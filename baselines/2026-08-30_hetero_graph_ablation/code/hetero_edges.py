"""Two SEPARATE relation adjacencies for the heterogeneous HNX volatility GNN probe (7th edge probe).

Unlike ``baselines/2026-08-29_corrlift_ablation`` (which SQUASHES the linear Pearson-|rho| edge and the
non-linear Apriori-lift edge into ONE adjacency), this builds the two relations SEPARATELY so a
heterogeneous network can learn INDEPENDENT weights per relation:

  * ``linear_corr``     -- edge if ``|rho_ij| > corr_thresh`` (default 0.25); weight = ``|rho_ij|``.
  * ``nonlinear_assoc`` -- edge if ``lift_ij > lift_thresh`` (default 1.2);  weight = ``lift_ij``.

Both are symmetric (undirected) with self-loop 1.0, matching the ``WeightedGATLayer`` convention.

PER-RELATION MIN-MAX NORMALIZATION (train-only): the two edge-weight spaces differ (|rho| in (0.25,1];
lift in (1.2,max]). Each relation's FIRED off-diagonal weights are Min-Max scaled to [0,1] INDEPENDENTLY
so gradients are not biased toward one relation's larger raw scale. A tiny ``EPS`` lower clip keeps the
weakest edge present under the GAT's ``adjacency != 0`` mask (values remain in [0,1]).

THRESHOLDS DEPART FROM THE PAPER: arXiv:2502.15813 §3.2 uses 0.7 / 1.7 (which give only 3 / 12 edges on the
154 HNX nodes -- a near-empty graph). The lowered 0.25 / 1.2 give each relation a non-trivial (but weak/noisy
at rho~0.25) graph so the heterogeneous architecture has something to propagate over. This is a denser-graph
heterogeneous VARIANT, not the paper's faithful thresholds -- reported honestly.

LEAKAGE (strict): the whole graph -- returns, correlations, per-stock move thresholds, supports, lifts, AND the
per-relation Min-Max min/max -- is computed from TRAIN close rows ONLY (``date < cutoff_date``, the caller uses
``D.d_va[0]``) then frozen. The Min-Max min/max are therefore train-only by construction.

The Pearson / lift primitives (``pearson_corr``, ``move_events``, ``pairwise_lift``, ``daily_returns``,
``load_close_wide``) are imported READ-ONLY from the corrlift baseline; this module only adds the per-relation
thresholding + Min-Max + diag.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CORRLIFT = Path(__file__).resolve().parents[2] / "2026-08-29_corrlift_ablation" / "code"
sys.path.insert(0, str(_CORRLIFT))

import corrlift_edge as CL  # noqa: E402  (read-only reuse of the published corr/lift primitives)

# Denser-than-paper thresholds (single source of truth; recorded in each result's edge_config).
CORR_THRESH = 0.25    # |Pearson rho| edge threshold for the linear_corr relation (paper uses 0.7)
LIFT_THRESH = 1.2     # association-rule lift threshold for the nonlinear_assoc relation (paper uses 1.7)
MIN_OVERLAP = CL.MIN_OVERLAP    # min co-finite days for a stable Pearson rho (reused: 100)
MIN_PAIRS = CL.MIN_PAIRS        # min co-observed transactions for a stable lift (reused: 30)
EPS = 1e-6            # lower clip so the weakest normalized edge stays present under the GAT != 0 mask

# re-export so the runner imports the close-panel loader from one place
load_close_wide = CL.load_close_wide


def fit_minmax(weights: np.ndarray) -> tuple[float, float]:
    """Train-only Min-Max (lo, hi) over a relation's FIRED edge weights. Empty -> (0.0, 1.0) (no-op)."""
    if weights.size == 0:
        return (0.0, 1.0)
    return (float(weights.min()), float(weights.max()))


def apply_minmax(weights: np.ndarray, lo: float, hi: float, eps: float = EPS) -> np.ndarray:
    """Map ``weights`` to [0,1] via Min-Max ``(w-lo)/(hi-lo)``, then lower-clip to ``eps`` so a normalized 0
    (the minimum edge) stays a present edge under the GAT ``!= 0`` mask. Degenerate ``hi <= lo`` (<=1 distinct
    fired weight) -> all map to 1.0 (a single flat relation still contributes its edges)."""
    if hi <= lo:
        return np.ones_like(weights, dtype=np.float32)
    norm = (weights - lo) / (hi - lo)
    return np.clip(norm, eps, 1.0).astype(np.float32)


def _relation_adjacency(fires: np.ndarray, raw_weight: np.ndarray) -> tuple[np.ndarray, dict]:
    """Build ONE symmetric relation adjacency (self-loop=1) from a boolean ``fires`` mask + ``raw_weight``
    magnitude, applying per-relation train-only Min-Max. Returns (adjacency [N,N] float32, stats dict)."""
    n = fires.shape[0]
    fires = fires.copy()
    np.fill_diagonal(fires, False)
    lo, hi = fit_minmax(raw_weight[fires])
    adj = np.zeros((n, n), dtype=np.float32)
    adj[fires] = apply_minmax(raw_weight[fires], lo, hi)
    np.fill_diagonal(adj, 1.0)
    off = adj.copy()
    np.fill_diagonal(off, 0.0)
    deg = (off != 0).sum(axis=1)
    iu = np.triu_indices(n, k=1)
    stats = {
        "n_edges": int(fires[iu].sum()),
        "avg_off_degree": float(deg.mean()) if n else 0.0,
        "max_off_degree": int(deg.max()) if n else 0,
        "n_singletons": int((deg == 0).sum()),
        "minmax": [lo, hi],
    }
    return adj, stats


def build_relation_adjacencies(
    close_wide: pd.DataFrame,
    cutoff_date,
    corr_thresh: float = CORR_THRESH,
    lift_thresh: float = LIFT_THRESH,
    min_overlap: int = MIN_OVERLAP,
    min_pairs: int = MIN_PAIRS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Two per-relation ``[N,N]`` float32 adjacencies (linear_corr, nonlinear_assoc) + an edge-density diag.

    TRAIN-ONLY: only close rows with ``date < cutoff_date`` feed returns/correlations/supports/lifts. An edge
    fires per relation independently (``|rho| > corr_thresh`` for linear; ``lift > lift_thresh`` for
    non-linear); its weight is the per-relation Min-Max-normalized magnitude. Both symmetric, diagonal 1.0.
    """
    cutoff = pd.Timestamp(cutoff_date)
    train = close_wide.loc[close_wide.index < cutoff]
    n = close_wide.shape[1]
    returns = CL.daily_returns(train)
    corr = CL.pearson_corr(returns, min_overlap)
    event, valid = CL.move_events(returns)
    lift = CL.pairwise_lift(event, valid, min_pairs)

    corr_fires = np.isfinite(corr) & (np.abs(corr) > corr_thresh)
    lift_fires = np.isfinite(lift) & (lift > lift_thresh)
    adj_lin, lin_stats = _relation_adjacency(corr_fires, np.abs(np.nan_to_num(corr)))
    adj_nl, nl_stats = _relation_adjacency(lift_fires, np.nan_to_num(lift))

    iu = np.triu_indices(n, k=1)
    np.fill_diagonal(corr_fires, False)
    np.fill_diagonal(lift_fires, False)
    diag = {
        "n_nodes": n,
        "n_pairs": int(iu[0].size),
        "linear_corr": {"thresh": corr_thresh, "min_overlap": min_overlap, **lin_stats},
        "nonlinear_assoc": {"thresh": lift_thresh, "min_pairs": min_pairs, **nl_stats},
        "n_both_relations_edges": int((corr_fires & lift_fires)[iu].sum()),
        "n_train_rows": int(len(train)),
    }
    return adj_lin, adj_nl, diag
