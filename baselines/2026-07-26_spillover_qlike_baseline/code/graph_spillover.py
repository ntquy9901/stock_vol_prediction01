"""Directed volatility-spillover graph construction (see design.md §2.1).

Every other baseline in this project (dual-group, macro, gated cross-attn, selective/top3 gate,
ablation) reused the SAME stock graph unchanged: `src/lstm_gat_hybrid/graph_correlation.py`'s
`construct_correlation_graph`/`construct_knn_graph`, both symmetric (contemporaneous Pearson
correlation, adj[i,j] = adj[j,i]). SOTA literature (Zhang, Pu, Cucuringu & Dong, IJF 2025; Chi et
al., J. Forecasting 2026 — see design.md §1) models volatility spillover as a DIRECTED network:
who transmits a shock to whom, not just who moves together same-day. This module provides that,
as a drop-in replacement adjacency (same shape/semantics the GAT layer already consumes) — no
model change needed.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def construct_directed_spillover_graph(volatility_window: np.ndarray, k: int = 8) -> np.ndarray:
    """Directed lead-lag volatility-spillover adjacency.

    ``adj[i, j] > 0`` means stock ``j`` (transmitter, day t) leads stock ``i``
    (receiver, day t+1): edge weight = ``abs(corr(vol_j[t], vol_i[t+1]))``, keeping only the
    top-``k`` strongest incoming edges per receiver ``i``. NOT symmetrized (adj[j, i] is
    independent and generally different) — this is the point: it is directional.

    Matches `src/lstm_gat_hybrid/model.py`'s `GraphAttentionLayer` semantics: attention score
    ``e[i, j]`` is softmax-normalized over ``j`` for fixed query/receiver ``i`` (dim=2), masked by
    ``adj[i, j] == 0``. So node ``i`` aggregates information from every ``j`` with a nonzero
    incoming edge — exactly "i listens to j's earlier volatility shock".

    Args:
        volatility_window: [seq_length, num_stocks] volatility values, same slice the sibling
            baseline's `construct_correlation_graph` receives per sequence window.
        k: number of strongest incoming edges to keep per receiver node.

    Returns:
        adj: [num_stocks, num_stocks] adjacency matrix, generally asymmetric.
    """
    seq_length, num_stocks = volatility_window.shape
    adj = np.zeros((num_stocks, num_stocks))

    if seq_length < 3:
        # Not enough points to form a lag-1 pair; return the all-zero graph (GAT layer's
        # self-loop fallback still lets each node see its own features).
        return adj

    lead = volatility_window[:-1]  # day t
    lag = volatility_window[1:]    # day t+1

    for i in range(num_stocks):  # receiver
        vol_i_lag = lag[:, i]
        if np.std(vol_i_lag) == 0:
            continue
        scores = []
        for j in range(num_stocks):  # transmitter
            if i == j:
                continue
            vol_j_lead = lead[:, j]
            if np.std(vol_j_lead) == 0:
                scores.append((j, 0.0))
                continue
            corr, _ = pearsonr(vol_j_lead, vol_i_lag)
            if np.isnan(corr):
                corr = 0.0
            scores.append((j, corr))

        scores.sort(key=lambda x: abs(x[1]), reverse=True)
        for j, w in scores[:k]:
            if w != 0.0:
                adj[i, j] = abs(w)

    return adj
