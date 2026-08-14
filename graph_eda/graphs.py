"""Graph construction/diagnostics: Top-K neighbours, stability, turnover, controls."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from graph_eda.leakage import assert_snapshot_no_lookahead


def top_k_neighbors(corr: pd.DataFrame, k: int, use_abs: bool = True) -> dict[str, list[str]]:
    """Top-K neighbours per node by (absolute) correlation, excluding self."""
    out: dict[str, list[str]] = {}
    for node in corr.index:
        row = corr.loc[node].drop(labels=[node], errors="ignore")
        rank = row.abs() if use_abs else row
        out[node] = list(rank.sort_values(ascending=False).head(k).index)
    return out


def jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard similarity of two neighbour sets (1.0 for two empty sets)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def rolling_snapshots(
    wide: pd.DataFrame, window: int, step: int, method: str = "pearson"
) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """Trailing-window correlation snapshots at ``step``-spaced dates (leakage-safe).

    Snapshot at date t uses rows (t-window+1 .. t) only. Returns [(date, corr_df)].
    """
    snaps = []
    idx = wide.index
    for end in range(window - 1, len(idx), step):
        block = wide.iloc[end - window + 1 : end + 1]
        if block.dropna(how="any").shape[0] >= max(10, window // 2):
            snaps.append((idx[end], block.corr(method=method)))
    return snaps


def neighbor_stability(
    snaps: list[tuple[pd.Timestamp, pd.DataFrame]], k: int, use_abs: bool = True
) -> pd.DataFrame:
    """Per-node mean Jaccard of Top-K neighbour sets across consecutive snapshots."""
    if len(snaps) < 2:
        return pd.DataFrame(columns=["node", "mean_jaccard"])
    per_node: dict[str, list[float]] = {n: [] for n in snaps[0][1].index}
    prev = top_k_neighbors(snaps[0][1], k, use_abs)
    for _, corr in snaps[1:]:
        cur = top_k_neighbors(corr, k, use_abs)
        for n in per_node:
            per_node[n].append(jaccard(prev.get(n, []), cur.get(n, [])))
        prev = cur
    return pd.DataFrame(
        [(n, float(np.mean(v)) if v else np.nan) for n, v in per_node.items()],
        columns=["node", "mean_jaccard"],
    )


def edge_turnover(
    snaps: list[tuple[pd.Timestamp, pd.DataFrame]], k: int, use_abs: bool = True
) -> pd.DataFrame:
    """Global Top-K edge-set turnover between consecutive snapshots (plan section 45)."""
    rows = []
    prev_edges: set | None = None
    for date, corr in snaps:
        nbrs = top_k_neighbors(corr, k, use_abs)
        edges = {frozenset((n, m)) for n, ms in nbrs.items() for m in ms}
        if prev_edges is not None:
            inter = len(edges & prev_edges)
            union = len(edges | prev_edges)
            rows.append((date, 1 - inter / union if union else 0.0, len(edges)))
        prev_edges = edges
    return pd.DataFrame(rows, columns=["date", "edge_turnover", "n_edges"])


def edge_stability(
    snaps: list[tuple[pd.Timestamp, pd.DataFrame]], thresholds=(0.2, 0.3, 0.4, 0.5)
) -> pd.DataFrame:
    """Per-pair rolling-corr stability metrics across snapshots (plan section 19)."""
    if not snaps:
        return pd.DataFrame()
    cols = list(snaps[0][1].index)
    series: dict[tuple[str, str], list[float]] = {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            series[(cols[i], cols[j])] = []
    for _, corr in snaps:
        for (a, b), lst in series.items():
            lst.append(corr.loc[a, b])
    rows = []
    for (a, b), vals in series.items():
        v = np.array(vals, dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        dom = np.sign(np.nanmean(v))
        rec = {
            "source": a,
            "target": b,
            "mean_corr": float(v.mean()),
            "std_corr": float(v.std()),
            "sign_consistency": float(np.mean(np.sign(v) == dom)),
        }
        for tau in thresholds:
            rec[f"persistence_{tau}"] = float(np.mean(np.abs(v) > tau))
        rows.append(rec)
    return pd.DataFrame(rows)


def graph_density_series(
    snaps: list[tuple[pd.Timestamp, pd.DataFrame]], tau: float
) -> pd.DataFrame:
    """Threshold-graph density/avg-degree over time (plan section 44)."""
    rows = []
    for date, corr in snaps:
        n = corr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        edges = int(np.sum(np.abs(corr.values[mask]) > tau) / 2)
        max_edges = n * (n - 1) / 2
        rows.append((date, edges, edges / max_edges, 2 * edges / n))
    return pd.DataFrame(rows, columns=["date", "n_edges", "density", "avg_degree"])


def random_matched_neighbors(
    nodes: list[str], k: int, seed: int = 0
) -> dict[str, list[str]]:
    """Random Top-K neighbour control matched on node/edge count (plan section 37)."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in nodes:
        others = [x for x in nodes if x != n]
        out[n] = list(rng.choice(others, size=min(k, len(others)), replace=False))
    return out


def _edge_graph(corr: pd.DataFrame, k: int | None, tau: float | None, use_abs: bool) -> nx.Graph:
    """Undirected weighted graph from a correlation matrix (Top-K union or threshold)."""
    if k is None and tau is None:
        raise ValueError("clustering_metrics requires exactly one of k or tau")
    g = nx.Graph()
    g.add_nodes_from(corr.index)
    a = corr.values
    cols = list(corr.index)
    if k is not None:
        nbrs = top_k_neighbors(corr, k, use_abs)
        for node, ms in nbrs.items():
            for m in ms:
                w = abs(corr.loc[node, m])
                if not np.isnan(w):  # a node with <k finite corrs yields NaN neighbours
                    g.add_edge(node, m, weight=w)
    else:
        n = len(cols)
        for i in range(n):
            for j in range(i + 1, n):
                w = abs(a[i, j])
                if not np.isnan(w) and w > tau:
                    g.add_edge(cols[i], cols[j], weight=w)
    return g


def clustering_metrics(
    corr: pd.DataFrame, k: int | None = None, tau: float | None = None, use_abs: bool = True
) -> dict:
    """Graph-clustering diagnostics for a correlation matrix (plan section 23).

    Build an undirected graph from either Top-K neighbours (``k``) or a ``tau`` absolute
    threshold, then report component structure, degree, clustering coefficient and greedy
    modularity communities. Weights are ``|corr|``.
    """
    g = _edge_graph(corr, k, tau, use_abs)
    n = g.number_of_nodes()
    e = g.number_of_edges()
    comps = list(nx.connected_components(g))
    largest = max((len(c) for c in comps), default=0)
    if e > 0:
        communities = list(nx.community.greedy_modularity_communities(g, weight="weight"))
        modularity = float(nx.community.modularity(g, communities, weight="weight"))
    else:
        communities = [{node} for node in g.nodes]
        modularity = 0.0
    return {
        "n_nodes": int(n),
        "n_edges": int(e),
        "density": float(nx.density(g)) if n > 1 else 0.0,
        "avg_degree": float(2 * e / n) if n else 0.0,
        "n_connected_components": len(comps),
        "largest_component_frac": float(largest / n) if n else 0.0,
        "avg_clustering": float(nx.average_clustering(g)) if n else 0.0,
        "n_communities": len(communities),
        "modularity": modularity,
    }


def sector_purity(neighbors: dict[str, list[str]], same_sector: pd.DataFrame) -> float:
    """Fraction of directed (node, neighbour) pairs that are same-sector (plan section 50)."""
    hits, total = 0, 0
    for node, ms in neighbors.items():
        for m in ms:
            total += 1
            if int(same_sector.loc[node, m]) == 1:
                hits += 1
    return float(hits / total) if total else float("nan")


def mean_edge_strength(
    corr: pd.DataFrame, neighbors: dict[str, list[str]], use_abs: bool = True
) -> float:
    """Mean (absolute) correlation over selected Top-K edges (plan section 50)."""
    vals = []
    for node, ms in neighbors.items():
        for m in ms:
            v = corr.loc[node, m]
            if not np.isnan(v):
                vals.append(abs(v) if use_abs else v)
    return float(np.mean(vals)) if vals else float("nan")


def snapshots_to_long(
    snaps: list[tuple[pd.Timestamp, pd.DataFrame]], value_name: str
) -> pd.DataFrame:
    """Long-format (date, source, target, value) table of unique unordered pairs.

    Used to assemble the multi-window ``dynamic_edge_features`` panel (plan sections 18/46).
    """
    rows = []
    for date, corr in snaps:
        cols = list(corr.index)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                rows.append((date, cols[i], cols[j], corr.iloc[i, j]))
    return pd.DataFrame(rows, columns=["date", "source", "target", value_name])


def multi_window_edge_panel(
    panels: dict[str, pd.DataFrame],
    windows,
    ends,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Long-format multi-window rolling edge panel (plan sections 18/46), leakage-safe.

    For each feature panel in ``panels`` (all sharing ``index``) and each ``window`` in
    ``windows``, compute trailing-window correlation snapshots at the row positions in
    ``ends`` and emit column ``{feature}_{window}``; the panels are outer-merged on
    (date, source, target). Every snapshot uses rows ``[end-window+1 .. end]`` only and is
    asserted not to look past its ``end`` date. Returns an empty-but-typed frame when no
    end position yields a valid snapshot (e.g. a panel shorter than the largest window).
    """
    dyn: pd.DataFrame | None = None
    for w in windows:
        for fname, fpanel in panels.items():
            snaps = []
            for end in ends:
                block = fpanel.iloc[end - w + 1 : end + 1]
                if block.dropna(how="any").shape[0] >= max(10, w // 2):
                    d = index[end]
                    assert_snapshot_no_lookahead(block.index, d)
                    snaps.append((d, block.corr()))
            long = snapshots_to_long(snaps, f"{fname}_{w}")
            dyn = long if dyn is None else dyn.merge(
                long, on=["date", "source", "target"], how="outer"
            )
    if dyn is None or dyn.empty:
        return pd.DataFrame(columns=["date", "source", "target"])
    return dyn
