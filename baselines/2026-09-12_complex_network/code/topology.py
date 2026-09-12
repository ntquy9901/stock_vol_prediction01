"""Complex-network topology features (Experiment A + B shared).

Builds time-varying financial networks from a blend of return- and volume-correlation matrices over a
trailing window, filters by a threshold, and extracts 6 GLOBAL topology metrics. ``global_feats``
and the window logic are copied faithfully from ``scripts/eda/complex_network_gbm.py`` (design doc Sections 4-5),
parameterised through :mod:`config`. Every window uses only rows strictly before its date (causal). The volume
correlation uses REAL ``ln(volume)`` from the raw OHLCV (Refinement 1), not ``volume_zscore_22``.
"""
import numpy as np
import pandas as pd
import networkx as nx

import config
import volume_io


def global_feats(C):
    """The 6 global topology metrics of the Threshold graph from combined correlation matrix ``C``:
    density, average degree, average clustering, average weight, diameter (largest component), average
    betweenness centrality. (Eigenvector centrality from the original 7-metric set was dropped -- ~0 mutual
    information with the target and perfectly collinear with the others; see config.TOPO.)"""
    n = C.shape[0]
    absC = np.abs(C); np.fill_diagonal(absC, 0.0)
    A = (absC > config.THR)
    G = nx.from_numpy_array(A.astype(float))
    dens = nx.density(G)
    avg_deg = float(np.mean([d for _, d in G.degree()]))
    clus = nx.average_clustering(G)
    ew = absC[A]
    avg_w = float(ew.mean()) if ew.size else 0.0
    if G.number_of_edges():
        H = G.subgraph(max(nx.connected_components(G), key=len))
        diam = float(nx.diameter(H)) if H.number_of_nodes() > 1 else 0.0
    else:
        diam = 0.0
    betw = float(np.mean(list(nx.betweenness_centrality(G).values()))) if n > 2 else 0.0
    return [dens, avg_deg, clus, avg_w, diam, betw]


def _return_panel(frames):
    """Wide daily-return panel (date x ticker), sorted by date, from the per-ticker frames."""
    return pd.DataFrame(
        {tk: d.set_index("date")["daily_return"] for tk, d in frames.items()}
    ).sort_index()


def build_topo_windows(frames, market, alpha, win):
    """One topology 6-vector per sliding window, plus the per-window common-ticker count ``N``.

    Returns ``(F, n_by_window)``: ``F`` is a DataFrame indexed by the window-end date ``d0`` with columns
    :data:`config.TOPO` (ONE row per window, no forward-fill); ``n_by_window`` is a Series (indexed by the same
    ``d0``) of the number of common tickers used in each window. Return matrix from ``daily_return``; volume
    matrix = REAL ``ln(volume)`` from :func:`volume_io.load_log_volume`, aligned to the return calendar. For each
    step ``i`` the window ``iloc[i-win:i]`` is strictly before ``d0 = dates[i]``; tickers with fewer than
    ``int(win*WIN_MIN_FRAC)`` observations in EITHER matrix are dropped; the ``common`` intersection is used for
    both correlations; a window with fewer than ``MIN_COMMON_TICKERS`` common tickers is skipped.
    """
    ret = _return_panel(frames)
    dates = ret.index
    lnvol = volume_io.load_log_volume(market, list(frames)).reindex(dates)  # align volume to the return calendar
    rows = {}
    n_by_window = {}
    for i in range(win, len(dates), config.STEP):
        d0 = dates[i]
        Rw = ret.iloc[i - win:i].dropna(axis=1, thresh=int(win * config.WIN_MIN_FRAC))
        Vw = lnvol.iloc[i - win:i].dropna(axis=1, thresh=int(win * config.WIN_MIN_FRAC))
        common = Rw.columns.intersection(Vw.columns)            # same tickers to combine the two matrices
        if len(common) < config.MIN_COMMON_TICKERS:
            continue
        rc = np.nan_to_num(Rw[common].corr().to_numpy()); vc = np.nan_to_num(Vw[common].corr().to_numpy())
        combined = alpha * rc + (1 - alpha) * vc                # paper eq. (1): alpha-weighted combined corr
        rows[d0] = global_feats(combined)
        n_by_window[d0] = int(len(common))
    F = pd.DataFrame.from_dict(rows, orient="index", columns=config.TOPO)
    return F, pd.Series(n_by_window, dtype=float, name="n_tickers")


def build_topo_daily(frames, market, alpha, win):
    """Daily topology panel for Experiment B: window rows reindexed to every date and forward-filled
    (topology moves slowly; each date carries the most recent strictly-past window's metrics)."""
    ret = _return_panel(frames)
    F, _ = build_topo_windows(frames, market, alpha, win)
    return F.reindex(ret.index).ffill()
