"""TRAIN-frozen directed volume->PK lead-lag adjacency (the EDA-recommended edge).

The EDA's ``edge_vol2pk_dir``: for target stock ``j`` the Top-K source stocks ``i`` are those whose
volume shock today most strongly leads ``j``'s Parkinson volatility tomorrow, ranked by
``|corr(vshock_i(t), sqrt(PK)_j(t+1))|`` estimated on TRAIN dates only. The neighbour identity and
weights are frozen from that train matrix and reused for every snapshot (the "dynamic" element is the
window-varying node features, not the topology -- freezing the topology on train is the leakage-safe
choice, EDA plan sections 31/55).

Message-passing convention (``models._ResidualMessagePassing``): ``out[node] = sum_k
softmax(A[node, k]) * feat[k]``, so ``A[j, i]`` is the edge from source ``i`` into target ``j``. We set
``A[j, i] = tv2p[i, j]`` for ``i`` in ``j``'s Top-K sources, plus a self-loop ``A[j, j] = 1``.

Tickers without a volume series (e.g. LPB) have a constant/NaN vshock column and can never be a
source; they remain valid targets via the self-loop.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from data import GraphManifest, GraphSnapshot
from features import VOLUME_ZSCORE_COLUMN

_MIN_PAIRS = 30


def _train_panels(augmented_frames) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide date x ticker vshock and sqrt(PK) panels over each ticker's OWN train split.

    Splits are per-ticker chronological, so using each ticker's own ``["train"]`` rows (rather than
    a single global date boundary applied to the full series) guarantees no ticker contributes a
    validation/test observation to the frozen lead-lag topology -- the leakage-safe estimate.
    """

    vshock: dict[str, pd.Series] = {}
    pk_vol: dict[str, pd.Series] = {}
    for ticker in augmented_frames.frames:
        train = augmented_frames.frames[ticker]["train"]
        index = pd.DatetimeIndex(pd.to_datetime(train["date"]))
        pk_vol[ticker] = pd.Series(
            np.sqrt(train["parkinson_variance"].to_numpy(dtype=float)), index=index
        )
        vshock[ticker] = pd.Series(train[VOLUME_ZSCORE_COLUMN].to_numpy(dtype=float), index=index)
    return pd.DataFrame(vshock).sort_index(), pd.DataFrame(pk_vol).sort_index()


def _cross_lead_lag(source_wide: pd.DataFrame, target_wide: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """Directed L[i, j] = corr(source_i(t), target_j(t+k)) over pairwise-complete rows."""

    source = source_wide.iloc[:-k]
    target = target_wide.shift(-k).iloc[:-k]
    out = pd.DataFrame(index=source_wide.columns, columns=target_wide.columns, dtype=float)
    for target_ticker in target_wide.columns:
        future = target[target_ticker]
        for source_ticker in source_wide.columns:
            present = source[source_ticker]
            mask = present.notna() & future.notna()
            if int(mask.sum()) < _MIN_PAIRS:
                out.loc[source_ticker, target_ticker] = np.nan
                continue
            a = present[mask].to_numpy(dtype=float)
            b = future[mask].to_numpy(dtype=float)
            if a.std() == 0.0 or b.std() == 0.0:
                out.loc[source_ticker, target_ticker] = np.nan
                continue
            out.loc[source_ticker, target_ticker] = float(np.corrcoef(a, b)[0, 1])
    return out


def build_vol2pk_adjacency(
    augmented_frames, ticker_to_id: Mapping[str, int], top_k: int = 5
) -> np.ndarray:
    """Fixed directed adjacency: each target's Top-K volume->PK lead-lag sources (train-frozen)."""

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    vshock_wide, pk_vol_wide = _train_panels(augmented_frames)
    tv2p = _cross_lead_lag(vshock_wide, pk_vol_wide, 1)
    node_count = len(ticker_to_id)
    adjacency = np.zeros((node_count, node_count), dtype=np.float32)
    for target_ticker in pk_vol_wide.columns:
        target_id = ticker_to_id[target_ticker]
        column = tv2p[target_ticker].drop(labels=[target_ticker], errors="ignore").dropna()
        if column.empty:
            continue
        top_sources = column.abs().sort_values(ascending=False).head(top_k).index
        for source_ticker in top_sources:
            adjacency[target_id, ticker_to_id[source_ticker]] = float(tv2p.loc[source_ticker, target_ticker])
    np.fill_diagonal(adjacency, 1.0)  # self-loop keeps every node valid under presence masking
    return adjacency


def swap_adjacency(graph: GraphManifest, adjacency: np.ndarray) -> GraphManifest:
    """Replace every snapshot's adjacency with the presence-masked fixed ``adjacency``.

    Absent nodes' rows and columns are zeroed per snapshot (they neither send nor receive), while
    present nodes keep their self-loop, so each present node always has at least a self-loop.
    """

    node_count = adjacency.shape[0]
    if adjacency.shape != (node_count, node_count):
        raise ValueError("adjacency must be square over the node vocabulary")
    snapshots: list[GraphSnapshot] = []
    for snapshot in graph.snapshots:
        presence = (snapshot.presence_mask if snapshot.presence_mask is not None
                    else np.ones(node_count, dtype=np.int8))
        masked = adjacency.copy()
        absent = np.flatnonzero(np.asarray(presence) == 0)
        masked[absent, :] = 0.0
        masked[:, absent] = 0.0
        snapshots.append(GraphSnapshot(
            snapshot.target_date, snapshot.split, snapshot.input_dates, snapshot.nodes,
            snapshot.x_price, snapshot.x_news, snapshot.news_mask, masked, snapshot.presence_mask,
        ))
    hashes = dict(graph.hashes)
    hashes["edge_construction"] = "vol2pk_dir_top5_frozen"
    return GraphManifest(tuple(snapshots), dict(graph.ticker_to_id), graph.train_end_date,
                         graph.val_end_date, hashes)
