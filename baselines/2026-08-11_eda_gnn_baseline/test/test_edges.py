"""Leakage + structure tests for the directed volume->PK lead-lag adjacency."""

import numpy as np
import pandas as pd

import edges
from data import SplitFrames, chronological_split


def _augmented_frames(seed: int = 0, n: int = 400, tickers=("AAA", "BBB", "CCC", "DDD")) -> SplitFrames:
    """Synthetic augmented frames with parkinson_volatility + market_pk + volume_zscore_20 columns."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    frames = {}
    for offset, ticker in enumerate(tickers):
        pk = np.abs(rng.normal(1e-3, 3e-4, size=n)) + 1e-5
        vshock = rng.normal(0.0, 1.0, size=n)
        frame = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"), "parkinson_volatility": pk,
            "market_pk": np.sqrt(pk), "volume_zscore_20": vshock,
        })
        frames[ticker] = chronological_split(frame)
    ticker_to_id = {ticker: index for index, ticker in enumerate(tickers)}
    return SplitFrames(frames=frames, ticker_to_id=ticker_to_id)


def test_adjacency_has_self_loops_and_bounded_degree():
    frames = _augmented_frames()
    adjacency = edges.build_vol2pk_adjacency(frames, frames.ticker_to_id, top_k=2)
    node_count = len(frames.ticker_to_id)
    assert adjacency.shape == (node_count, node_count)
    assert np.allclose(np.diag(adjacency), 1.0)  # self-loop on every node
    off_diagonal = adjacency.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    # each target keeps at most top_k incoming sources
    assert (np.count_nonzero(off_diagonal, axis=1) <= 2).all()
    assert np.isfinite(adjacency).all()


def test_adjacency_uses_train_dates_only():
    """Altering post-train (val/test) rows must not change the frozen adjacency (no leakage)."""

    frames = _augmented_frames(seed=1)
    base = edges.build_vol2pk_adjacency(frames, frames.ticker_to_id, top_k=2)

    bumped = {t: {s: frames.frames[t][s].copy() for s in frames.frames[t]} for t in frames.frames}
    for ticker in bumped:
        for split in ("val", "test"):
            part = bumped[ticker][split]
            part["volume_zscore_20"] = 999.0
            part["parkinson_volatility"] = 5.0
    bumped_frames = SplitFrames(frames=bumped, ticker_to_id=dict(frames.ticker_to_id))
    after = edges.build_vol2pk_adjacency(bumped_frames, bumped_frames.ticker_to_id, top_k=2)
    assert np.array_equal(base, after)


def test_adjacency_no_leak_with_heterogeneous_histories():
    """Per-ticker splits: a SHORT-history ticker's val/test overlaps others' train calendar dates.

    Bumping every ticker's own val/test rows must still not move the frozen topology -- guards the
    per-ticker (not global-boundary) train restriction (review finding H1).
    """

    rng = np.random.default_rng(7)
    dates_long = pd.bdate_range("2015-01-01", periods=500)
    dates_short = pd.bdate_range("2015-01-01", periods=250)  # ends inside the long ticker's train
    frames = {}
    for ticker, dates in (("AAA", dates_long), ("BBB", dates_long), ("CCC", dates_short),
                          ("DDD", dates_short)):
        n = len(dates)
        pk = np.abs(rng.normal(1e-3, 3e-4, size=n)) + 1e-5
        frame = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "parkinson_volatility": pk,
                              "market_pk": np.sqrt(pk), "volume_zscore_20": rng.normal(0, 1, size=n)})
        frames[ticker] = chronological_split(frame)
    split_frames = SplitFrames(frames=frames, ticker_to_id={t: i for i, t in enumerate(frames)})
    base = edges.build_vol2pk_adjacency(split_frames, split_frames.ticker_to_id, top_k=2)

    bumped = {t: {s: split_frames.frames[t][s].copy() for s in split_frames.frames[t]} for t in frames}
    for ticker in bumped:
        for split in ("val", "test"):
            bumped[ticker][split]["volume_zscore_20"] = 999.0
            bumped[ticker][split]["parkinson_volatility"] = 5.0
    bumped_frames = SplitFrames(frames=bumped, ticker_to_id=dict(split_frames.ticker_to_id))
    after = edges.build_vol2pk_adjacency(bumped_frames, bumped_frames.ticker_to_id, top_k=2)
    assert np.array_equal(base, after)


def test_swap_adjacency_masks_absent_nodes():
    from data import GraphNode, GraphSnapshot, GraphManifest

    node_count = 3
    adjacency = np.array([[1, 0.5, 0], [0.3, 1, 0.2], [0, 0.4, 1]], dtype=np.float32)
    nodes = tuple(GraphNode(i, f"T{i}", "train", 1.0, 0.0) for i in range(node_count))
    presence = np.array([1, 0, 1], dtype=np.int8)  # middle node absent
    price = np.zeros((node_count, 4, 2), dtype=np.float32)
    news = np.zeros((node_count, 4, 0), dtype=np.float32)
    mask = np.zeros((node_count, 4), dtype=np.int8)
    snapshot = GraphSnapshot("2020-01-01", "train", (), nodes, price, news, mask,
                             np.eye(node_count, dtype=np.float32), presence)
    graph = GraphManifest((snapshot,), {"T0": 0, "T1": 1, "T2": 2}, "2020-01-01", "2020-01-02", {})
    swapped = edges.swap_adjacency(graph, adjacency)
    result = swapped.snapshots[0].adjacency
    assert result[1, :].sum() == 0.0 and result[:, 1].sum() == 0.0  # absent row/col zeroed
    assert result[0, 0] == 1.0 and result[2, 2] == 1.0  # present self-loops kept
    assert result[0, 2] == 0.0  # T0 had no edge from T2 originally
