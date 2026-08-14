"""Plan sections 19/37/43/45/65: Top-K neighbours, stability, turnover, controls."""

import numpy as np
import pandas as pd
import pytest

from graph_eda import graphs


def _corr(vals):
    idx = ["A", "B", "C", "D"]
    return pd.DataFrame(vals, index=idx, columns=idx)


def test_top_k_neighbors_excludes_self_and_ranks_by_abs():
    c = _corr(
        [
            [1.0, 0.9, -0.8, 0.1],
            [0.9, 1.0, 0.2, 0.0],
            [-0.8, 0.2, 1.0, 0.3],
            [0.1, 0.0, 0.3, 1.0],
        ]
    )
    nb = graphs.top_k_neighbors(c, k=2, use_abs=True)
    assert "A" not in nb["A"]
    assert nb["A"] == ["B", "C"]  # 0.9 then |-0.8|


def test_jaccard_bounds():
    assert graphs.jaccard(["A", "B"], ["A", "B"]) == 1.0
    assert graphs.jaccard(["A", "B"], ["C", "D"]) == 0.0
    assert graphs.jaccard([], []) == 1.0


def test_rolling_snapshots_are_trailing_only():
    idx = pd.date_range("2021-01-01", periods=40, freq="B")
    rng = np.random.default_rng(0)
    w = pd.DataFrame(rng.standard_normal((40, 4)), index=idx, columns=list("ABCD"))
    snaps = graphs.rolling_snapshots(w, window=10, step=5)
    d0, _ = snaps[0]
    # first snapshot ends at the 10th row (index position 9), trailing window only
    assert d0 == idx[9]


def test_edge_turnover_zero_when_static():
    c = _corr(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ]
    )
    snaps = [(pd.Timestamp("2021-01-01"), c), (pd.Timestamp("2021-02-01"), c)]
    to = graphs.edge_turnover(snaps, k=1)
    assert to["edge_turnover"].iloc[0] == 0.0


def test_random_matched_neighbors_count_and_no_self():
    nodes = list("ABCDE")
    nb = graphs.random_matched_neighbors(nodes, k=3, seed=1)
    for n, lst in nb.items():
        assert n not in lst
        assert len(lst) == 3


def test_clustering_metrics_two_blocks_topk():
    # two disconnected blocks {A,B} and {C,D}: 2 communities, 2 components
    c = _corr(
        [
            [1.0, 0.9, 0.0, 0.0],
            [0.9, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.9],
            [0.0, 0.0, 0.9, 1.0],
        ]
    )
    m = graphs.clustering_metrics(c, k=1)
    assert m["n_nodes"] == 4
    assert m["n_edges"] == 2
    assert m["n_connected_components"] == 2
    assert m["largest_component_frac"] == 0.5
    assert m["n_communities"] >= 2
    assert 0.0 <= m["modularity"] <= 1.0
    assert 0.0 <= m["density"] <= 1.0


def test_clustering_metrics_threshold_and_empty():
    c = _corr(
        [
            [1.0, 0.1, 0.1, 0.1],
            [0.1, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.1],
            [0.1, 0.1, 0.1, 1.0],
        ]
    )
    # tau=0.5 keeps no off-diagonal edge -> empty graph, all singletons
    m = graphs.clustering_metrics(c, tau=0.5)
    assert m["n_edges"] == 0
    assert m["n_connected_components"] == 4
    assert m["avg_clustering"] == 0.0


def test_sector_purity_fraction_same_sector():
    same = pd.DataFrame(
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    nb = {"A": ["B"], "B": ["A"], "C": ["A"]}  # 2 same-sector of 3 pairs
    assert graphs.sector_purity(nb, same) == 2 / 3
    assert np.isnan(graphs.sector_purity({}, same))


def test_mean_edge_strength_uses_abs():
    c = _corr(
        [
            [1.0, 0.4, -0.8, 0.0],
            [0.4, 1.0, 0.0, 0.0],
            [-0.8, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    nb = {"A": ["C"], "B": ["A"]}  # |-0.8|, |0.4| -> mean 0.6
    assert abs(graphs.mean_edge_strength(c, nb) - 0.6) < 1e-9
    assert np.isnan(graphs.mean_edge_strength(c, {}))


def test_snapshots_to_long_shape_and_pairs():
    c1 = _corr(np.eye(4) + 0.2 * (1 - np.eye(4)))
    c2 = _corr(np.eye(4) + 0.5 * (1 - np.eye(4)))
    snaps = [(pd.Timestamp("2021-01-01"), c1), (pd.Timestamp("2021-02-01"), c2)]
    long = graphs.snapshots_to_long(snaps, "pk_corr_20")
    # 4 nodes -> 6 unique unordered pairs per snapshot, 2 snapshots -> 12 rows
    assert len(long) == 12
    assert set(long.columns) == {"date", "source", "target", "pk_corr_20"}
    assert (long["source"] < long["target"]).all()


def test_clustering_metrics_requires_k_or_tau():
    c = _corr(np.eye(4) + 0.2 * (1 - np.eye(4)))
    with pytest.raises(ValueError):
        graphs.clustering_metrics(c)  # neither k nor tau given


def test_clustering_metrics_skips_nan_topk_edges():
    # A has only one finite corr (to B); its 2nd Top-K neighbour is a NaN pair, which must
    # be skipped so modularity stays finite (review F3)
    c = _corr(
        [
            [1.0, 0.9, np.nan, np.nan],
            [0.9, 1.0, 0.8, np.nan],
            [np.nan, 0.8, 1.0, 0.7],
            [np.nan, np.nan, 0.7, 1.0],
        ]
    )
    m = graphs.clustering_metrics(c, k=2)
    assert not np.isnan(m["modularity"])
    assert m["n_edges"] >= 1


def test_mean_edge_strength_ignores_nan_edges():
    c = _corr(
        [
            [1.0, np.nan, np.nan, 0.0],
            [np.nan, 1.0, 0.0, 0.0],
            [np.nan, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # both selected edges are NaN -> nan, without a RuntimeWarning (review F4)
    assert np.isnan(graphs.mean_edge_strength(c, {"A": ["B"], "B": ["A"]}))


def test_multi_window_edge_panel_windows_and_empty():
    idx = pd.date_range("2021-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    panels = {
        "pk_corr": pd.DataFrame(rng.standard_normal((200, 3)), index=idx, columns=list("ABC")),
        "return_corr": pd.DataFrame(rng.standard_normal((200, 3)), index=idx, columns=list("ABC")),
    }
    ends = list(range(119, 200, 21))
    panel = graphs.multi_window_edge_panel(panels, (20, 60, 120), ends, idx)
    for col in ("pk_corr_20", "pk_corr_60", "pk_corr_120", "return_corr_60"):
        assert col in panel.columns
    assert len(panel) > 0
    # no valid end position (empty ends) -> typed-but-empty frame, no crash (review BH4)
    empty = graphs.multi_window_edge_panel(panels, (20, 60, 120), [], idx)
    assert empty.empty
    assert list(empty.columns) == ["date", "source", "target"]
