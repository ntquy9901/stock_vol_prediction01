"""Plan sections 19/37/43/45/65: Top-K neighbours, stability, turnover, controls."""

import numpy as np
import pandas as pd

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
