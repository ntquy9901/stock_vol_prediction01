"""Tests for incremental-over-HAR node/edge ranking (leakage-safe)."""

import numpy as np
import pandas as pd

from graph_eda import incremental


def test_har_base_columns():
    s = pd.Series(np.arange(30.0))
    b = incremental.har_base(s)
    assert list(b.columns) == ["pk_d", "pk_w", "pk_m"]
    assert b["pk_d"].iloc[10] == 10.0


def test_incoming_neighbors_directed():
    mat = pd.DataFrame(
        [[0.0, 0.9, 0.1], [0.2, 0.0, 0.8], [0.7, 0.3, 0.0]],
        index=["A", "B", "C"], columns=["A", "B", "C"],
    )
    # incoming leaders of column A: rank mat[i, A] over i != A -> C(0.7) then B(0.2)
    nb = incremental._incoming_neighbors(mat, k=1, use_abs=True)
    assert nb["A"] == ["C"]


def _panel(n=400, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    drv = np.abs(rng.standard_normal(n)) + 0.5
    dep = np.empty(n)
    dep[0] = 0.5
    for t in range(1, n):
        # DEP's FUTURE depends on the driver's CURRENT value -> a real cross-stock edge
        dep[t] = 0.3 * dep[t - 1] + 0.6 * drv[t - 1] + 0.05 * rng.standard_normal()
    dep = np.abs(dep)
    return pd.DataFrame({"DRV": drv, "DEP": dep}, index=idx)


def test_rank_candidates_detects_useful_edge_over_har():
    pk = _panel()
    n = len(pk)
    tr = np.arange(n) < 280
    te = np.arange(n) >= 340
    # edge "good": DEP's neighbour is DRV (the true driver); DRV has no neighbour
    edge_neighbors = {"good": ({"DEP": ["DRV"], "DRV": []}, pk)}
    node_features = {"own_drv_state": pk}  # a benign own-feature candidate
    node_rank, edge_rank = incremental.rank_candidates(
        pk, node_features, edge_neighbors, market=pk.median(axis=1),
        train_mask=tr, test_mask=te, horizon=1,
    )
    assert set(["edge_definition", "gain_over_har_pct", "gain_over_market_pct",
                "sign_p_over_market", "n_stocks"]) <= set(edge_rank.columns)
    row = edge_rank[edge_rank["edge_definition"] == "good"].iloc[0]
    # the true driver edge must lower RMSE vs HAR-only
    assert row["gain_over_har_pct"] > 0
    assert set(["node_feature", "rmse_gain_pct", "win_rate", "sign_p"]) <= set(node_rank.columns)


def test_evaluate_empty_when_insufficient_rows():
    pk = _panel(n=40)  # too few rows -> every stock skipped -> {}
    n = len(pk)
    tr = np.arange(n) < 30
    te = np.arange(n) >= 35
    base = {j: incremental.har_base(pk[j]) for j in pk.columns}
    out = incremental._evaluate(pk, base, {}, tr, te, horizon=1)
    assert out == {}
