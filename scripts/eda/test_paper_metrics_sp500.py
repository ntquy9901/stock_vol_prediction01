"""Unit tests for the pure helpers of ``paper_metrics_sp500`` (the SP500 all-metrics/oracle driver).

Covers ``all_metrics``, ``adj_sector`` and ``nb_col`` with tiny synthetic frames -- the ``main()`` walk-forward
entry driver is ``# pragma: no cover`` (needs the real panel + folds), mirroring the sibling ``full_matrix``/
``vn_gbm_graph_stage1`` drivers. Importing the module also exercises its path-bootstrap + dependency imports,
which is the fresh-clone regression this file guards (Colab clone must resolve every import)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_metrics_sp500 as PM  # noqa: E402


def test_all_metrics_keys_and_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.1, 1.9, 3.2, 3.7])
    m = PM.all_metrics(y, p)
    assert set(m) == {"mse", "rmse", "mae", "r2", "qlike"}
    assert all(np.isfinite(v) for v in m.values())
    assert m["qlike"] > 0.0


def test_adj_sector_row_normalised_same_sector():
    tickers = ["A", "B", "C"]                 # A,B share sector 1; C alone in sector 2; C_missing hits .get default
    sect = {"A": 1, "B": 1}                    # C absent -> get(...,-1); alone -> empty neighbour row
    W = PM.adj_sector(tickers, sect)
    assert W.shape == (3, 3)
    assert W[0, 1] == 1.0 and W[1, 0] == 1.0   # A<->B are each other's only same-sector neighbour
    assert W[0, 0] == 0.0                       # no self edge
    assert W[2].sum() == 0.0                    # C has no same-sector neighbour -> zero row


def test_nb_col_handles_missing_pair_and_maps_positions():
    W = PM.adj_sector(["A", "B", "C"], {"A": 1, "B": 1, "C": 2})
    d1, d2 = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")
    # (d2, C) intentionally absent -> pivot produces a NaN the row-mean fill must handle
    fold = pd.DataFrame({
        "date": [d1, d1, d1, d2, d2],
        "ticker": ["A", "B", "C", "A", "B"],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    out = PM.nb_col(fold, ["A", "B", "C"], W, "y")
    assert out.shape == (5,)
    assert np.all(np.isfinite(out))
    # A's neighbour aggregate on d1 = 1.0 * B's value (row-normalised same-sector) = 2.0
    a_d1 = out[(fold["ticker"] == "A") & (fold["date"] == d1)]
    assert a_d1[0] == 2.0
