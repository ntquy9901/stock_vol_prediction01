"""Tests for the pure LSTM-vs-HAR-X QLIKE gap decomposition (parquet I/O is pragma'd)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_harx_cell_gap as G  # noqa: E402


def test_per_cell_qlike_zero_at_perfect_forecast():
    y = np.array([1e-3, 2e-3, 5e-4])
    assert np.allclose(G.per_cell_qlike(y, y), 0.0)              # y==f -> QLIKE 0
    # under-forecast (f<y) costs more than the symmetric over-forecast by the same ratio
    assert G.per_cell_qlike([1e-2], [1e-3])[0] > G.per_cell_qlike([1e-3], [1e-2])[0]


def test_gap_decomposition_tail_dominates():
    # 99 cells where LSTM == HAR-X (zero gap) + 1 catastrophic LSTM under-forecast of a spike
    y = np.array([1e-3] * 99 + [1e-2])
    harx = np.array([1e-3] * 99 + [8e-3])       # HAR-X tracks the spike
    lstm = np.array([1e-3] * 99 + [1e-4])       # LSTM collapses on the spike
    out = G.gap_decomposition(y, harx, lstm, top_frac=0.01)
    assert out["n_cells"] == 100
    assert out["qlike_lstm"] > out["qlike_harx"]                 # LSTM worse in aggregate
    assert out["gap_median"] == 0.0                             # typical cell identical
    assert out["top_share_of_gap"] >= 0.99                       # the 1 tail cell carries ~all the gap
    assert out["qlike_lstm_ex_worst"] <= out["qlike_harx_ex_worst"] + 1e-12   # ex-tail, LSTM not worse
    assert 0.0 <= out["lstm_underforecast_rate"] <= 1.0


def test_gap_decomposition_zero_gap_no_nan():
    y = np.array([1e-3, 2e-3]); f = np.array([1e-3, 2e-3])
    out = G.gap_decomposition(y, f, f)
    assert out["gap"] == 0.0 and np.isnan(out["top_share_of_gap"])   # total gap 0 -> share is nan (guarded)


def test_worst_cells_orders_descending():
    assert G.worst_cells([0.1, 5.0, 2.0, 9.0], 2) == [3, 1]         # highest QLIKE first
    assert G.worst_cells([1.0], 0) == []


def test_top_by_group_sums_and_ranks():
    keys = ["A", "B", "A", "C"]; vals = [1.0, 10.0, 2.0, 4.0]
    assert G.top_by_group(keys, vals, 2) == [("B", 10.0), ("C", 4.0)]   # B=10, C=4, A=3


def test_anchored_forecast_falls_back_and_clips():
    harx = np.array([1.27e-4, 2.0e-4])
    assert np.allclose(G.anchored_forecast(harx, [0.0, 0.0]), harx)          # z=0 -> exactly HAR-X (no collapse)
    up = G.anchored_forecast(harx, [0.3, 0.3]); assert np.all(up > harx)      # positive z scales up
    # a huge negative z is clipped, so the forecast cannot collapse far below HAR-X
    floor_side = G.anchored_forecast(harx, [-10.0, -10.0], clip=0.5)
    assert np.allclose(floor_side, harx * np.exp(-0.5))


def test_qlike_excluding_top_y_is_model_agnostic():
    # exclude_frac=0 -> full mean; excluding the top-y cell drops the same index for any forecast
    y = np.array([1e-3, 1e-3, 1e-3, 1.0])       # last cell is a spike (top by y)
    f = np.array([1e-3, 1e-3, 1e-3, 1e-3])
    full = G.qlike_excluding_top_y(y, f, 0.0)
    ex = G.qlike_excluding_top_y(y, f, 0.25)     # drop top 25% (the spike) -> only perfect cells remain
    assert full > ex and abs(ex) < 1e-9          # remaining cells are perfect -> ~0
