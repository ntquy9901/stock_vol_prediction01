"""Tests for the deep+HARQ equal-weight stack: alignment, val-fit weight selection, and a real-data smoke
that the equal-weight stack beats HAR-X at h5."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import deep_harq_stack as S  # noqa: E402


def test_align_intersects_common_keys():
    harx = {("A", "d1"): (1.0, 0.9), ("A", "d2"): (2.0, 1.8), ("B", "d1"): (3.0, 2.9)}
    harxq = {("A", "d1"): (1.0, 0.8), ("A", "d2"): (2.0, 1.9)}          # only A rows
    volga = {("A", "d1"): 0.85, ("A", "d2"): 1.95, ("C", "d9"): 5.0}     # only A/C
    y, dates, hx, hq, vg = S._align(harx, harxq, volga)
    assert len(y) == 2 and set(dates) == {"d1", "d2"}                    # intersection = A/d1, A/d2
    assert np.allclose(sorted(vg), [0.85, 1.95])


def test_fit_w_selects_val_optimal():
    """When HAR-X-Q is exact on val and VolGA is biased, the val-fit weight must lean to HAR-X-Q (w small)."""
    y = np.full(200, 1e-3)
    harxq = {("A", str(i)): (y[i], 1e-3) for i in range(200)}            # perfect
    volga = {("A", str(i)): 5e-3 for i in range(200)}                    # 5x over-forecast
    harx = {("A", str(i)): (y[i], 2e-3) for i in range(200)}
    w = S._fit_w(harx, harxq, volga, 1e-8)
    assert w <= 0.1                                                      # should trust HAR-X-Q, not VolGA


def test_blend_endpoints():
    hq = np.array([1.0, 2.0]); vg = np.array([3.0, 4.0])
    assert np.allclose(np.maximum(0.0 * vg + 1.0 * hq, 1e-8), hq)        # w=0 -> HAR-X-Q
    assert np.allclose(np.maximum(1.0 * vg + 0.0 * hq, 1e-8), vg)        # w=1 -> VolGA
    assert np.allclose(0.5 * vg + 0.5 * hq, np.array([2.0, 3.0]))        # equal weight


def test_dict_by_ticker_maps_node_index():
    pred = np.array([[0.5, 0.6]]); y = np.array([[1.0, 2.0]]); tm = np.array([[True, True]])
    d = S._dict_by_ticker(pred, y, tm, ["2024-01-02"], ["ACB", "FPT"], 2)
    assert ("ACB", "2024-01-02") in d and ("FPT", "2024-01-02") in d    # node index -> ticker
    assert d[("FPT", "2024-01-02")] == (2.0, 0.6)


def test_require_overlap_guards_stale_cells():
    assert S._require_overlap(90, 100) == 90                             # enough overlap -> ok
    with pytest.raises(ValueError):
        S._require_overlap(0, 100)                                       # empty intersection -> loud
    with pytest.raises(ValueError):
        S._require_overlap(10, 100)                                      # <50% survived -> loud


@pytest.mark.smoke
def test_smoke_equal_stack_beats_harx_h5():
    import glob
    if not glob.glob(str(S.REPO / "results" / "qlike_anchor" / "cells" / "cells_vn100_qlike_none_h5.parquet")):
        pytest.skip("vn100 h5 canonical cells not present")
    r = S.run("vn100", 5, out=str(S.REPO / "results" / "deep_harq_stack" / "_smoke_vn100_h5.json"))
    eq = r["stack"]["equal_0.5"]
    assert 0.0 <= r["w_fit_on_val"] <= 1.0
    assert eq["qlike"] < r["qlike"]["HAR-X"]                             # stack beats HAR-X
    assert eq["qlike"] < r["qlike"]["VolGA"] and eq["qlike"] < r["qlike"]["HAR-X-Q"]  # beats both members
    assert eq["beats_HARX_sig"]                                          # significantly at h5
