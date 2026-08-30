"""TDD tests for the NEW ETL cleaning functions (scripts/etl_audit/etl_cleaning.py).

Failing-first: written before the implementation. Each cleaning rule has a formula/behaviour test
(independent recompute, not a reuse of the implementation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "etl_audit"))

import etl_cleaning as C  # noqa: E402

_RTOL = 1e-5


def _frame(rows):
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def _oc_internally_consistent(df):
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    hi_oc, lo_oc = np.maximum(o, c), np.minimum(o, c)
    return bool(np.all((h >= hi_oc * (1 - _RTOL)) & (lo <= lo_oc * (1 + _RTOL)) & (h >= lo)))


def test_widen_range_makes_oc_internally_consistent():
    df = _frame([("2020-01-01", 13, 11, 9, 10, 100),    # open 13 > high 11
                 ("2020-01-02", 10, 11, 9, 7, 100)])    # close 7 < low 9
    assert not _oc_internally_consistent(df)
    out, info = C.widen_range(df)
    assert _oc_internally_consistent(out)
    assert info["n_widened"] == 2
    # widen sets high=max(h,o,c): row0 high -> 13; low=min(l,o,c): row1 low -> 7
    assert out.loc[0, "high"] == 13
    assert out.loc[1, "low"] == 7


def test_widen_range_preserves_parkinson_when_already_consistent():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100)])
    out, info = C.widen_range(df)
    assert info["n_widened"] == 0
    assert out.loc[0, "high"] == 11 and out.loc[0, "low"] == 9


def test_clip_oc_pulls_open_close_into_range():
    df = _frame([("2020-01-01", 13, 11, 9, 7, 100)])    # open 13>high, close 7<low
    out, info = C.clip_oc(df)
    assert out.loc[0, "open"] == 11 and out.loc[0, "close"] == 9
    assert info["n_clipped"] == 1


def test_swap_or_drop_high_low_swaps_transposition():
    # high<low but swapping high<->low yields valid geometry (O/C inside) -> swap, keep row.
    df = _frame([("2020-01-01", 10, 9, 11, 10, 100)])   # high 9 < low 11; swap -> high 11, low 9
    out, info = C.swap_or_drop_high_low(df)
    assert len(out) == 1
    assert out.loc[0, "high"] == 11 and out.loc[0, "low"] == 9
    assert info["n_swapped"] == 1 and info["n_dropped"] == 0


def test_swap_or_drop_high_low_drops_unfixable():
    # high<low and even after swap O/C fall outside -> drop.
    df = _frame([("2020-01-01", 100, 9, 11, 100, 100)])  # open/close 100 outside even swapped [9,11]
    out, info = C.swap_or_drop_high_low(df)
    assert len(out) == 0
    assert info["n_dropped"] == 1


def test_reconstruct_nonpositive_yields_positive_ohlc():
    df = _frame([("2020-01-01", 0, 11, 9, 10, 100),      # open 0 -> reconstruct
                 ("2020-01-02", 10, 11, -1, 10, 100)])   # low -1 -> reconstruct
    out, info = C.reconstruct_nonpositive(df)
    for k in ("open", "high", "low", "close"):
        assert np.all(out[k].to_numpy(float) > 0)
    assert info["n_reconstructed"] == 2
    # high=max(positive OHLC), low=min(positive OHLC); row0 positives {11,9,10} -> H=11,L=9
    assert out.loc[0, "high"] == 11 and out.loc[0, "low"] == 9


def test_reconstruct_nonpositive_drops_when_too_few_positive():
    df = _frame([("2020-01-01", 0, 0, 0, 5, 100)])       # only one positive value -> cannot reconstruct
    out, info = C.reconstruct_nonpositive(df)
    assert len(out) == 0
    assert info["n_dropped"] == 1


def test_backadjust_splits_removes_the_level_jump():
    # A clean 2:1 split-like doubling on 2020-01-03: close goes 10 -> 20 unadjusted.
    df = _frame([("2020-01-01", 10, 10.2, 9.8, 10, 100),
                 ("2020-01-02", 10, 10.2, 9.8, 10, 100),
                 ("2020-01-03", 20, 20.4, 19.6, 20, 100),   # +100% jump (unadjusted split)
                 ("2020-01-04", 20, 20.4, 19.6, 20, 100)])
    out, info = C.backadjust_splits(df, thresh=0.5)
    c = out["close"].to_numpy(float)
    r = np.diff(np.log(c))
    assert np.all(np.abs(r) < 0.5)          # the jump is gone
    assert info["n_adjusted"] == 1
    # pre-split rows scaled up by factor 2 -> close 10 becomes 20
    assert out.loc[0, "close"] == pytest.approx(20.0, rel=1e-6)


def test_backadjust_splits_noop_without_jump():
    df = _frame([("2020-01-01", 10, 10.2, 9.8, 10, 100),
                 ("2020-01-02", 10, 10.2, 9.8, 10.1, 100)])
    out, info = C.backadjust_splits(df, thresh=0.5)
    assert info["n_adjusted"] == 0
    assert out.loc[0, "close"] == 10


def test_cut_to_listing_drops_leading_backfill():
    rows = [("2020-01-01", 5, 5, 5, 5, 0), ("2020-01-02", 5, 5, 5, 5, 0),
            ("2020-01-03", 5, 6, 4, 5.5, 1000), ("2020-01-04", 5.5, 6, 5, 5.8, 1200)]
    df = _frame(rows)
    out, info = C.cut_to_listing(df)
    assert len(out) == 2
    assert str(out.iloc[0]["date"].date()) == "2020-01-03"
    assert info["n_cut"] == 2


def test_cut_to_listing_noop_when_no_backfill():
    df = _frame([("2020-01-01", 5, 6, 4, 5.5, 1000), ("2020-01-02", 5.5, 6, 5, 5.8, 1200)])
    out, info = C.cut_to_listing(df)
    assert len(out) == 2 and info["n_cut"] == 0


def test_drop_naninf_removes_nonfinite_rows():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                 ("2020-01-02", np.nan, 11, 9, 10, 100),
                 ("2020-01-03", 10, np.inf, 9, 10, 100)])
    out, info = C.drop_naninf(df)
    assert len(out) == 1
    assert info["n_dropped"] == 2


def test_flag_zero_range_keeps_rows_and_marks():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                 ("2020-01-02", 10, 10, 10, 10, 100)])
    out, info = C.flag_zero_range(df)
    assert len(out) == 2                       # nothing deleted
    assert out["zero_range_flag"].tolist() == [False, True]
    assert info["n_flagged"] == 1


def test_flag_zero_volume_keeps_rows_and_marks():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                 ("2020-01-02", 10, 11, 9, 10, 0)])
    out, info = C.flag_zero_volume(df)
    assert len(out) == 2
    assert out["zero_volume_flag"].tolist() == [False, True]
    assert info["n_flagged"] == 1


def test_backadjust_skips_jump_with_nonpositive_close():
    # a big |simple return| driven by a NEGATIVE close -> the guard skips it (cannot back-adjust).
    df = _frame([("2020-01-01", 10, 10.2, 9.8, 10, 100),
                 ("2020-01-02", 10, 10.2, 9.8, -5, 100)])   # ret = -5/10-1 = -1.5 (>50%) but close<=0
    out, info = C.backadjust_splits(df, thresh=0.5)
    assert info["n_adjusted"] == 0
    assert out.loc[0, "close"] == 10                        # unchanged (skipped)


def test_cut_to_listing_missing_volume_does_not_over_cut():
    # constant close + real range + NO volume column -> not backfill -> nothing cut.
    df = pd.DataFrame([(5, 6, 4, 5), (5, 6, 4, 5)], columns=["open", "high", "low", "close"])
    out, info = C.cut_to_listing(df)
    assert len(out) == 2 and info["n_cut"] == 0


def test_reconstruct_all_nonpositive_row_drops_without_warning():
    import warnings
    df = _frame([("2020-01-01", 0, 0, 0, 0, 100),        # all nonpositive -> nanmax all-NaN slice
                 ("2020-01-02", 10, 11, 9, 10, 100)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # any RuntimeWarning would fail the test
        out, info = C.reconstruct_nonpositive(df)
    assert len(out) == 1 and info["n_dropped"] == 1


def test_cut_to_listing_empty_frame():
    df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    out, info = C.cut_to_listing(df)
    assert len(out) == 0 and info["n_cut"] == 0


def test_cut_to_listing_entire_flat_series_is_noop():
    df = _frame([("2020-01-01", 5, 5, 5, 5, 0), ("2020-01-02", 5, 5, 5, 5, 0)])
    out, info = C.cut_to_listing(df)
    assert len(out) == 2 and info["n_cut"] == 0             # k>=n -> nothing cut


def test_cleaning_functions_do_not_mutate_input():
    df = _frame([("2020-01-01", 13, 11, 9, 7, 100)])
    before = df.copy(deep=True)
    C.widen_range(df)
    C.clip_oc(df)
    C.reconstruct_nonpositive(df)
    pd.testing.assert_frame_equal(df, before)
