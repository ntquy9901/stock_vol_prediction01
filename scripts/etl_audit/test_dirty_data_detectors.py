"""TDD tests for the per-(ticker,date) dirty-data detectors (scripts/etl_audit/dirty_data_detectors.py).

Failing-first: written before the implementation. Each detector is checked on a synthetic frame with a
planted defect, plus a cross-check that the counts agree with the existing read-only ``vnmarkets_eda``
detectors, plus a real-data-sample smoke that skips cleanly when a market's data is absent.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "etl_audit"))
sys.path.insert(0, str(REPO / "scripts" / "eda"))
sys.path.insert(0, str(REPO / "scripts" / "garch_masked"))

import dirty_data_detectors as D  # noqa: E402


def _frame(rows):
    """rows = list of (date, open, high, low, close, volume)."""
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def test_high_lt_low_locates_exact_dates():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                 ("2020-01-02", 10, 8, 12, 10, 100),   # high 8 < low 12
                 ("2020-01-03", 10, 11, 9, 10, 100)])
    assert D.high_lt_low(df) == ["2020-01-02"]


def test_open_close_outside_locates_and_measures_magnitude():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),    # clean
                 ("2020-01-02", 13, 11, 9, 10, 100),    # open 13 > high 11
                 ("2020-01-03", 10, 11, 9, 7, 100)])    # close 7 < low 9
    out = D.open_close_outside(df)
    dates = [d for d, _m in out]
    assert dates == ["2020-01-02", "2020-01-03"]
    # magnitude is a positive relative violation for both flagged rows
    assert all(m > 0 for _d, m in out)
    # open 13 vs high 11 -> relative overshoot (13-11)/11
    assert out[0][1] == pytest.approx((13 - 11) / 11, rel=1e-6)


def test_nonpositive_locates_zero_or_negative_ohlc():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                 ("2020-01-02", 0, 11, 9, 10, 100),     # open 0
                 ("2020-01-03", 10, 11, -1, 10, 100)])  # low -1
    assert D.nonpositive(df) == ["2020-01-02", "2020-01-03"]


def test_zero_range_locates_high_eq_low_positive():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                 ("2020-01-02", 10, 10, 10, 10, 100)])  # H==L
    assert D.zero_range(df) == ["2020-01-02"]


def test_zero_range_ignores_nonpositive_flatline():
    df = _frame([("2020-01-01", 0, 0, 0, 0, 0)])        # H==L but nonpositive -> not a zero-range trade day
    assert D.zero_range(df) == []


def test_split_jumps_locates_extreme_return_with_sign():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                 ("2020-01-02", 10, 11, 9, 20, 100),    # +100% jump
                 ("2020-01-03", 20, 21, 19, 20, 100)])
    out = D.split_jumps(df, thresh=0.5)
    assert [d for d, _r in out] == ["2020-01-02"]
    assert out[0][1] == pytest.approx(1.0, rel=1e-6)


def test_stale_runs_locates_repeated_closes():
    closes = [10, 12, 12, 12, 12, 12, 13]   # 5-long run of 12 (indices 1..5)
    df = _frame([(f"2020-01-{i+1:02d}", c, c + 1, c - 1, c, 100) for i, c in enumerate(closes)])
    runs = D.stale_runs(df, min_run=5)
    assert len(runs) == 1
    start, end, length = runs[0]
    assert (start, end, length) == ("2020-01-02", "2020-01-06", 5)


def test_naninf_locates_nonfinite_rows():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                 ("2020-01-02", np.nan, 11, 9, 10, 100),
                 ("2020-01-03", 10, np.inf, 9, 10, 100)])
    assert D.naninf(df) == ["2020-01-02", "2020-01-03"]


def test_zero_volume_locates_zero_volume_days():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                 ("2020-01-02", 10, 11, 9, 10, 0)])
    assert D.zero_volume(df) == ["2020-01-02"]


def test_leading_backfill_finds_prelisting_constant_run():
    # 3 leading pre-listing rows: constant close, zero volume, zero range; then real trading.
    rows = [("2020-01-01", 5, 5, 5, 5, 0), ("2020-01-02", 5, 5, 5, 5, 0), ("2020-01-03", 5, 5, 5, 5, 0),
            ("2020-01-06", 5, 6, 4, 5.5, 1000), ("2020-01-07", 5.5, 6, 5, 5.8, 1200)]
    df = _frame(rows)
    info = D.leading_backfill(df)
    assert info["n_leading"] == 3
    assert info["first_trade_date"] == "2020-01-06"


def test_leading_backfill_zero_when_trades_from_start():
    df = _frame([("2020-01-01", 5, 6, 4, 5.5, 1000), ("2020-01-02", 5.5, 6, 5, 5.8, 1200)])
    info = D.leading_backfill(df)
    assert info["n_leading"] == 0


def test_detect_all_returns_counts_for_every_class():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                 ("2020-01-02", 10, 8, 12, 10, 0),      # high<low + zero volume
                 ("2020-01-03", 13, 11, 9, 10, 100)])   # open>high
    res = D.detect_all(df)
    for key in ("high_lt_low", "open_close_outside", "nonpositive", "zero_range", "split_jumps",
                "stale_runs", "naninf", "zero_volume", "leading_backfill"):
        assert key in res["counts"]
    assert res["counts"]["high_lt_low"] == 1
    # the high<low row (2020-01-02) also has open/close outside its (empty) [low,high]; classes are
    # independent detectors, so it is counted in BOTH -> open_close_outside == 2 (that row + 2020-01-03).
    assert res["counts"]["open_close_outside"] == 2
    assert res["counts"]["zero_volume"] == 1


def test_per_ticker_summary_is_flat_row():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                 ("2020-01-02", 10, 8, 12, 10, 100)])
    row = D.per_ticker_summary("TEST", df)
    assert row["ticker"] == "TEST"
    assert row["rows"] == 2
    assert row["high_lt_low"] == 1


def test_counts_agree_with_vnmarkets_eda_detectors():
    """Cross-check (read-only reuse) that our locate-detectors agree with the existing count detectors."""
    vn = pytest.importorskip("vnmarkets_eda")
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                 ("2020-01-02", 10, 8, 12, 10, 100),     # high<low
                 ("2020-01-03", 13, 11, 9, 10, 100),     # open>high
                 ("2020-01-04", 10, 10, 10, 10, 100),    # zero range
                 ("2020-01-05", 0, 11, 9, 10, 100)])     # nonpositive
    ref = vn.detect_ohlc_violations(df)
    assert len(D.high_lt_low(df)) == ref["high_lt_low"]
    assert len(D.open_close_outside(df)) == ref["open_close_outside"]
    assert len(D.nonpositive(df)) == ref["nonpositive"]
    assert len(D.zero_range(df)) == ref["zero_range"]


def test_frame_without_date_uses_positional_index():
    df = pd.DataFrame([(10, 8, 12, 10, 100)], columns=["open", "high", "low", "close", "volume"])
    assert D.high_lt_low(df) == ["0"]           # positional index string when no date column


def test_naninf_and_zero_volume_without_volume_column():
    df = pd.DataFrame([(10, 11, 9, 10)], columns=["open", "high", "low", "close"])
    assert D.naninf(df) == []                    # no volume col -> volume treated as finite zeros
    assert D.zero_volume(df) == []               # no volume col -> nothing to flag


def test_leading_backfill_empty_frame():
    df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    assert D.leading_backfill(df) == {"n_leading": 0, "first_trade_date": None}


def test_leading_backfill_entire_flat_series_flags_nothing():
    df = _frame([("2020-01-01", 5, 5, 5, 5, 0), ("2020-01-02", 5, 5, 5, 5, 0)])
    assert D.leading_backfill(df) == {"n_leading": 0, "first_trade_date": None}


def test_leading_backfill_missing_volume_does_not_over_flag():
    # constant close but a REAL H-L range and NO volume column -> not backfill (must not read volume as 0).
    df = pd.DataFrame([(5, 6, 4, 5, ), (5, 6, 4, 5)], columns=["open", "high", "low", "close"])
    assert D.leading_backfill(df)["n_leading"] == 0


def test_detect_all_counts_stale_runs_as_days_not_runs():
    # one run of 6 identical closes -> stale DAYS == 6 (comparable to the other day-based classes).
    df = _frame([(f"2020-03-{i+1:02d}", 12, 13, 11, 12, 100) for i in range(6)])
    assert D.detect_all(df)["counts"]["stale_runs"] == 6


def test_detect_all_open_close_examples_sorted_by_magnitude():
    df = _frame([("2020-01-01", 12, 11, 9, 10, 100),    # open 12 -> small overshoot (12-11)/11
                 ("2020-01-02", 30, 11, 9, 10, 100)])   # open 30 -> large overshoot (30-11)/11
    ex = D.detect_all(df)["examples"]["open_close_outside"]
    assert ex[0][0] == "2020-01-02"                     # largest violation first
    assert ex[0][1] > ex[1][1]


@pytest.mark.parametrize("panel", ["vn30", "vn100", "hose", "hnx", "sp500"])
def test_real_data_sample_smoke(panel):
    import volatility_estimators as VE
    files = sorted(glob.glob(str(VE.PRICE[panel] / "*_ohlcv.csv")))
    if not files:  # pragma: no cover - depends on which market data is present locally
        pytest.skip(f"no raw data for {panel}")
    df = pd.read_csv(files[0])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    res = D.detect_all(df)
    assert res["counts"]["high_lt_low"] >= 0
    row = D.per_ticker_summary(Path(files[0]).stem.replace("_ohlcv", ""), df)
    assert row["rows"] == len(df)
