"""Unit tests for the pure detectors / statistics in vnmarkets_eda (unique basenames to avoid the pytest
duplicate-module collision). Known-answer fixtures; independent recompute where a formula is involved."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import scripts.eda.vnmarkets_eda as E


def test_log_and_simple_returns_handle_nonpositive():
    close = np.array([10.0, 11.0, 0.0, 12.0])
    lr = E.log_returns(close)
    assert math.isnan(lr[0])
    assert lr[1] == pytest.approx(math.log(11.0 / 10.0))
    assert math.isnan(lr[2])          # C_t == 0 -> NaN
    assert math.isnan(lr[3])          # prev close 0 -> NaN
    sr = E.simple_returns(close)
    assert sr[1] == pytest.approx(11.0 / 10.0 - 1.0)
    assert math.isnan(sr[0])
    assert math.isnan(sr[3])          # prev 0


def test_skewness_and_kurtosis_branches():
    assert math.isnan(E.skewness(np.array([1.0, 2.0])))          # <3 pts
    assert math.isnan(E.skewness(np.array([5.0, 5.0, 5.0])))     # zero sigma
    assert math.isnan(E.excess_kurtosis(np.array([1.0, 2.0, 3.0])))   # <4 pts
    assert math.isnan(E.excess_kurtosis(np.array([2.0, 2.0, 2.0, 2.0])))  # zero sigma
    x = np.array([0.0, 0.0, 0.0, 10.0])                          # right-skewed
    assert E.skewness(x) > 0
    # independent recompute vs formula
    v = np.array([1.0, 2.0, 3.0, 10.0])
    mu, sd = v.mean(), v.std()
    assert E.skewness(v) == pytest.approx(np.mean(((v - mu) / sd) ** 3))
    assert E.excess_kurtosis(v) == pytest.approx(np.mean(((v - mu) / sd) ** 4) - 3.0)


def test_summary_stats_empty_and_normal():
    empty = E.summary_stats(np.array([np.nan, np.nan]))
    assert empty["n"] == 0 and math.isnan(empty["mean"])
    s = E.summary_stats(np.arange(1, 101, dtype=float))
    assert s["n"] == 100
    assert s["median"] == pytest.approx(50.5)
    assert s["min"] == 1.0 and s["max"] == 100.0


def test_robust_z_and_outlier_counts():
    assert np.all(E.robust_z(np.array([np.nan, np.nan])) == 0)     # all-nan -> zeros
    assert np.all(E.robust_z(np.array([3.0, 3.0, 3.0])) == 0)      # zero MAD -> zeros
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0])   # nonzero MAD, one gross outlier
    assert E.count_robust_outliers(x, thresh=3.5) >= 1
    assert E.iqr_outlier_count(np.array([])) == 0
    assert E.iqr_outlier_count(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100.0])) >= 1


def test_detect_ohlc_violations_every_kind():
    df = pd.DataFrame({
        "open": [10.0, 10.0, -1.0, 10.0, 10.0, np.nan],
        "high": [11.0, 9.0, 5.0, 9.5, 10.0, 11.0],   # row1 high<low; row3 high<open(10)
        "low": [9.0, 10.0, 4.0, 9.0, 10.0, 9.0],     # row4 zero range (high==low==10)
        "close": [10.5, 9.5, 4.5, 9.2, 10.0, 10.0],
    })
    d = E.detect_ohlc_violations(df)
    assert d["n"] == 6
    assert d["high_lt_low"] == 1
    assert d["nonpositive"] == 1
    assert d["open_close_outside"] >= 1
    assert d["zero_range"] == 1
    assert d["nan_rows"] == 1


def test_zero_parkinson_and_volume_fraction():
    assert math.isnan(E.zero_parkinson_fraction(np.array([np.nan])))
    assert E.zero_parkinson_fraction(np.array([0.0, 0.0, 1.0, 2.0])) == pytest.approx(0.5)
    assert math.isnan(E.zero_volume_fraction(np.array([np.nan])))
    assert E.zero_volume_fraction(np.array([0.0, 1.0])) == pytest.approx(0.5)


def test_detect_split_jumps_sorted_and_empty():
    dates = np.array(["d0", "d1", "d2", "d3"])
    close = np.array([10.0, 10.1, 20.5, 10.0])   # d2 +100%, d3 ~-51%
    jumps = E.detect_split_jumps(dates, close, thresh=0.5)
    assert jumps[0][0] == "d2"                    # largest |ret| first
    assert len(jumps) == 2
    assert E.detect_split_jumps(dates, np.array([10.0, 10.1, 10.2, 10.3])) == []


def test_detect_stale_runs():
    dates = np.array([f"d{i}" for i in range(8)])
    close = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0])   # 5-long run of 5.0
    runs = E.detect_stale_runs(dates, close, min_run=5)
    assert runs == [("d0", "d4", 5)]
    assert E.detect_stale_runs(dates, np.array([1.0, 1.0, 2, 3, 4, 5, 6, 7])) == []   # below min
    neg = E.detect_stale_runs(dates, np.array([-1.0, -1, -1, -1, -1, 2, 3, 4]))
    assert neg == []                              # nonpositive not counted


def test_acf_branches():
    assert np.all(np.isnan(E.acf(np.array([1.0, 2.0]), 5)))       # <3 pts
    assert np.all(np.isnan(E.acf(np.array([4.0, 4.0, 4.0, 4.0]), 3)))  # zero variance
    a = E.acf(np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]), 2)
    assert a[0] < 0                                # alternating -> lag1 negative
    short = E.acf(np.array([1.0, 2.0, 3.0]), 10)   # k>=n break path
    assert math.isnan(short[9])


def test_zero_parkinson_by_year():
    assert E.zero_parkinson_by_year(np.array(["x"]), np.array([np.nan])) == {}
    dates = np.array(["2020-01-01", "2020-06-01", "2021-03-01"])
    pk = np.array([0.0, 1.0, 0.0])
    ym = E.zero_parkinson_by_year(dates, pk)
    assert ym[2020] == pytest.approx(0.5)
    assert ym[2021] == pytest.approx(1.0)


def test_correlation_stats_empty_and_normal():
    one = pd.DataFrame({"A": np.arange(300.0)})   # single column -> no pairs -> nan branch
    r = E.correlation_stats(one, min_overlap=20)
    assert r["n_pairs"] == 0 and math.isnan(r["median_abs_rho"])
    rng = np.random.default_rng(0)
    base = rng.normal(size=400)
    wide = pd.DataFrame({"A": base, "B": base + rng.normal(scale=0.1, size=400),
                         "C": rng.normal(size=400)})
    s = E.correlation_stats(wide, min_overlap=50)
    assert s["n_tickers"] == 3 and s["n_pairs"] == 3
    assert 0.0 <= s["median_abs_rho"] <= 1.0


def test_build_cross_market_table_and_compact_and_md():
    def mk(name):
        return {"panel": name, "raw_tickers": 3, "screened_tickers": 2, "total_rows": 100,
                "date_min": "2020-01-01", "date_max": "2021-01-01",
                "corr": {"median_abs_rho": 0.1, "mean_abs_rho": 0.12},
                "zero_parkinson_rate": 0.05, "zero_volume_rate": 0.01,
                "dirty": {"high_lt_low": 1, "open_close_outside": 2, "nonpositive": 0,
                          "zero_range": 5, "split_jumps": 3, "stale_run_tickers": 4}}
    summaries = {"hose": mk("hose"), "vn30": mk("vn30")}   # unordered
    rows = E.build_cross_market_table(summaries)
    assert [r["market"] for r in rows] == ["vn30", "hose"]  # canonical order
    assert rows[0]["median_abs_rho"] == 0.1
    md = E._comparison_md(rows)
    assert "vn30" in md and "| market |" in md
    comp = E._compact({**mk("vn30"), "per_ticker": [1, 2], "_charts": {}, "corr": mk("vn30")["corr"],
                       "tickers_per_year": {}, "split_examples": [], "stale_examples": [],
                       "zero_parkinson_by_year": {}, "stats": {}})
    assert "per_ticker" not in comp and comp["panel"] == "vn30"
