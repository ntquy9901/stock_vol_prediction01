"""Unit + smoke tests for scripts/eda/sp500_eda.py (S&P 500 EDA generator).

Detectors are tested against hand-built fixtures with an independently-known answer; the panel scan /
report / renderers are exercised on a tiny synthetic multi-ticker fixture (fast, no GPU); one real-data
smoke reads a single delivered S&P 500 file and asserts the pipeline runs without exception.

Unique basename (test_sp500_eda.py) — no duplicate-basename pytest collision with other test_*.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sp500_eda as S  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------------------
# Pure detector unit tests
# --------------------------------------------------------------------------------------------------
def test_ohlc_geometry_violations_counts_each_kind():
    # row0 clean | row1 high<low (also forces open/close outside the inverted band) | row2 open>high |
    # row3 close>high | row4 zero-range | row5 nonpositive
    df = pd.DataFrame({
        "open":  [10.0, 10.0, 20.0, 10.0, 10.0, -1.0],
        "high":  [11.0, 8.0, 11.0, 11.0, 10.0, 11.0],
        "low":   [9.0, 9.0, 9.0, 9.0, 10.0, 9.0],
        "close": [10.5, 9.0, 10.5, 30.0, 10.0, 10.5],
    })
    r = S.ohlc_geometry_violations(df)
    assert r["n_rows"] == 6
    assert r["nonpositive"] == 1            # row5 open=-1
    assert r["high_lt_low"] == 1            # row1 (high 8 < low 9)
    assert r["open_outside"] == 2           # row1 (inverted band) + row2 (open 20 > high 11)
    assert r["close_outside"] == 2          # row1 (inverted band) + row3 (close 30 > high 11)
    assert r["zero_range"] == 1             # row4 high==low==10


def test_zero_parkinson_fraction_and_empty():
    assert S.zero_parkinson_fraction(np.array([0.0, 0.0, 1.0, 2.0])) == 0.5
    assert np.isnan(S.zero_parkinson_fraction(np.array([np.nan, np.nan])))


def test_log_returns_handles_nonpositive():
    r = S.log_returns(np.array([10.0, 20.0, 0.0, 5.0]))
    assert r.shape == (3,)
    assert np.isclose(r[0], np.log(2.0))
    assert np.isnan(r[1])                   # ln(0/20) -> -inf -> NaN


def test_extreme_jump_indices():
    close = np.array([10.0, 10.5, 30.0, 30.1])   # index2 is +>50%
    idx = S.extreme_jump_indices(close)
    assert list(idx) == [2]


def test_stale_runs_and_nonfinite_break():
    close = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, np.nan, 3.0, 3.0])
    runs = S.stale_runs(close, min_run=5)
    assert runs == [(0, 5)]                 # only the 5-length flat run qualifies
    assert S.stale_runs(np.array([1.0, 1.0]), min_run=5) == []


def test_robust_z_outlier_fraction_branches():
    x = np.concatenate([np.arange(50.0), [1000.0]])   # nonzero MAD + one extreme value
    assert S.robust_z_outlier_fraction(x) > 0.0
    assert S.robust_z_outlier_fraction(np.array([5.0, 5.0, 5.0])) == 0.0   # MAD==0
    assert np.isnan(S.robust_z_outlier_fraction(np.array([np.nan])))


def test_dist_stats_and_empty():
    s = S.dist_stats(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s["n"] == 4 and s["min"] == 1.0 and s["max"] == 4.0
    assert np.isnan(S.dist_stats(np.array([]))["mean"])


def test_acf_branches():
    ac = S.acf(np.sin(np.arange(200) / 3.0), nlags=10)
    assert ac.shape == (10,) and np.isfinite(ac).all()
    assert np.isnan(S.acf(np.array([1.0, 2.0]), nlags=10)).all()          # too short
    assert np.all(S.acf(np.ones(50), nlags=5) == 0.0)                     # zero variance -> zeros


def test_reservoir_subsamples_and_empty():
    assert S._reservoir([], 10).size == 0
    big = [np.arange(1000, dtype=float)]
    assert S._reservoir(big, 100).size == 100


# --------------------------------------------------------------------------------------------------
# Synthetic multi-ticker fixture (raw OHLCV + matching processed Parkinson)
# --------------------------------------------------------------------------------------------------
def _write_ticker(raw_dir: Path, proc_dir: Path, tk: str, n=150, seed=0, jump=False, stale=False,
                  zero_vol=False):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    if jump:
        close[60] = close[59] * 2.0                       # +100% single-day move
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)) + 1e-3)
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)) - 1e-3)
    open_ = np.clip(close * (1 + rng.normal(0, 0.003, n)), low, high)
    vol = rng.integers(1_000, 10_000, n).astype(float)
    if stale:
        close[10:20] = close[10]                          # 10-day flat run
        high[10:20] = close[10]; low[10:20] = close[10]   # zero-range on those days -> parkinson 0
        open_[10:20] = close[10]                           # keep OHLC geometry valid on the flat run
    if zero_vol:
        vol[5:8] = 0.0
    raw = pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low,
                        "close": close, "volume": vol})
    raw.to_csv(raw_dir / f"{tk}_ohlcv.csv", index=False)
    pk = np.log(high / low) ** 2 / (4 * np.log(2.0))
    pd.DataFrame({"date": dates, "parkinson_volatility": pk}).to_csv(
        proc_dir / f"{tk}_processed.csv", index=False)


@pytest.fixture
def fixture_panels(tmp_path, monkeypatch):
    """Two synthetic panels wired into sp500_eda.PROC / RAW: 'sp500' (detail) + 'p2' (comparison row)."""
    raw1, proc1 = tmp_path / "raw_sp", tmp_path / "proc_sp"
    raw2, proc2 = tmp_path / "raw_p2", tmp_path / "proc_p2"
    for d in (raw1, proc1, raw2, proc2):
        d.mkdir()
    for i in range(24):
        _write_ticker(raw1, proc1, f"T{i:02d}", seed=i, jump=(i == 0), stale=(i == 1), zero_vol=(i == 2))
    for i in range(22):
        _write_ticker(raw2, proc2, f"U{i:02d}", seed=100 + i)
    monkeypatch.setitem(S.PROC, "sp500", proc1)
    monkeypatch.setitem(S.RAW, "sp500", raw1)
    monkeypatch.setitem(S.PROC, "p2", proc2)
    monkeypatch.setitem(S.RAW, "p2", raw2)
    return {"raw1": raw1, "proc1": proc1, "raw2": raw2, "proc2": proc2}


def test_scan_ticker_detects_anomalies(fixture_panels):
    raw1, proc1 = fixture_panels["raw1"], fixture_panels["proc1"]
    rec = S.scan_ticker(str(raw1 / "T00_ohlcv.csv"), str(proc1 / "T00_processed.csv"))
    assert rec["n_jumps"] >= 1
    rec_stale = S.scan_ticker(str(raw1 / "T01_ohlcv.csv"), str(proc1 / "T01_processed.csv"))
    assert rec_stale["max_stale_run"] >= 10
    assert rec_stale["zero_pk_frac"] > 0.0                # zero-range days -> zero parkinson
    rec_zv = S.scan_ticker(str(raw1 / "T02_ohlcv.csv"), str(proc1 / "T02_processed.csv"))
    assert rec_zv["n_zero_vol"] == 3


def test_scan_ticker_empty_file(tmp_path):
    p = tmp_path / "EMPTY_ohlcv.csv"
    pd.DataFrame({"date": pd.bdate_range("2021-01-01", periods=3)}).to_csv(p, index=False)
    rec = S.scan_ticker(str(p), None)
    assert rec.get("empty") is True


def test_scan_panel_pools(fixture_panels):
    out = S.scan_panel("sp500")
    assert len(out["df"]) == 24
    assert out["ret_pool"].size > 0 and out["pk_pool"].size > 0
    assert out["har_w_pool"].size > 0 and out["volz_pool"].size > 0


def test_correlation_stats_and_early_return(fixture_panels):
    c = S.correlation_stats("sp500", window=120)
    assert c["corr"] is not None and 0.0 <= c["mean_abs"] <= 1.0
    none = S.correlation_stats("sp500", window=120, min_names=100)   # too few names -> None branch
    assert none["corr"] is None


def test_market_pk_series_and_empty(fixture_panels, tmp_path):
    s = S.market_pk_series("sp500")
    assert len(s) > 0
    empty = tmp_path / "empty_proc"
    empty.mkdir()
    monkey_dir = empty
    S.PROC["_empty"] = monkey_dir
    S.RAW["_empty"] = monkey_dir
    try:
        assert S.market_pk_series("_empty").empty
    finally:
        del S.PROC["_empty"]; del S.RAW["_empty"]


def test_panel_dirty_summary(fixture_panels):
    df = S.scan_panel("sp500")["df"]
    d = S.panel_dirty_summary(df)
    assert d["n_tickers"] == 24 and d["geom_violations"] == 0
    assert d["zero_range"] > 0 and d["n_jumps"] >= 1


# --------------------------------------------------------------------------------------------------
# Chart helper tests (matplotlib -> base64)
# --------------------------------------------------------------------------------------------------
def test_chart_helpers_return_b64():
    assert len(S.hist_png(np.random.randn(500), "t", "x")) > 100
    assert len(S.hist_png(np.abs(np.random.randn(500)) + 1e-3, "t", "x", logx=True)) > 100
    assert len(S.hist_png(np.array([-1.0, -2.0]), "t", "x", logx=True)) > 100   # empty-after-filter
    assert len(S.line_png([1, 2, 3], {"a": [1, 2, 3], "b": [3, 2, 1]}, "t", "x", "y")) > 100
    assert len(S.bar_png(["a", "b"], [1, 2], "t", "y", rot=90)) > 100
    assert len(S.heatmap_png(np.eye(70), "t")) > 100        # >60 -> subsample branch
    assert len(S.acf_png([1, 2, 3], [0.1, 0.2, 0.3], "t")) > 100
    assert S._img(S.hist_png(np.random.randn(10), "t", "x")).startswith("<img")


# --------------------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------------------
def test_num_and_date_helpers():
    assert S._num(None) == "-"
    assert S._num(float("nan")) == "-"
    assert S._num(0.5, pct=True) == "50.000%"
    assert S._num(1e-5, 3) == "1e-05"           # tiny -> %g
    assert S._num(0.123, 2) == "0.12"
    assert S._num(5) == "5"
    assert S._date("2021-01-01") == "2021-01-01"
    assert S._date("not-a-date") == "-"


# --------------------------------------------------------------------------------------------------
# Full report + renderers on the synthetic fixture (integration)
# --------------------------------------------------------------------------------------------------
def test_build_report_and_render(fixture_panels, tmp_path):
    rep = S.build_report(corr_window=120, panels=("sp500", "p2"))
    assert "sp500" in rep and len(rep["comparison"]) == 2
    html = tmp_path / "eda.html"
    md = tmp_path / "eda.md"
    S.render_html(rep, str(html))
    S.render_md(rep, str(md))
    h = html.read_text(encoding="utf-8")
    assert "Executive summary" in h and "data:image/png;base64," in h
    assert "S&P 500" in md.read_text(encoding="utf-8")


def test_render_html_corr_none_branch(fixture_panels, tmp_path):
    rep = S.build_report(corr_window=120, panels=("sp500",))
    rep["sp500"]["corr"]["corr"] = None            # force the no-heatmap branch
    out = tmp_path / "eda2.html"
    S.render_html(rep, str(out))
    assert out.exists()


def test_stats_table_and_comparison_table(fixture_panels):
    scan = S.scan_panel("sp500")
    t = S._stats_table({"a": scan["ret_pool"]})
    assert "<table>" in t and "feature" in t
    rows = [{"panel": "x", "n_tickers": 1, "total_rows": 10, "mean_abs_rho": 0.1, "mean_signed_rho": 0.1,
             "zero_range_frac": 0.0, "zero_pk_frac_mean": 0.0, "geom_violations": 0, "zero_vol_frac": 0.0,
             "median_history_yrs": 1.0}]
    assert "<table>" in S._comparison_table_html(rows)


def test_zero_range_over_time_png(fixture_panels):
    assert len(S._zero_range_over_time_png(fixture_panels["raw1"], sample=5)) > 100


# --------------------------------------------------------------------------------------------------
# Edge-branch coverage (malformed / minimal inputs)
# --------------------------------------------------------------------------------------------------
def test_read_sorted_no_date_column(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(p, index=False)
    assert len(S._read_sorted(str(p))) == 3          # no 'date' column -> returned unsorted


def test_scan_ticker_no_volume_no_processed(tmp_path):
    dates = pd.bdate_range("2021-01-01", periods=30)
    close = 100 + np.arange(30.0)
    raw = pd.DataFrame({"date": dates, "open": close, "high": close + 1, "low": close - 1, "close": close})
    p = tmp_path / "NV_ohlcv.csv"
    raw.to_csv(p, index=False)
    rec = S.scan_ticker(str(p), None)                # no volume column, no processed file
    assert "volz" not in rec and "n_proc" not in rec and rec["n_jumps"] == 0


def test_scan_panel_corr_marketpk_edge(tmp_path, monkeypatch):
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    raw.mkdir(); proc.mkdir()
    for i in range(3):
        _write_ticker(raw, proc, f"G{i}", seed=i)
    pd.DataFrame({"date": pd.bdate_range("2021-01-01", periods=3), "close": [1.0, 1.1, 1.2]}).to_csv(
        raw / "EMP_ohlcv.csv", index=False)          # missing open/high/low -> 'empty' for the OHLC scan
    pd.DataFrame({"date": ["2021-01-01"], "open": [1.0], "high": [1.1], "low": [0.9], "close": [1.0],
                  "volume": [10]}).to_csv(raw / "ONE_ohlcv.csv", index=False)     # 1 row -> corr <2 skip
    pd.DataFrame({"date": [], "parkinson_volatility": []}).to_csv(proc / "ZR_processed.csv", index=False)
    monkeypatch.setitem(S.PROC, "edge", proc)
    monkeypatch.setitem(S.RAW, "edge", raw)
    out = S.scan_panel("edge")
    assert len(out["df"]) == 4                        # EMP skipped (empty), G0-2 + ONE kept
    assert S.correlation_stats("edge", window=120, min_names=1)["corr"] is not None
    assert len(S.market_pk_series("edge")) > 0        # ZR (0-row) processed skipped without crashing


def test_zero_range_over_time_with_malformed_file(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_ticker(raw, tmp_path, "GOOD", seed=1)      # valid file for a non-empty year axis
    pd.DataFrame({"a": [1, 2]}).to_csv(raw / "BAD_ohlcv.csv", index=False)   # no date/high/low -> skipped
    assert len(S._zero_range_over_time_png(raw, sample=10)) > 100


# --------------------------------------------------------------------------------------------------
# Real-data-sample smoke
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (REPO / "data" / "raw" / "prices" / "sp500" / "AAPL_ohlcv.csv").exists(),
                    reason="delivered S&P 500 sample not present")
def test_real_data_smoke():
    raw = str(REPO / "data" / "raw" / "prices" / "sp500" / "AAPL_ohlcv.csv")
    proc = str(REPO / "data" / "processed" / "sp500" / "AAPL_processed.csv")
    rec = S.scan_ticker(raw, proc if Path(proc).exists() else None)
    assert rec["n_raw"] > 1000 and rec["high_lt_low"] == 0
    assert np.isfinite(rec["rz_out_frac"])
