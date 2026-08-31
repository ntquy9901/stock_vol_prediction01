"""Unit + integration tests for the HNX per-ticker diagnostic generator.

Unique filename (test_hnx_perticker_*) to avoid the repo's pytest duplicate-basename collision.
Covers the pure Part-A stats, Part-B per-ticker metric aggregation, flag rules, summary, and HTML/MD
rendering, plus one integration run of the Part-A driver on a tiny synthetic panel and a real-data
slice smoke.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hnx_per_ticker_diagnostic as H  # noqa: E402


# --------------------------------------------------------------------------- Part A pure helpers
def test_weekday_gaps_counts_missing_business_days():
    # Mon..Fri present except one missing weekday (Wed) -> exactly 1 gap; weekend never counts.
    dates = ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"]  # missing Wed 01-03
    assert H.weekday_gaps(dates) == 1


def test_weekday_gaps_single_or_empty_is_zero():
    assert H.weekday_gaps(["2024-01-01"]) == 0
    assert H.weekday_gaps([]) == 0


def test_parkinson_dist_reports_validity_and_quantiles():
    vals = [1.0, 2.0, 4.0, 0.0, -1.0, np.nan, np.inf]
    d = H.parkinson_dist(vals)
    assert d["n_nan"] == 1 and d["n_inf"] == 1
    assert d["n_nonpos"] == 2                      # the 0.0 and the -1.0
    assert d["min"] == 1.0 and d["max"] == 4.0
    assert d["median"] == 2.0


def test_parkinson_dist_all_invalid_yields_nan_stats():
    d = H.parkinson_dist([0.0, -1.0, np.nan])
    assert np.isnan(d["min"]) and np.isnan(d["median"]) and np.isnan(d["p95"]) and np.isnan(d["max"])
    assert d["n_nonpos"] == 2 and d["n_nan"] == 1


def test_ohlc_sanity_detects_all_violation_types_and_backfill():
    raw = pd.DataFrame({
        "open": [10.0, 10.0, 5.0, 10.0],
        "high": [11.0, 9.0, 6.0, 12.0],     # row1: high<low; row2 open/close outside handled below
        "low": [9.0, 10.0, 4.0, 8.0],
        "close": [10.5, 10.0, 5.5, -1.0],   # row3 nonpositive price
        "volume": [0.0, 0.0, 100.0, 50.0],  # two leading zero-volume rows -> backfill proxy = 2
    })
    s = H.ohlc_sanity(raw)
    assert s["n_high_lt_low"] == 1            # row index 1 (high 9 < low 10)
    assert s["n_oc_outside"] >= 1             # row1 open/close (10) outside [10,9] etc.
    assert s["n_nonpos_price"] == 1           # row3 close -1
    assert s["backfill_lead_zerovol"] == 2
    assert s["n_raw_rows"] == 4


def test_ohlc_sanity_missing_columns_returns_zeros():
    raw = pd.DataFrame({"date": ["2024-01-01"], "price": [1.0]})
    s = H.ohlc_sanity(raw)
    assert s["n_high_lt_low"] == 0 and s["backfill_lead_zerovol"] == 0 and s["n_raw_rows"] == 1


def test_ohlc_sanity_no_volume_column_backfill_zero():
    raw = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]})
    assert H.ohlc_sanity(raw)["backfill_lead_zerovol"] == 0


def test_ohlc_sanity_all_zero_volume_counts_all_leading():
    # every row zero-volume -> the leading-count loop exhausts without break -> counts all rows
    raw = pd.DataFrame({"open": [1.0, 1.0], "high": [2.0, 2.0], "low": [0.5, 0.5],
                        "close": [1.5, 1.5], "volume": [0.0, 0.0]})
    assert H.ohlc_sanity(raw)["backfill_lead_zerovol"] == 2


def _proc(dates, vals):
    return pd.DataFrame({"date": dates, "parkinson_variance": vals})


def test_data_stats_for_ticker_kept_and_excluded_paths():
    dates = pd.bdate_range("2020-01-01", periods=5).strftime("%Y-%m-%d").tolist()
    proc = _proc(dates, [1.0, 0.0, 2.0, 3.0, 4.0])
    raw = pd.DataFrame({"open": [1, 1, 1, 1, 1], "high": [2, 2, 2, 2, 2],
                        "low": [0.5] * 5, "close": [1.5] * 5, "volume": [10] * 5})
    kept = H.data_stats_for_ticker("AAA", proc, raw, "")
    assert kept["kept"] is True and kept["screen_decision"] == "kept"
    assert kept["n_valid"] == 5
    assert kept["zero_park_frac"] == pytest.approx(0.2)   # one exact zero of five
    assert kept["first_date"] == dates[0] and kept["last_date"] == dates[-1]
    excl = H.data_stats_for_ticker("BBB", proc, raw, "high-zero-frac: 0.60 > 0.5")
    assert excl["kept"] is False and excl["screen_decision"] == "excluded"
    assert excl["screen_reason"].startswith("high-zero-frac")


def test_data_stats_for_ticker_empty_valid():
    proc = _proc(["2020-01-01", "2020-01-02"], [np.nan, np.inf])
    row = H.data_stats_for_ticker("CCC", proc, pd.DataFrame(), "too-few-rows: 2 < 250")
    assert row["n_valid"] == 0
    assert row["first_date"] == "" and row["last_date"] == ""
    assert np.isnan(row["zero_park_frac"]) and np.isnan(row["coverage"])
    assert row["calendar_days"] == 0


# ----------------------------------------------------------------------- Part A panel split stats
def test_panel_split_stats_counts_zero_and_floor_activation():
    tickers = ["AAA", "BBB"]
    y_tr = np.array([[1.0, 0.0], [2.0, 0.0]])
    tm_tr = np.array([[1, 1], [1, 0]])
    y_va = np.array([[1.0, 5.0]])
    tm_va = np.array([[1, 1]])
    y_te = np.array([[0.001, 3.0], [10.0, 3.0]])
    tm_te = np.array([[1, 1], [1, 1]])
    t_mean = np.array([1.0, 100.0])   # AAA floor = 0.01; BBB floor = 1.0
    splits = {"train": (y_tr, tm_tr), "val": (y_va, tm_va), "test": (y_te, tm_te)}
    st = H.panel_split_stats(tickers, splits, t_mean)
    assert st["AAA"]["train"]["n"] == 2 and st["AAA"]["train"]["zero_frac"] == 0.0
    assert st["BBB"]["train"]["n"] == 1 and st["BBB"]["train"]["zero_frac"] == 1.0  # only valid cell is 0
    # AAA test floor 0.01: value 0.001<=floor (activated), 10.0 not -> 0.5
    assert st["AAA"]["test"]["floor_activation"] == pytest.approx(0.5)
    assert st["BBB"]["test"]["floor_activation"] == 0.0


def test_panel_split_stats_empty_test_mask_nan():
    tickers = ["AAA"]
    splits = {"train": (np.array([[1.0]]), np.array([[1]])),
              "val": (np.array([[1.0]]), np.array([[0]])),
              "test": (np.array([[1.0]]), np.array([[0]]))}
    st = H.panel_split_stats(tickers, splits, np.array([1.0]))
    assert np.isnan(st["AAA"]["val"]["zero_frac"])
    assert np.isnan(st["AAA"]["test"]["floor_activation"])


# ---------------------------------------------------------------------- Part B model aggregation
def test_per_ticker_model_metrics_groups_and_uses_shared_floor():
    tickers = ["AAA", "BBB"]
    y_te = np.array([[1.0, 2.0], [1.5, 2.5]])
    tm_te = np.array([[1, 1], [1, 0]])            # BBB has only 1 valid test cell
    preds = {"LSTM": np.array([[1.0, 2.0], [1.5, 9.9]]),
             "HARX": np.array([[2.0, 4.0], [3.0, 9.9]])}
    out = H.per_ticker_model_metrics(y_te, tm_te, preds, tickers, floor=1e-8)
    assert set(out) == {"AAA", "BBB"}
    assert out["AAA"]["LSTM"]["qlike"] == pytest.approx(0.0, abs=1e-9)   # exact forecast
    assert out["AAA"]["LSTM"]["mse"] == pytest.approx(0.0, abs=1e-12)
    assert out["BBB"]["LSTM"]["n"] == 1
    assert out["AAA"]["HARX"]["qlike"] > 0.0


def test_per_ticker_model_metrics_skips_empty_ticker():
    tickers = ["AAA", "EMPTY"]
    y_te = np.array([[1.0, 2.0]])
    tm_te = np.array([[1, 0]])
    preds = {"LSTM": np.array([[1.0, 2.0]])}
    out = H.per_ticker_model_metrics(y_te, tm_te, preds, tickers, floor=1e-8)
    assert "EMPTY" not in out and "AAA" in out


# --------------------------------------------------------------------------------- flags + summary
def _row(**kw):
    base = dict(ticker="AAA", zero_park_frac=0.0, n_high_lt_low=0, n_oc_outside=0, n_nonpos_price=0,
                n_valid=1000, floor_activation=0.0, ohlc_total=0)
    base.update(kw)
    return base


def test_flag_row_red_conditions():
    assert H.flag_row(_row(zero_park_frac=0.5), None)["severity"] == "red"
    assert H.flag_row(_row(n_high_lt_low=3), None)["severity"] == "red"
    assert H.flag_row(_row(n_valid=100), None)["severity"] == "red"


def test_flag_row_amber_floor_and_qlike():
    amber_floor = H.flag_row(_row(floor_activation=0.3), None)
    assert amber_floor["severity"] == "amber" and any("floor" in f for f in amber_floor["flags"])
    amber_q = H.flag_row(_row(qlike_max=5.0), qlike_median=1.0)
    assert amber_q["severity"] == "amber" and any("QLIKE" in f for f in amber_q["flags"])


def test_flag_row_ok_and_qlike_ignored_when_no_median():
    r = H.flag_row(_row(qlike_max=99.0), qlike_median=None)   # no median -> QLIKE not evaluated
    assert r["severity"] == "ok" and r["flags"] == []


def test_flag_row_nan_values_do_not_flag():
    r = H.flag_row(_row(zero_park_frac=float("nan"), floor_activation=float("nan")), None)
    assert r["severity"] == "ok"


def test_build_summary_counts_and_aggregate():
    rows = [
        H.flag_row(_row(ticker="RED", zero_park_frac=0.6, n_valid=300, ohlc_total=0), None),
        H.flag_row(_row(ticker="AMB", floor_activation=0.5, n_valid=500), None),
        H.flag_row(_row(ticker="OK1", zero_park_frac=0.1, n_valid=1000), None),
    ]
    s = H.build_summary(rows)
    assert s["n_tickers"] == 3 and s["n_red"] == 1 and s["n_amber"] == 1 and s["n_ok"] == 1
    # row-weighted aggregate zero-frac = (0.6*300 + 0.5*0? no zf) ... compute explicitly
    exp = (0.6 * 300 + 0.0 * 500 + 0.1 * 1000) / (300 + 500 + 1000)
    assert s["agg_zero_park_frac"] == pytest.approx(exp)
    assert s["worst_zero_frac"][0][0] == "RED"
    assert s["worst_fewest_rows"][0][0] == "RED"        # 300 is fewest


def test_build_summary_empty_rows_nan_aggregate():
    s = H.build_summary([])
    assert s["n_tickers"] == 0 and np.isnan(s["agg_zero_park_frac"])
    assert s["worst_zero_frac"] == []


# ------------------------------------------------------------------------------------- rendering
def _sample_flagged():
    rows = [
        H.flag_row(_row(ticker="AAA", zero_park_frac=0.6, n_valid=300, first_date="2020-01-01",
                        last_date="2022-01-01", screen_decision="kept", screen_reason="kept",
                        coverage=0.9, weekday_gaps=3, pk_median=1e-4, pk_p95=1e-3, pk_max=1e-2,
                        pk_n_nan=0, pk_n_inf=0, pk_n_nonpos=0, backfill_lead_zerovol=0,
                        n_raw_rows=300, train_n=100, val_n=20, test_n=20,
                        qlike_LSTM=1.5, qlike_HARX=1.6, qlike_VolGA=1.55, r2_LSTM=0.2,
                        qlike_max=1.6), qlike_median=1.5),
        H.flag_row(_row(ticker="BBB", n_valid=1000, first_date="2018-01-01", last_date="2024-01-01",
                        screen_decision="excluded", screen_reason="high-zero-frac", coverage=0.8,
                        weekday_gaps=5, pk_median=2e-4, pk_p95=2e-3, pk_max=2e-2, pk_n_nan=1,
                        pk_n_inf=0, pk_n_nonpos=0, backfill_lead_zerovol=1, n_raw_rows=1000),
                   qlike_median=1.5),
    ]
    return rows


def test_render_html_is_self_contained_and_has_rows():
    rows = _sample_flagged()
    summary = H.build_summary(rows)
    html = H.render_html(rows, summary, {"generated": "2026-08-30 00:00",
                                         "model_status": "5 seeds"})
    assert "<html" in html and "http://" not in html and "https://" not in html   # no external CDN
    assert "AAA" in html and "BBB" in html
    assert "function srt" in html and "function flt" in html                       # interactivity
    assert "pending" in html                                                       # BBB has no model cols


def test_render_md_has_sections_and_recommendation():
    rows = _sample_flagged()
    summary = H.build_summary(rows)
    md = H.render_md(rows, summary, {"generated": "2026-08-30 00:00", "model_status": "5 seeds"})
    assert "# HNX per-ticker diagnostic" in md
    assert "## Worst tickers" in md and "## Recommendation" in md
    assert "AAA" in md


def test_cell_formats_pending_and_nonfinite():
    assert "pending" in H._cell(None)
    assert ">-<" in H._cell(float("nan"))
    assert "1.50" in H._cell(1.5, "{:.2f}")
    assert ">7<" in H._cell(7)


# ------------------------------------------------------------------- integration (Part A driver)
def _make_synth_panel(tmp: Path, n_tickers=10, n_rows=700, seed=0):
    """Write a tiny synthetic processed+raw HNX-like panel that clears the screen and builds a panel."""
    rng = np.random.default_rng(seed)
    proc_dir = tmp / "processed"
    price_dir = tmp / "raw"
    proc_dir.mkdir(parents=True)
    price_dir.mkdir(parents=True)
    dates = pd.bdate_range("2015-01-01", periods=n_rows)
    for i in range(n_tickers):
        base = 20.0 + i
        close = base + np.cumsum(rng.normal(0, 0.2, n_rows))
        close = np.abs(close) + 1.0
        high = close + np.abs(rng.normal(0.3, 0.1, n_rows))
        low = close - np.abs(rng.normal(0.3, 0.1, n_rows))
        low = np.minimum(low, np.minimum(close, high) - 1e-3)
        open_ = (high + low) / 2
        vol = rng.integers(1000, 5000, n_rows).astype(float)
        pk = (np.log(high / low) ** 2) / (4 * np.log(2))
        tk = f"T{i:02d}"
        pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": open_, "high": high,
                      "low": low, "close": close, "volume": vol}).to_csv(
            price_dir / f"{tk}_ohlcv.csv", index=False)
        pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "parkinson_variance": pk}).to_csv(
            proc_dir / f"{tk}_processed.csv", index=False)
    return proc_dir, price_dir


def test_collect_data_rows_integration_builds_panel(tmp_path):
    proc_dir, price_dir = _make_synth_panel(tmp_path)
    rows, panel, meta = H.collect_data_rows(proc_dir, price_dir, lookback=5, horizon=1,
                                            min_valid=3, min_train_rows=50)
    assert meta["n_all"] == 10 and meta["n_kept"] == 10 and meta["n_excluded"] == 0
    assert panel is not None and panel.N >= 3
    # every ticker got a data row; panel tickers got split counts
    assert len(rows) == 10
    in_panel = [r for r in rows if r.get("in_panel")]
    assert in_panel and all("test_n" in r and "floor_activation" in r for r in in_panel)
    # end-to-end assemble + render must not raise on the real driver output (DATA-only path)
    flagged, summary = H.assemble(rows, panel, None)
    html = H.render_html(flagged, summary, {"generated": "x", "model_status": "pending"})
    assert "T00" in html


def test_real_hnx_slice_smoke():
    """Real-data-sample smoke: run the pure stats on a small slice of ONE real HNX ticker if present."""
    files = sorted(H.PROCESSED_DIR.glob("*_processed.csv"))
    if not files:  # pragma: no cover - HNX data present in this repo; guard for a data-less checkout
        pytest.skip("no real HNX processed data available")
    f = files[0]
    tk = f.name.replace("_processed.csv", "")
    proc = pd.read_csv(f).head(400)
    rf = H.PRICE_DIR / f"{tk}_ohlcv.csv"
    raw = pd.read_csv(rf).head(400) if rf.exists() else pd.DataFrame()
    row = H.data_stats_for_ticker(tk, proc, raw, "")
    assert row["ticker"] == tk and row["n_valid"] >= 0
    assert np.isfinite(row["zero_park_frac"]) or row["n_valid"] == 0
