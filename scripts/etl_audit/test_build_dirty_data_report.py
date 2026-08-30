"""TDD + smoke tests for the dirty-data report driver (scripts/etl_audit/build_dirty_data_report.py).

Covers the pure aggregation / measurement / spec logic on synthetic frames (deterministic, no real-data
dependency) and a real-data-sample end-to-end run() smoke that skips cleanly when a market's data is absent.
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

import build_dirty_data_report as B  # noqa: E402


def _frame(rows):
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def _clean_frame(n=6, start_close=10.0):
    rows = [(f"2020-01-{i+1:02d}", start_close, start_close + 0.2, start_close - 0.2, start_close + 0.05, 100)
            for i in range(n)]
    return _frame(rows)


def test_raw_parkinson_matches_formula_and_nan_on_bad_geometry():
    df = _frame([("2020-01-01", 10, 12, 8, 10, 100),
                 ("2020-01-02", 10, 8, 12, 10, 100)])   # high<low -> NaN
    pk = B.raw_parkinson(df)
    expected = np.log(12 / 8) ** 2 / (4 * np.log(2.0))
    assert pk[0] == pytest.approx(expected, rel=1e-9)
    assert np.isnan(pk[1])


def test_clip_evidence_none_processed():
    df = _clean_frame()
    assert B.clip_evidence(df, None)["has_processed"] is False


def test_clip_evidence_detects_upper_clip():
    # raw Parkinson on 2020-01-01 is huge (H=100,L=1) -> > 0.1; processed value pinned at 0.1 -> clipped.
    raw = _frame([("2020-01-01", 10, 100, 1, 10, 100), ("2020-01-02", 10, 10.2, 9.8, 10, 100)])
    proc = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                         "parkinson_volatility": [0.1, 0.0]})
    ce = B.clip_evidence(raw, proc)
    assert ce["has_processed"] is True
    assert ce["n_at_cap"] == 1
    assert ce["n_clipped_from_raw"] == 1        # raw pk on day 1 exceeds 0.1
    assert ce["n_zero_processed"] == 1
    assert ce["proc_max"] == pytest.approx(0.1)


def test_aggregate_frames_totals_and_worst():
    dirty = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                    ("2020-01-02", 10, 8, 12, 10, 100),   # high<low
                    ("2020-01-03", 13, 11, 9, 10, 0)])    # open>high + zero volume
    clean = _clean_frame()
    s = B.aggregate_frames([("DIRTY", dirty, None), ("CLEAN", clean, None)])
    assert s["n_tickers"] == 2
    assert s["totals"]["high_lt_low"] == 1
    assert s["totals"]["zero_volume"] == 1
    assert s["worst"]["high_lt_low"][0]["ticker"] == "DIRTY"
    assert s["clip"]["has_processed"] is False


def test_aggregate_frames_with_processed_pools_clip():
    raw = _frame([("2020-01-01", 10, 100, 1, 10, 100), ("2020-01-02", 10, 10.2, 9.8, 10, 100)])
    proc = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                         "parkinson_volatility": [0.1, 0.02]})
    s = B.aggregate_frames([("T", raw, proc)])
    assert s["clip"]["has_processed"] is True
    assert s["clip"]["n_clipped_from_raw"] == 1


def test_build_executive_table_has_all_classes_and_flags():
    s = B.aggregate_frames([("CLEAN", _clean_frame(), None)])
    tbl = B.build_executive_table(s)
    assert [r["issue"] for r in tbl] == B.CLASSES
    # open_close_outside is cosmetic for Parkinson; nonpositive is REAL
    m = {r["issue"]: r for r in tbl}
    assert m["open_close_outside"]["target_affecting"].startswith("cosmetic")
    assert m["nonpositive"]["target_affecting"].startswith("REAL")
    # split jumps do NOT move the Parkinson target (scale-invariant) -> cosmetic (code review 2026-08-30)
    assert m["split_jumps"]["target_affecting"].startswith("cosmetic")
    assert "scale-invariant" in m["split_jumps"]["estimators"]


def test_clip_evidence_missing_date_column_returns_no_processed():
    raw = _clean_frame()
    proc = pd.DataFrame({"parkinson_volatility": [0.1, 0.02]})   # has target col but NO date column
    assert B.clip_evidence(raw, proc)["has_processed"] is False


def test_build_spec_md_covers_every_class_and_both_processed_states():
    with_proc = {"totals": {k: 1 for k in B.CLASSES}, "n_tickers": 3, "total_rows": 100,
                 "clip": {"has_processed": True, "n_processed": 100, "proc_max": 0.1, "n_at_cap": 5,
                          "n_clipped_from_raw": 4, "n_zero_processed": 10}}
    no_proc = {"totals": {k: 0 for k in B.CLASSES}, "n_tickers": 2, "total_rows": 50,
               "clip": {"has_processed": False}}
    md = B.build_spec_md({"hnx": with_proc, "vn30": no_proc})
    for k in B.CLASSES:
        assert k in md
    assert "clipped-from-raw" in md
    assert "(no processed)" in md      # the no-processed branch rendered


def test_example_dates_all_branches():
    df = _frame([("2020-01-01", 10, 11, 9, 10, 100),
                 ("2020-01-02", 13, 11, 9, 10, 100),     # open>high (oc + magnitude tuple)
                 ("2020-01-03", 16, 21, 15, 20, 100)])   # +100% jump, geometry clean
    res = B.DD.detect_all(df)
    assert B._example_dates(res, "open_close_outside") == ["2020-01-02"]
    assert B._example_dates(res, "split_jumps") == ["2020-01-03"]
    assert B._example_dates(res, "leading_backfill") == []
    # stale + default (list) branches
    stale_df = _frame([(f"2020-02-{i+1:02d}", 12, 13, 11, 12, 100) for i in range(6)])
    sres = B.DD.detect_all(stale_df)
    assert B._example_dates(sres, "stale_runs")[0] == "2020-02-01"
    assert isinstance(B._example_dates(res, "nonpositive"), list)


@pytest.mark.smoke
@pytest.mark.parametrize("panel", B.PANELS)
def test_run_real_data_smoke(panel, tmp_path):
    files = glob.glob(str(B.RAW[panel] / "*_ohlcv.csv"))
    if not files:  # pragma: no cover - depends on which market data is present locally
        pytest.skip(f"no raw data for {panel}")
    res = B.run(panels=[panel], limit=6, out_dir=tmp_path)
    html_path = Path(res["written"][0])
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    # self-contained: charts embedded, no external CDN / http(s) resource references
    assert "data:image/png;base64," in html
    assert "http://" not in html and "https://" not in html
    assert "Executive dirty-data summary" in html
    # spec md written with every class
    spec = Path(res["written"][-1]).read_text(encoding="utf-8")
    for k in B.CLASSES:
        assert k in spec
