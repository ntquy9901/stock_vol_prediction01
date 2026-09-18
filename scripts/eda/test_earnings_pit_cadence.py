"""Unit + real-data tests for the PIT earnings-cadence schedule and its actual-vs-PIT discrepancy check."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from earnings_pit_cadence import (  # noqa: E402
    MIN_HISTORY, _load_edates, pit_cadence, pit_vs_actual, predict_schedule, summarize,
)


def _quarterly(n, start="2018-01-31", step=91):
    d0 = np.datetime64(start)
    return np.array([d0 + np.timedelta64(step * i, "D") for i in range(n)], dtype="datetime64[D]")


# ---- predict_schedule / pit_cadence ----

def test_short_series_passthrough_and_dtype():
    dates = _quarterly(MIN_HISTORY)                          # <= MIN_HISTORY -> unchanged branch
    out = pit_cadence({"AAA": dates})
    assert out["AAA"].dtype == np.dtype("datetime64[ns]")   # matches the panel builder's dtype
    assert np.array_equal(out["AAA"].astype("datetime64[D]"), dates)


def test_regular_cadence_predicts_actual_exactly():
    dates = _quarterly(8, step=91)                          # perfectly regular -> median gap 91 = actual
    pred = predict_schedule(dates)
    assert np.array_equal(pred[:MIN_HISTORY], dates[:MIN_HISTORY])    # first MIN_HISTORY untouched
    assert np.array_equal(pred[MIN_HISTORY:], dates[MIN_HISTORY:])    # loop reproduces regular dates


def test_prediction_uses_only_prior_gaps_causal():
    # gaps [30, 90, 90]: causal median(gaps[:i-1])=[30,90]->60d; leaky median(gaps[:i])=[30,90,90]->90d
    # would need the gap TO event i. This case distinguishes causal from the off-by-one look-ahead leak.
    base = np.datetime64("2020-01-01")
    d = np.array([base, base + np.timedelta64(30, "D"), base + np.timedelta64(120, "D"),
                  base + np.timedelta64(210, "D")], dtype="datetime64[D]")   # gaps 30, 90, 90
    pred = predict_schedule(d)
    assert pred[MIN_HISTORY] == d[MIN_HISTORY - 1] + np.timedelta64(60, "D")   # median([30,90]) = strictly prior
    assert pred[MIN_HISTORY] != d[MIN_HISTORY - 1] + np.timedelta64(90, "D")   # not the leaky median([30,90,90])


def test_multiple_tickers_and_list_input():
    out = pit_cadence({"AAA": list(_quarterly(6)), "BBB": _quarterly(2)})
    assert set(out) == {"AAA", "BBB"}
    assert len(out["BBB"]) == 2                             # short ticker preserved


# ---- pit_vs_actual / summarize ----

def test_pit_vs_actual_schema_and_regular_zero_error():
    disc = pit_vs_actual({"AAA": _quarterly(6, step=91), "SHORT": _quarterly(2)})
    assert list(disc.columns) == ["ticker", "event_index", "actual_date", "predicted_date",
                                  "abs_err_days", "signed_err_days"]
    assert set(disc["ticker"]) == {"AAA"}                   # SHORT (<=MIN_HISTORY) emits no rows
    assert len(disc) == 6 - MIN_HISTORY                     # one row per predicted event
    assert (disc["abs_err_days"] == 0).all()               # regular cadence -> exact
    assert (disc["abs_err_days"] >= 0).all()


def test_pit_vs_actual_signed_error_on_delay():
    d = _quarterly(4, step=90)
    d = np.sort(np.append(d, d[-1] + np.timedelta64(400, "D")).astype("datetime64[D]"))
    disc = pit_vs_actual({"AAA": d})
    late = disc[disc["event_index"] == 4].iloc[0]           # actual later than predicted -> negative sign
    assert late["signed_err_days"] < 0
    assert late["abs_err_days"] == abs(late["signed_err_days"])


def test_summarize_empty_and_nonempty():
    assert summarize(pit_vs_actual({"SHORT": _quarterly(2)})) == {"n_events": 0, "n_tickers": 0}
    s = summarize(pit_vs_actual({"AAA": _quarterly(8, step=91)}))
    assert s["n_events"] == 8 - MIN_HISTORY and s["n_tickers"] == 1
    assert s["median_abs_days"] == 0.0 and s["within_1d_pct"] == 100.0


# ---- real-data smoke: runs over the full crawled earnings history for both markets ----

@pytest.mark.parametrize("market", ["hose", "sp500"])
def test_real_data_discrepancy_runs(market):
    edates = _load_edates(market)
    disc = pit_vs_actual(edates)
    assert len(disc) > 0                                    # both markets have multi-event tickers
    assert (disc["abs_err_days"] >= 0).all()               # non-negative by construction
    assert disc["predicted_date"].notna().all() and disc["actual_date"].notna().all()
    s = summarize(disc)
    assert 0 <= s["within_14d_pct"] <= 100 and s["n_tickers"] >= 1
