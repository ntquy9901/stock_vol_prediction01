"""Unit tests for the strictly-causal PIT earnings-cadence schedule (paper leakage-safe robustness)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from earnings_pit_cadence import MIN_HISTORY, pit_cadence  # noqa: E402


def _quarterly(n, start="2018-01-31", step=91):
    d0 = np.datetime64(start)
    return np.array([d0 + np.timedelta64(step * i, "D") for i in range(n)], dtype="datetime64[D]")


def test_short_series_passthrough_and_dtype():
    dates = _quarterly(MIN_HISTORY)                          # exactly MIN_HISTORY -> unchanged branch
    out = pit_cadence({"AAA": dates})
    assert out["AAA"].dtype == np.dtype("datetime64[ns]")   # matches the panel builder's dtype
    assert np.array_equal(out["AAA"].astype("datetime64[D]"), dates)


def test_regular_cadence_predicts_actual_exactly():
    dates = _quarterly(8, step=91)                          # perfectly regular -> median gap 91 = actual
    out = pit_cadence({"AAA": dates})["AAA"].astype("datetime64[D]")
    assert len(out) == len(dates)
    assert np.array_equal(out[:MIN_HISTORY], dates[:MIN_HISTORY])       # first MIN_HISTORY untouched
    assert np.array_equal(out[MIN_HISTORY:], dates[MIN_HISTORY:])       # loop reproduces regular dates


def test_prediction_uses_only_prior_gaps_causal():
    d = _quarterly(4, step=90)
    d = np.sort(np.append(d, d[-1] + np.timedelta64(400, "D")).astype("datetime64[D]"))  # a reschedule
    out = pit_cadence({"AAA": d})["AAA"].astype("datetime64[D]")
    step = int(np.median(np.diff(d[:MIN_HISTORY]).astype(int)))
    assert out[MIN_HISTORY] == d[MIN_HISTORY - 1] + np.timedelta64(step, "D")   # uses only prior gaps
    assert out[4] != d[4]                                   # the 400-day jump is not foreseen -> differs


def test_multiple_tickers_and_list_input():
    out = pit_cadence({"AAA": list(_quarterly(6)), "BBB": _quarterly(2)})
    assert set(out) == {"AAA", "BBB"}
    assert len(out["BBB"]) == 2                             # short ticker preserved
