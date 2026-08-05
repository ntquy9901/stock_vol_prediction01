"""Smoke + unit tests for sentiment_price_eda (pure + statistical functions).

Run:
    pytest baselines/2026-07-11_sentiment_price_eda/test/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Bootstrap import path (baseline folder name has dashes -> not a package import)
CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import sentiment_price_eda as eda  # noqa: E402


# --- winsorize -------------------------------------------------------------
def test_winsorize_preserves_shape_and_nans():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 100.0, np.nan])
    out = eda.winsorize(arr, 10, 90)
    assert out.shape == arr.shape
    assert np.isnan(out[-1])              # NaN preserved
    assert out[4] < 100.0                 # outlier clipped down


def test_winsorize_all_nan_safe():
    out = eda.winsorize(np.array([np.nan, np.nan]), 1, 99)
    assert np.all(np.isnan(out))


# --- forward returns (pure shift, no winsorize) ----------------------------
def test_forward_return_values_and_tail_nan():
    price = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "close": [100.0, 110.0, 121.0],
    })
    fr = eda.compute_forward_returns(price, [1, 2])
    assert len(fr) == 3
    np.testing.assert_allclose(fr["ret_1d"].to_numpy()[:2], [0.10, 0.10], rtol=1e-9)
    assert np.isnan(fr["ret_1d"].to_numpy()[2])           # last day has no tomorrow
    assert abs(fr["ret_2d"].to_numpy()[0] - 0.21) < 1e-9
    assert np.isnan(fr["ret_2d"].to_numpy()[1])


def test_forward_return_zero_close_no_inf():
    """A zero close must not produce inf; any return touching it becomes NaN."""
    price = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "close": [100.0, 0.0, 110.0],
    })
    fr = eda.compute_forward_returns(price, [1])
    arr = fr["ret_1d"].to_numpy()
    assert not np.any(np.isinf(arr))                 # no inf leaked
    # close[1]=0 is guarded to NaN -> ret at day0 (numerator) and day1 (denominator) are NaN
    assert np.isnan(arr[0]) and np.isnan(arr[1])


def test_property_forward_return_matches_manual_raw_shift():
    """compute_forward_returns is a pure close-shift (no clipping). Verify against raw formula."""
    rng = np.random.default_rng(0)
    closes = np.cumprod(1 + rng.normal(0, 0.01, 50)) * 100
    price = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=50), "close": closes})
    fr = eda.compute_forward_returns(price, [5])
    k = 5
    manual = closes[k:] / closes[:-k] - 1.0
    got = fr["ret_5d"].to_numpy()[: len(manual)]
    np.testing.assert_allclose(got, manual, rtol=1e-9)


# --- classify --------------------------------------------------------------
def test_classify_thresholds():
    assert eda.classify(0.5) == "pos"
    assert eda.classify(0.2) == "neu"      # boundary is neutral (strict >)
    assert eda.classify(-0.2) == "neu"     # boundary is neutral (strict <)
    assert eda.classify(-0.5) == "neg"
    assert eda.classify(np.nan) == "neu"


# --- statistical functions -------------------------------------------------
def _make_events(pos_ret, neg_ret, ticker="T"):
    rows = []
    for r in pos_ret:
        rows.append({"ticker": ticker, "grp": "pos", "sentiment_1d": 0.5, "ret_5d": r})
    for r in neg_ret:
        rows.append({"ticker": ticker, "grp": "neg", "sentiment_1d": -0.5, "ret_5d": r})
    return pd.DataFrame(rows)


def test_mann_whitney_detects_known_difference():
    rng = np.random.default_rng(1)
    # pos returns clearly higher than neg returns
    ev = _make_events(rng.normal(0.05, 0.01, 30), rng.normal(-0.05, 0.01, 30))
    mw = eda.mann_whitney_pos_neg(ev, [5]).iloc[0]
    assert mw["spread_bp"] > 0
    assert mw["p_raw"] < 0.05
    assert mw["p_demeaned"] < 0.05


def test_mann_whitney_no_difference_not_significant():
    rng = np.random.default_rng(2)
    base = rng.normal(0.0, 0.01, 30)
    ev = _make_events(base, rng.normal(0.0, 0.01, 30))
    mw = eda.mann_whitney_pos_neg(ev, [5]).iloc[0]
    assert mw["p_raw"] > 0.05


def test_spearman_safe_degenerate_returns_nan():
    s = pd.Series([0.5, 0.5, 0.5, 0.5])      # no variation in sentiment
    r = pd.Series([0.1, 0.2, -0.1, 0.0])
    assert np.isnan(eda._spearman_safe(s, r))


def test_neg_composition_counts():
    ev = pd.DataFrame({"ticker": ["A", "A", "B", "C", "C", "C", "D"],
                       "grp": ["neg", "pos", "neg", "neg", "neg", "pos", "pos"],
                       "sentiment_1d": [-0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5]})
    comp = eda.neg_composition(ev)
    assert comp["n_neg_events_total"] == 4
    assert comp["n_tickers_with_neg"] == 3      # A, B, C
    assert comp["n_tickers_with_ge2_neg"] == 1  # only C (2 events)
    assert comp["top_contributors"]["C"] == 2
