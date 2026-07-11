"""Smoke tests for sentiment_market_eda (lag-correlation correctness)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import sentiment_market_eda as mkt  # noqa: E402


def test_spearman_lag_shift_direction():
    """corr(x[T], y[T+lag]) — verify lag alignment, not reversed."""
    x = pd.Series(np.arange(1.0, 41.0))           # monotonic ramp, >= MIN_PAIRS(30)
    # lag-0 autocorrelation of a ramp is 1
    assert abs(mkt.spearman_lag(x, x, lag=0) - 1.0) < 1e-9
    # lag-1: corr(x[T], x[T+1]) ~ 1 for a ramp
    assert mkt.spearman_lag(x, x, lag=1) > 0.95
    # anti-correlated series at lag 0
    y_desc = pd.Series(np.arange(1.0, 41.0)[::-1])
    assert mkt.spearman_lag(x, y_desc, lag=0) < -0.95


def test_spearman_lag_too_few_pairs_returns_nan():
    x = pd.Series([1.0, 2.0, 3.0])     # below MIN_PAIRS
    y = pd.Series([1.0, 2.0, 3.0])
    assert np.isnan(mkt.spearman_lag(x, y, lag=0))


def test_spearman_lag_constant_series_returns_nan():
    x = pd.Series([5.0] * 40)          # no variation
    y = pd.Series(np.arange(40.0))
    assert np.isnan(mkt.spearman_lag(x, y, lag=0))


def test_event_study_returns_structure():
    """Synthetic panel: news days have higher next-day vol -> ratio>1, p<0.05."""
    rng = np.random.default_rng(0)
    n = 400
    has_news = rng.integers(0, 2, n)
    # next-day vol higher when has_news
    vol = np.where(has_news == 1, rng.normal(0.03, 0.002, n), rng.normal(0.01, 0.002, n))
    panel = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n),
        "has_news": has_news,
        "mkt_fwd_vol_22d": vol,
        "mkt_abs_ret_1d": vol,
        "mkt_vol_avg": vol,
    })
    out = mkt.event_study_news_vs_nonews(panel)
    assert out["fwd_vol_22d"]["ratio_news_over_none"] > 1.0
    assert out["fwd_vol_22d"]["p_value"] < 0.05
