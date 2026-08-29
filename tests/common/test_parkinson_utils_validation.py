"""OHLC-geometry validation tests for the ACTIVE loader boundary
``src.common.parkinson_utils.validate_ohlc_data`` (used by
``src.common.process_parkinson_pipeline.process_single_stock``).

Non-negativity + NaN were already enforced; these tests pin down the four
OHLC ordering inequalities (high>=low, high>=open, high>=close, low<=open,
low<=close) so impossible geometry cannot silently produce a finite
Parkinson variance (follow-up code review HIGH-03, 2026-08-29).
"""
import numpy as np
import pandas as pd
import pytest

from src.common.parkinson_utils import validate_ohlc_data


def _frame(open_, high, low, close, volume=1000.0):
    """One-row OHLCV frame with the given prices (all otherwise valid)."""
    return pd.DataFrame(
        {"open": [open_], "high": [high], "low": [low], "close": [close], "volume": [volume]}
    )


def test_validate_accepts_valid_geometry():
    df = _frame(open_=10.0, high=12.0, low=9.0, close=11.0)
    assert validate_ohlc_data(df) is True


def test_validate_rejects_high_below_low():
    # open=close=low=10, high=9 — the exact probe from the review
    with pytest.raises(ValueError, match="geometry"):
        validate_ohlc_data(_frame(open_=10.0, high=9.0, low=10.0, close=10.0))


def test_validate_rejects_high_below_open():
    with pytest.raises(ValueError, match="geometry"):
        validate_ohlc_data(_frame(open_=11.0, high=10.5, low=9.0, close=10.0))


def test_validate_rejects_high_below_close():
    with pytest.raises(ValueError, match="geometry"):
        validate_ohlc_data(_frame(open_=10.0, high=10.5, low=9.0, close=11.0))


def test_validate_rejects_low_above_open():
    with pytest.raises(ValueError, match="geometry"):
        validate_ohlc_data(_frame(open_=9.0, high=12.0, low=9.5, close=11.0))


def test_validate_rejects_low_above_close():
    with pytest.raises(ValueError, match="geometry"):
        validate_ohlc_data(_frame(open_=11.0, high=12.0, low=10.5, close=10.0))


def test_valid_geometry_at_bounds_open_close_equal_extremes():
    # open==low and close==high is valid (bar where the extremes are the open/close)
    df = _frame(open_=9.0, high=12.0, low=9.0, close=12.0)
    assert validate_ohlc_data(df) is True


def test_multi_row_one_bad_row_raises():
    df = pd.DataFrame(
        {
            "open": [10.0, 10.0],
            "high": [12.0, 9.0],   # 2nd row high<low
            "low": [9.0, 10.0],
            "close": [11.0, 10.0],
            "volume": [1000.0, 1000.0],
        }
    )
    with pytest.raises(ValueError, match="geometry"):
        validate_ohlc_data(df)


def test_validate_tolerates_floating_point_noise():
    # adjusted-price fp noise: low above high by ~1e-13 of the price scale -> accepted
    df = _frame(open_=100.0, high=100.0, low=100.0 + 1e-11, close=100.0)
    assert validate_ohlc_data(df) is True


def test_validate_still_rejects_material_violation_above_tolerance():
    # low above high by ~0.5% (>> 1e-6 relative tol) -> rejected
    df = _frame(open_=100.0, high=100.0, low=100.5, close=100.0)
    with pytest.raises(ValueError, match="geometry"):
        validate_ohlc_data(df)


def test_geometry_check_does_not_break_finite_variance_path():
    # Sanity: a valid frame still yields a finite Parkinson variance downstream.
    from src.common.parkinson_utils import calculate_parkinson_volatility

    df = _frame(open_=10.0, high=12.0, low=9.0, close=11.0)
    validate_ohlc_data(df)
    v = calculate_parkinson_volatility(df)
    assert np.isfinite(v.iloc[0])
