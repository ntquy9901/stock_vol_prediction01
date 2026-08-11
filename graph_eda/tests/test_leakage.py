"""Plan sections 31/55/65: leakage-boundary and correlation-matrix assertions."""

import numpy as np
import pandas as pd
import pytest

from graph_eda import leakage


def test_corr_symmetry_pass_and_fail():
    m = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["a", "b"], columns=["a", "b"])
    leakage.assert_corr_matrix_valid(m)  # symmetric, unit diag -> ok
    bad = pd.DataFrame([[1.0, 0.5], [0.4, 1.0]], index=["a", "b"], columns=["a", "b"])
    with pytest.raises(AssertionError):
        leakage.assert_corr_matrix_valid(bad)


def test_snapshot_no_lookahead():
    dates = pd.DatetimeIndex(["2021-01-01", "2021-01-04", "2021-01-05"])
    leakage.assert_snapshot_no_lookahead(dates, pd.Timestamp("2021-01-05"))
    with pytest.raises(AssertionError):
        leakage.assert_snapshot_no_lookahead(dates, pd.Timestamp("2021-01-04"))


def test_target_after_features():
    leakage.assert_target_after_features(
        pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")
    )
    with pytest.raises(AssertionError):
        leakage.assert_target_after_features(
            pd.Timestamp("2021-01-05"), pd.Timestamp("2021-01-05")
        )


def test_no_negative_pk_raises():
    with pytest.raises(AssertionError):
        leakage.assert_no_negative_pk(np.array([0.1, -0.2, 0.3]))


def test_dates_sorted_raises():
    df = pd.DataFrame({"date": pd.to_datetime(["2021-01-02", "2021-01-01"])})
    with pytest.raises(AssertionError):
        leakage.assert_dates_sorted(df)
