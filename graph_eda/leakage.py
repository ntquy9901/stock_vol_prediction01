"""Automated leakage / integrity assertions (plan sections 31, 55, 65).

These are called by the runner before results are trusted and are exercised directly by
the test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assert_dates_sorted(df: pd.DataFrame, date_col: str = "date") -> None:
    d = df[date_col]
    if not d.is_monotonic_increasing:
        raise AssertionError("dates are not chronologically sorted")


def assert_no_negative_pk(pk: np.ndarray | pd.Series) -> None:
    v = np.asarray(pk, dtype=float)
    v = v[~np.isnan(v)]
    if np.any(v < 0):
        raise AssertionError("negative Parkinson volatility encountered")


def assert_corr_matrix_valid(mat: pd.DataFrame, tol: float = 1e-8) -> None:
    a = mat.values
    if not np.allclose(np.diag(a), 1.0, atol=1e-6):
        raise AssertionError("correlation diagonal is not ~1")
    if not np.allclose(a, a.T, atol=tol, equal_nan=True):
        raise AssertionError("contemporaneous correlation matrix is not symmetric")


def assert_snapshot_no_lookahead(
    snapshot_dates: pd.DatetimeIndex, prediction_date: pd.Timestamp
) -> None:
    """No observation used to build a snapshot may be after the prediction date."""
    if len(snapshot_dates) and snapshot_dates.max() > prediction_date:
        raise AssertionError("graph snapshot uses observations after prediction date")


def assert_target_after_features(
    feature_cutoff: pd.Timestamp, target_date: pd.Timestamp
) -> None:
    if not target_date > feature_cutoff:
        raise AssertionError("target date is not strictly after the feature cutoff")
