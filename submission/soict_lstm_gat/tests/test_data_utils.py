"""TDD tests for data_utils (self-contained submission module).

conftest.py at the submission-folder root puts that folder on sys.path, so the module is
imported by its bare name.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

import data_utils

# Repo root: tests/ -> soict_lstm_gat -> submission -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]
PROCESSED = REPO_ROOT / "data" / "processed"


# --------------------------------------------------------------------------- har_features
def test_har_features_shapes_and_daily_equals_pk():
    pk = np.arange(1, 31, dtype=np.float64)  # 30 values 1..30
    feat = data_utils.har_features(pk)
    assert feat.shape == (30, 3)
    # daily column is pk itself
    np.testing.assert_array_equal(feat[:, 0], pk)


def test_har_features_rolling_nan_boundaries():
    pk = np.arange(1, 31, dtype=np.float64)
    feat = data_utils.har_features(pk)
    weekly, monthly = feat[:, 1], feat[:, 2]
    # weekly valid from index 4 (needs 5 obs)
    assert np.isnan(weekly[:4]).all()
    assert np.isfinite(weekly[4:]).all()
    # monthly valid from index 21 (needs 22 obs)
    assert np.isnan(monthly[:21]).all()
    assert np.isfinite(monthly[21:]).all()


def test_har_features_rolling_values():
    pk = np.arange(1, 31, dtype=np.float64)
    feat = data_utils.har_features(pk)
    # weekly at idx 4 = mean(pk[0:5]) = mean(1..5) = 3.0
    assert feat[4, 1] == pytest.approx(3.0)
    # monthly at idx 21 = mean(pk[0:22]) = mean(1..22) = 11.5
    assert feat[21, 2] == pytest.approx(11.5)


# --------------------------------------------------------------------------- make_windows
def test_make_windows_anchor_range():
    n = 100
    pk = np.arange(1, n + 1, dtype=np.float64)
    lookback, horizon = 10, 1
    anchors = data_utils.make_windows(pk, lookback, horizon)
    assert isinstance(anchors, np.ndarray)
    # first valid anchor = first_valid(21) + lookback - 1 = 30
    assert anchors.min() == 30
    # last anchor = n - horizon - 1 = 98
    assert anchors.max() == n - horizon - 1
    # every window start monthly-valid: t - lookback + 1 >= 21
    assert (anchors - lookback + 1 >= 21).all()
    # target exists for every anchor
    assert (anchors + horizon <= n - 1).all()


# --------------------------------------------------------------------------- per_stock_split
def test_per_stock_split_fractions_and_order():
    anchors = np.arange(1000)
    a_tr, a_va, a_te = data_utils.per_stock_split(anchors, train_frac=0.8, val_frac=0.1)
    assert (len(a_tr), len(a_va), len(a_te)) == (800, 100, 100)
    # chronological, contiguous, no overlap
    assert a_tr[-1] < a_va[0] < a_te[0]
    np.testing.assert_array_equal(np.concatenate([a_tr, a_va, a_te]), anchors)


# --------------------------------------------------------------------------- TickerScaler
def test_ticker_scaler_fit_train_only():
    rng = np.random.default_rng(0)
    train_rows = rng.normal(loc=5.0, scale=2.0, size=(200, 3))
    sc = data_utils.TickerScaler()
    sc.fit_features(train_rows)
    sc.fit_target(rng.normal(size=200))

    # fit uses TRAIN rows only
    np.testing.assert_allclose(sc.f_mean, train_rows.mean(0))
    np.testing.assert_allclose(sc.f_std, train_rows.std(0) + 1e-8)

    m0, s0 = sc.f_mean.copy(), sc.f_std.copy()
    # transforming OTHER rows must not refit / mutate the fitted stats
    other = rng.normal(loc=-10.0, scale=9.0, size=(7, 4, 3))
    out = sc.transform_windows(other)
    assert out.shape == other.shape
    np.testing.assert_array_equal(sc.f_mean, m0)
    np.testing.assert_array_equal(sc.f_std, s0)
    # correct normalization applied
    np.testing.assert_allclose(out, (other - m0) / s0)


def test_ticker_scaler_target_stats():
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    sc = data_utils.TickerScaler()
    sc.fit_target(y)
    assert sc.t_mean == pytest.approx(y.mean())
    assert sc.t_std == pytest.approx(y.std() + 1e-8)


# --------------------------------------------------------------------------- build_pooled (real data)
@pytest.mark.smoke
def test_build_pooled_real_data_smoke():
    files = sorted(glob.glob(str(PROCESSED / "*_processed.csv")))[:2]
    assert len(files) == 2, f"need 2 processed CSVs under {PROCESSED}"

    out = data_utils.build_pooled(files, lookback=10, horizon=1)

    # window geometry + dtype
    assert out["X_tr"].shape[1:] == (10, 3)
    assert out["X_tr"].dtype == np.float32
    assert out["X_va"].shape[1:] == (10, 3)
    assert out["X_te"].shape[1:] == (10, 3)

    # finiteness of all feature tensors
    for key in ("X_tr", "X_va", "X_te"):
        assert np.isfinite(out[key]).all(), f"{key} has non-finite values"

    # target arrays present and finite
    for key in ("y_tr_norm", "y_va_norm", "y_va_raw", "y_te_raw"):
        assert np.isfinite(out[key]).all()

    # test ticker ids all have a scaler
    assert len(out["te_ticker_ids"]) == out["X_te"].shape[0]
    for tid in np.unique(out["te_ticker_ids"]):
        assert int(tid) in out["scalers"]
        t_mean, t_std = out["scalers"][int(tid)]
        assert np.isfinite(t_mean) and t_std > 0

    # target dates aligned + ISO formatted
    assert len(out["te_target_dates"]) == len(out["te_ticker_ids"])
    for d in out["te_target_dates"][:5]:
        assert isinstance(d, str) and len(d) == 10 and d[4] == "-" and d[7] == "-"

    # ticker_order is a non-empty list
    assert isinstance(out["ticker_order"], list) and len(out["ticker_order"]) >= 1
