"""Tests for the S&P500 cross-market runner's pure logic (no training / no network).

Run (GPU venv, torch present):  python -m pytest scripts/sp500_crossmarket/test_run_sp500_crossmarket.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

rc = pytest.importorskip("run_sp500_crossmarket", reason="needs torch (run under the GPU venv)")


def test_qlike_matches_formula():
    t = np.array([0.5, 0.2, 1e-12])   # last is below floor
    p = np.array([0.5, 0.1, 0.3])
    got = rc.qlike(t, p)
    tf = np.maximum(t, rc.QLIKE_FLOOR); pf = np.maximum(p, rc.QLIKE_FLOOR)
    r = tf / pf
    assert np.allclose(got, r - np.log(r) - 1.0)
    assert got[0] == pytest.approx(0.0)   # perfect prediction -> QLIKE 0


def test_make_windows_is_contiguous_valid_range():
    n = 100
    feats = np.zeros((n, 3)); pk = np.ones(n)
    a = rc.make_windows(feats, pk, horizon=1)
    # anchors are t in [first_valid + SEQ-1 .. n-h-1] = [42 .. 98]
    assert a[0] == 21 + (rc.SEQ - 1) and a[-1] == n - 1 - 1
    assert a == list(range(42, n - 1))
    # every anchor leaves a full monthly-valid 22-day window and a target at t+h
    for t in a:
        assert t - (rc.SEQ - 1) >= 21 and (t + 1) < n


def test_make_windows_shrinks_with_horizon():
    n = 100
    feats = np.zeros((n, 3)); pk = np.ones(n)
    assert rc.make_windows(feats, pk, 1)[-1] == 98
    assert rc.make_windows(feats, pk, 22)[-1] == n - 22 - 1   # target t+22 must exist


def test_build_ticker_features(tmp_path):
    n = 260
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "parkinson_volatility": np.linspace(0.1, 0.2, n)})
    p = tmp_path / "AAA_processed.csv"
    df.to_csv(p, index=False)
    tk, feats, pk = rc.build_ticker_features(str(p))
    assert tk == "AAA"
    assert feats.shape == (n, 3) and pk.shape == (n,)
    assert np.isnan(feats[:4, 1]).all() and not np.isnan(feats[4, 1])     # weekly valid from idx 4
    assert np.isnan(feats[:21, 2]).all() and not np.isnan(feats[21, 2])   # monthly valid from idx 21
    assert feats[10, 0] == pytest.approx(pk[10])                          # daily == pk(t)


def test_build_ticker_features_too_short_returns_none(tmp_path):
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=150),
                       "parkinson_volatility": np.ones(150)})
    p = tmp_path / "SHORT_processed.csv"
    df.to_csv(p, index=False)
    assert rc.build_ticker_features(str(p)) is None   # < 200 rows
