"""Unit tests for the causal market-regime features."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

import regime_features as RF  # noqa: E402


def test_shape_and_empty():
    out = RF.compute_regime_features([1.0, 2.0, 3.0], window=2)
    assert out.shape == (3, 3)
    assert RF.compute_regime_features([], window=5).shape == (0, 3)


def test_causal_no_lookahead():
    # features for the first k days must not change when future days are appended
    rng = np.random.default_rng(0)
    v = np.abs(rng.standard_normal(60)) + 0.1
    full = RF.compute_regime_features(v, window=10)
    prefix = RF.compute_regime_features(v[:30], window=10)
    assert np.allclose(full[:30], prefix, atol=1e-9)


def test_vol_z_flags_extreme_day():
    v = [1.0] * 20 + [50.0]          # last day is a huge spike
    out = RF.compute_regime_features(v, window=10)
    assert out[-1, 0] > 2.0           # vol_z large on the spike day


def test_regime_and_days_since_change():
    v = [1.0] * 10 + [100.0] * 10     # low then high regime
    out = RF.compute_regime_features(v, window=5)
    assert out[-1, 1] == 1.0          # high-vol regime at the end
    # days-since-change resets to 0 at some flip point then increases
    dsc = out[:, 2]
    assert (dsc == 0).sum() >= 1 and dsc[-1] >= 1


def test_nan_handled():
    out = RF.compute_regime_features([1.0, np.nan, 2.0, np.nan, 3.0], window=3)
    assert out.shape == (5, 3) and np.isfinite(out).all()
