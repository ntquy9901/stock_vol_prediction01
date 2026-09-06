"""Unit tests for the causal limit-lock features (TEST-FIRST)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))

import limitlock_features as LL  # noqa: E402


def test_shape_and_empty():
    out = LL.compute_limitlock_features([0.0, -0.07, -0.07], [False, True, True],
                                        window=2, limit_frac=0.065, near_mult=0.9)
    assert out.shape == (3, 3)
    assert LL.compute_limitlock_features([], [], window=5, limit_frac=0.065, near_mult=0.9).shape == (0, 3)


def test_limit_down_streak_counts_and_resets():
    rets = [-0.07, -0.08, -0.02, -0.07]     # limit-down, limit-down, normal, limit-down
    out = LL.compute_limitlock_features(rets, [False] * 4, window=5, limit_frac=0.065, near_mult=0.9)
    streak = out[:, 0]
    assert list(streak) == [1.0, 2.0, 0.0, 1.0]


def test_recent_lock_freq_trailing_window():
    locks = [False, True, True, False]
    out = LL.compute_limitlock_features([0.0] * 4, locks, window=2, limit_frac=0.065, near_mult=0.9)
    # trailing fraction over window=2, min_periods=1: [0/1, 1/2, 2/2, 1/2]
    assert np.allclose(out[:, 1], [0.0, 0.5, 1.0, 0.5])


def test_near_limit_freq_uses_absolute_return():
    rets = [0.0, 0.06, -0.06, 0.0]          # 0.06 >= 0.065*0.9 = 0.0585 -> near-limit
    out = LL.compute_limitlock_features(rets, [False] * 4, window=2, limit_frac=0.065, near_mult=0.9)
    assert np.allclose(out[:, 2], [0.0, 0.5, 1.0, 0.5])


def test_causal_no_lookahead():
    rng = np.random.default_rng(1)
    rets = rng.standard_normal(80) * 0.05
    locks = rng.random(80) < 0.1
    full = LL.compute_limitlock_features(rets, locks, window=22, limit_frac=0.065, near_mult=0.9)
    prefix = LL.compute_limitlock_features(rets[:40], locks[:40], window=22, limit_frac=0.065, near_mult=0.9)
    assert np.allclose(full[:40], prefix, atol=1e-12)


def test_nan_returns_and_flags_finite():
    out = LL.compute_limitlock_features([np.nan, -0.07, np.nan, 0.0], [False, np.nan, True, False],
                                        window=3, limit_frac=0.065, near_mult=0.9)
    assert out.shape == (4, 3) and np.isfinite(out).all()
    # NaN return is treated as no move (not a limit-down), so streak stays reset there
    assert out[0, 0] == 0.0
