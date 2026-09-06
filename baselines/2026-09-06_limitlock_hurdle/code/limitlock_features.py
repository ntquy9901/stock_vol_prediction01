"""Causal limit-lock features for the hurdle model.

Given one ticker's daily-return series and its ``zero_range_flag`` (limit-lock) series, compute features
that flag price-limit pressure using ONLY information up to each day t (no look-ahead). On a HOSE
price-limit day the high-low range collapses, so the Parkinson variance target is ~0; these features let
a classifier anticipate such a day and pull the variance forecast to the floor.

Features per day t (all causal):
  * ``limit_down_streak`` : consecutive days with ``daily_return <= -limit_frac`` ending at t (reset to 0
    on a non-limit-down day) -- a run of limit-down days is the classic pre-lock pattern.
  * ``recent_lock_freq``  : trailing fraction of ``zero_range_flag`` over ``window`` (min_periods=1).
  * ``near_limit_freq``   : trailing fraction of ``|daily_return| >= limit_frac*near_mult`` over ``window``.

Prefix-invariance (features on days 0..k unchanged when future days are appended) is unit-tested. NaN
returns are treated as no move (0.0); NaN lock flags as not-locked (False) -- these keep the features
finite and never let a future value leak backwards.
"""
from __future__ import annotations

import numpy as np


def _clean_returns(returns):
    """1-D float returns with NaN/inf -> 0.0 (a non-finite return is treated as 'no move')."""
    r = np.asarray(returns, dtype=float)
    return np.where(np.isfinite(r), r, 0.0)


def _clean_flags(flags):
    """1-D 0/1 float lock flags with NaN -> 0.0 (treated as not-locked)."""
    f = np.asarray(flags, dtype=float)
    return np.where(np.isfinite(f), f, 0.0)


def _trailing_fraction(indicator, window):
    """Causal trailing mean of a 0/1 indicator over ``window`` days (min_periods=1): element t uses
    ``indicator[max(0,t-window+1)..t]``."""
    n = len(indicator)
    out = np.empty(n)
    for t in range(n):
        out[t] = indicator[max(0, t - window + 1):t + 1].mean()
    return out


def compute_limitlock_features(returns, lock_flags, window, limit_frac, near_mult):
    """Return an ``[T, 3]`` causal feature array ``[limit_down_streak, recent_lock_freq, near_limit_freq]``.

    ``window`` = trailing window for the two frequencies; ``limit_frac`` = the (near-)limit return
    threshold; ``near_mult`` scales it for the near-limit band. Empty input -> ``(0, 3)``."""
    r = _clean_returns(returns)
    f = _clean_flags(lock_flags)
    n = len(r)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    limit_down = (r <= -limit_frac).astype(float)
    near_limit = (np.abs(r) >= limit_frac * near_mult).astype(float)

    streak = np.empty(n)
    run = 0.0
    for t in range(n):
        run = run + 1.0 if limit_down[t] else 0.0
        streak[t] = run

    recent_lock_freq = _trailing_fraction(f, window)
    near_limit_freq = _trailing_fraction(near_limit, window)
    return np.column_stack([streak, recent_lock_freq, near_limit_freq]).astype(np.float32)
