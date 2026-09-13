"""Daily-frequency realized semivariance (the only feature not already precomputed in the enriched frames).

Barndorff-Nielsen, Kinnebrock & Shephard (2010); Patton & Sheppard (2015) "Good volatility, bad volatility".
At daily frequency each day contributes one signed return; the downside/upside semivariance over a trailing
window is the mean of squared negative / positive daily returns. (Garman-Klass, Rogers-Satchell, Yang-Zhang and
Parkinson variances are already precomputed + formula-tested in the enriched frames and are used from there.)
"""
import numpy as np
import pandas as pd


def semivariance(ret, window, min_periods):
    """Trailing downside/upside realized semivariance of daily log-returns ``ret``.

    Returns ``(rs_minus, rs_plus)`` = rolling mean over ``window`` days of ``ret**2 * 1(ret<0)`` and
    ``ret**2 * 1(ret>0)``; NaN before ``min_periods`` observations. Causal (trailing)."""
    r = pd.Series(np.asarray(ret, float))
    neg = (r ** 2).where(r < 0, 0.0).where(r.notna())   # a NaN return stays NaN (excluded), not counted as 0
    pos = (r ** 2).where(r > 0, 0.0).where(r.notna())
    rm = neg.rolling(window, min_periods=min_periods).mean().to_numpy()
    rp = pos.rolling(window, min_periods=min_periods).mean().to_numpy()
    return rm, rp
