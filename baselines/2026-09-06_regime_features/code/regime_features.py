"""Causal market-regime features for shock-aware volatility forecasting (Tier-1 of the shock-detection research).

Given a daily market-level volatility series (e.g. the market Parkinson variance), compute features that flag
the volatility regime and its shifts, using ONLY information up to each day $t$ (no look-ahead). The research
(regime-switching HAR; Marcucci 2005 MRS-GARCH) found regime/change-point signals are the highest-leverage,
lowest-effort add for daily data, since the high-frequency jump estimators (bipower/HARQ) need intraday returns.

Features per day (all causal, expanding/rolling up to t):
  * ``vol_z``   : (v_t - rolling_mean) / rolling_std over the trailing window (how extreme today's market vol is)
  * ``regime``  : 1 if the trailing-mean market vol exceeds its expanding median, else 0 (high/low-vol regime)
  * ``dsc``     : days since the regime last flipped (0 on a flip day)
No heavy dependency (no PELT/HMM): a rolling threshold is reproducible and, per the research, the PELT
"accurate detection" claim did not survive verification, so we keep the signal simple and auditable.
"""
from __future__ import annotations

import numpy as np


def _rolling_mean_std(v, window):
    """Causal trailing mean/std with min_periods=1 (element t uses v[max(0,t-window+1)..t])."""
    n = len(v)
    mean = np.empty(n); std = np.empty(n)
    for t in range(n):
        w = v[max(0, t - window + 1):t + 1]
        mean[t] = w.mean()
        std[t] = w.std() if len(w) > 1 else 0.0
    return mean, std


def compute_regime_features(market_vol, window):
    """Return an [T, 3] array [vol_z, regime, days_since_change] from a 1-D daily market-vol series.

    All columns are causal. ``regime`` compares the trailing-mean vol to its EXPANDING median (only past
    values), so day t's label never uses the future. NaNs in the input are treated as the running mean."""
    v = np.asarray(market_vol, dtype=float)
    n = len(v)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    v = np.where(np.isfinite(v), v, np.nan)
    # fill NaN with the causal running mean so the rolling stats stay defined
    filled = v.copy()
    run = 0.0; cnt = 0
    for t in range(n):
        if np.isfinite(filled[t]):
            run += filled[t]; cnt += 1
        else:
            filled[t] = run / cnt if cnt else 0.0
    mean, std = _rolling_mean_std(filled, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_z = np.where(std > 0, (filled - mean) / std, 0.0)
    regime = np.zeros(n); dsc = np.zeros(n)
    running = []
    prev = 0
    for t in range(n):
        running.append(mean[t])
        med = float(np.median(running))            # expanding median of the trailing-mean vol (causal)
        r = 1 if mean[t] > med else 0
        regime[t] = r
        dsc[t] = 0 if (t > 0 and r != prev) else (dsc[t - 1] + 1 if t > 0 else 0)
        prev = r
    return np.column_stack([vol_z, regime, dsc]).astype(np.float32)
