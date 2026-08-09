"""Diebold-Mariano test for equal predictive accuracy.

Tests whether two forecasts have equal expected loss on the SAME evaluation set.
The loss differential is ``d_t = loss_a - loss_b`` (per observation), so a negative
statistic means model A carries the smaller loss (A is more accurate).

The long-run variance of the mean differential is estimated with a HAC estimator
truncated at ``h - 1`` lags -- the Newey-West Bartlett kernel by default, or the
classic Diebold-Mariano (1995) rectangular (un-weighted) truncated sum. The
Harvey-Leybourne-Newbold (1997) small-sample correction is applied and the statistic
is referred to a Student-t distribution with ``n - 1`` degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DieboldMarianoResult:
    """Outcome of a Diebold-Mariano test.

    ``dm_stat`` is the raw statistic; ``dm_hln`` applies the HLN small-sample
    correction and is the value referred to the Student-t(n-1) distribution for
    ``p_value``. Negative statistics favour model A (smaller loss).
    """

    dm_stat: float
    dm_hln: float
    p_value: float
    mean_diff: float
    long_run_variance: float
    n: int
    h: int
    kernel: str


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    h: int = 1,
    kernel: str = "bartlett",
) -> DieboldMarianoResult:
    """Two-sided Diebold-Mariano test on per-observation losses.

    Args:
        loss_a: Per-observation loss of forecast A (e.g. G1).
        loss_b: Per-observation loss of forecast B (e.g. G0), same order/length.
        h: Forecast horizon; the HAC truncation lag is ``h - 1``.
        kernel: ``"bartlett"`` (Newey-West weights) or ``"rectangular"`` (classic DM).
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("loss_a and loss_b must have the same shape")
    if a.ndim != 1:
        raise ValueError("losses must be one-dimensional per-observation series")
    if h < 1:
        raise ValueError("horizon h must be >= 1")
    if kernel not in ("bartlett", "rectangular"):
        raise ValueError("kernel must be 'bartlett' or 'rectangular'")
    n = a.size
    if n < 2:
        raise ValueError("Diebold-Mariano requires at least two observations")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("losses must be finite")

    d = a - b
    mean_diff = float(d.mean())
    dev = d - mean_diff
    max_lag = h - 1

    gamma0 = float(np.dot(dev, dev) / n)
    long_run = gamma0
    for lag in range(1, max_lag + 1):
        gamma = float(np.dot(dev[lag:], dev[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1) if kernel == "bartlett" else 1.0
        long_run += 2.0 * weight * gamma

    if long_run <= 0.0:
        raise ValueError("non-positive long-run variance; DM statistic undefined")

    dm_stat = mean_diff / np.sqrt(long_run / n)
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = float(dm_stat * hln_factor)
    p_value = float(2.0 * stats.t.cdf(-abs(dm_hln), df=n - 1))

    return DieboldMarianoResult(
        dm_stat=float(dm_stat),
        dm_hln=dm_hln,
        p_value=p_value,
        mean_diff=mean_diff,
        long_run_variance=long_run,
        n=n,
        h=h,
        kernel=kernel,
    )
