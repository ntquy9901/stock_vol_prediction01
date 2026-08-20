"""Self-contained evaluation metrics for the SOICT LSTM-GAT submission.

Point-forecast metrics (MSE/RMSE/MAE/R2), the QLIKE volatility loss with a shared
positivity floor, and a Diebold-Mariano test with the Harvey-Leybourne-Newbold (1997)
small-sample correction and a HAC (Newey-West / Bartlett) long-run variance truncated
at ``h - 1`` lags.

The QLIKE formula mirrors ``baselines/2026-08-15_volatility/code/dm_report.py`` (``_qlike``,
``_EPSILON = 1e-8``): clamp both target and prediction to ``>= floor`` on a single shared
floor, ratio ``r = y / p``, loss ``r - log(r) - 1``. The Diebold-Mariano logic is copied
from ``baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/diebold_mariano.py``.

Dependencies: numpy for every metric; scipy.stats only for the Student-t p-value of the DM
test (matching the referenced implementation exactly -- there is no numpy-only Student-t CDF).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

_EPSILON = 1e-8


def mse(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((y - p) ** 2))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y, p)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute error."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(y - p)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    """Coefficient of determination (1 - SS_res / SS_tot)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot)


def per_obs_se(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-observation squared error ``(y - p) ** 2``."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return (y - p) ** 2


def per_obs_qlike(y: np.ndarray, p: np.ndarray, floor: float = _EPSILON) -> np.ndarray:
    """Per-observation QLIKE with a single shared positivity floor.

    Both target and prediction are clamped to ``>= floor`` (identical basis), then
    ``r = y / p`` and loss ``= r - log(r) - 1`` (== 0 when the forecast is exact).
    """
    y = np.maximum(np.asarray(y, dtype=float), floor)
    p = np.maximum(np.asarray(p, dtype=float), floor)
    ratio = y / p
    return ratio - np.log(ratio) - 1.0


def qlike(y: np.ndarray, p: np.ndarray, floor: float = _EPSILON) -> float:
    """Mean QLIKE loss over all observations (shared positivity floor)."""
    return float(np.mean(per_obs_qlike(y, p, floor=floor)))


@dataclass(frozen=True)
class DMResult:
    """Diebold-Mariano outcome. ``dm_hln`` carries the HLN small-sample correction and
    is referred to Student-t(n-1) for ``p_value``. Negative ``dm_hln`` / ``mean_diff``
    favours series A (smaller loss). ``n`` is the number of observations."""

    dm_hln: float
    p_value: float
    mean_diff: float
    n: int


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> DMResult:
    """Two-sided Diebold-Mariano test on per-observation losses.

    Loss differential ``d = loss_a - loss_b``; a negative statistic means series A carries
    the smaller loss (A more accurate). The long-run variance uses a Newey-West Bartlett
    HAC estimator truncated at ``h - 1`` lags; the HLN (1997) small-sample correction is
    applied and the statistic referred to Student-t with ``n - 1`` degrees of freedom.

    Copied from baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/diebold_mariano.py.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("loss_a and loss_b must have the same shape")
    if a.ndim != 1:
        raise ValueError("losses must be one-dimensional per-observation series")
    if h < 1:
        raise ValueError("horizon h must be >= 1")
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
        weight = 1.0 - lag / (max_lag + 1)
        long_run += 2.0 * weight * gamma

    if long_run <= 0.0:
        raise ValueError("non-positive long-run variance; DM statistic undefined")

    dm_stat = mean_diff / np.sqrt(long_run / n)
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = float(dm_stat * hln_factor)
    p_value = float(2.0 * stats.t.cdf(-abs(dm_hln), df=n - 1))

    return DMResult(dm_hln=dm_hln, p_value=p_value, mean_diff=mean_diff, n=n)
