"""Walk-forward global-minimum-variance (GMV) portfolio evaluation of covariance forecasts (Engle-Ledoit-Wolf
2019): the best covariance forecast is the one whose GMV portfolio has the lowest OUT-OF-SAMPLE realized vol.

Leakage-safe: at each rebalance date t, Sigma_hat_t is estimated from the WINDOW rows strictly before t; the
resulting weights are held over the next h days and scored on returns strictly after t. No intraday data
needed. Significance via a DM test on the squared portfolio returns of two models.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_STATS = Path(__file__).resolve().parents[2] / "2026-08-21_har_anchored_residual" / "code"
if str(_STATS) not in sys.path:
    sys.path.insert(0, str(_STATS))  # pragma: no cover - path bootstrap (conftest pre-seeds under pytest)
import stats as ST  # noqa: E402

import cov_config as C  # noqa: E402


def gmv_weights(sigma: np.ndarray, long_only: bool = False) -> np.ndarray:
    """Global-minimum-variance weights w = Sigma^-1 1 / (1^T Sigma^-1 1). If `long_only`, clip negatives and
    renormalize (a simple non-negativity projection; sum(w)=1)."""
    n = sigma.shape[0]
    ones = np.ones(n)
    x = np.linalg.solve(sigma, ones)
    w = x / (ones @ x)
    if long_only:
        w = np.clip(w, 0.0, None)
        s = w.sum()
        w = w / s if s > 0 else np.full(n, 1.0 / n)
    return w


def _rebalance_dates(dates: np.ndarray, eval_start: pd.Timestamp, h: int) -> list[int]:
    """Row indices (into `dates`) of rebalance points: every h trading days from the first date >= eval_start,
    leaving at least h future rows to score the last hold."""
    idx = np.searchsorted(dates, np.datetime64(eval_start))
    return [i for i in range(idx, len(dates) - h, h)]


def walk_forward(R: pd.DataFrame, estimator, h: int, window: int | None = None,
                 eval_start: str | None = None, gbm_diag: pd.DataFrame | None = None) -> dict:
    """Roll over rebalance dates: estimate Sigma_hat from the trailing `window` rows (< t), form GMV weights,
    hold h days, collect OOS portfolio returns (long-short + long-only) and turnover. Returns pooled series.
    `gbm_diag` (optional, wide [date x ticker] sigma) triggers the DCC combination `D_gbm R D_gbm`: at each
    rebalance the GBM per-stock vols replace the estimator's diagonal (missing entries fall back to it)."""
    from estimators import with_gbm_diagonal
    window = window or C.WINDOW
    eval_start = pd.Timestamp(eval_start or C.EVAL_START)
    Rv = R.to_numpy(float)
    dates = R.index.to_numpy()
    g_aligned = gbm_diag.reindex(index=R.index, columns=R.columns).ffill().to_numpy() \
        if gbm_diag is not None else None
    rp_ls, rp_lo, rp_dates = [], [], []
    prev_w = None
    turnover = []
    for t in _rebalance_dates(dates, eval_start, h):
        if t < window:
            continue
        train = Rv[t - window:t]
        sigma = estimator(train)
        if g_aligned is not None:
            g = g_aligned[t].copy()
            own = np.sqrt(np.clip(np.diag(sigma), 0.0, None))
            nan = ~np.isfinite(g)
            g[nan] = own[nan]                       # fall back to estimator diagonal where GBM missing
            sigma = with_gbm_diagonal(sigma, g)
        w = gmv_weights(sigma, long_only=False)
        w_lo = gmv_weights(sigma, long_only=True)
        fut = Rv[t:t + h]                       # returns strictly AFTER t (rows t .. t+h-1 are future)
        rp_ls.append(fut @ w)
        rp_lo.append(fut @ w_lo)
        rp_dates.append(dates[t:t + h])
        if prev_w is not None:
            turnover.append(float(np.abs(w - prev_w).sum()))
        prev_w = w
    if not rp_ls:
        raise ValueError("no rebalance dates: eval_start beyond the panel, or window+h exceeds the panel")
    return {
        "rp_ls": np.concatenate(rp_ls), "rp_lo": np.concatenate(rp_lo),
        "dates": np.concatenate(rp_dates), "turnover": float(np.mean(turnover)) if turnover else 0.0,
    }


def ann_vol(rp: np.ndarray) -> float:
    """Annualized realized volatility of a pooled daily portfolio-return series."""
    return float(np.std(rp, ddof=1) * np.sqrt(C.ANNUALIZE))


def dm_vol(rp_a: np.ndarray, rp_b: np.ndarray, dates: np.ndarray, h: int) -> float:
    """DM p-value on squared portfolio returns (ELW volatility-timing comparison). Lower r_p^2 = better."""
    try:
        return float(ST.date_clustered_dm(rp_a ** 2, rp_b ** 2, dates, h)["p_value"])
    except (ValueError, ZeroDivisionError):
        return 1.0


def _spike_mask(dates: np.ndarray) -> np.ndarray:
    """Boolean mask of portfolio dates inside any configured regime-spike window."""
    d = pd.to_datetime(dates)
    m = np.zeros(len(d), bool)
    for lo, hi in C.SPIKE_WINDOWS:
        m |= (d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))
    return m
