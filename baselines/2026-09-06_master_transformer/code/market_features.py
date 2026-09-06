"""Causal market features for the MASTER market-guided gate (no look-ahead).

MASTER's Gate reads market-level features at the last timestep to modulate each stock's features. We
build M=4 causal market features per trading date from the enriched panel and standardize them with a
scaler fit on TRAIN dates only (val/test never enter the scaler):

  1. ``m_mean``   cross-sectional mean of Parkinson variance over stocks   (market vol level)
  2. ``m_disp``   cross-sectional dispersion (std) over stocks             (market dispersion)
  3. ``m_vshock`` cross-sectional mean of the volume z-shock               (market volume shock)
  4. ``m_volz``   causal rolling z of ``m_mean`` over a trailing window    (how extreme today's vol is)

All columns use only information up to each day t. ``m_volz`` uses a trailing window ending at t
(``[max(0, t-window+1) .. t]``), so day t never sees the future.
"""
from __future__ import annotations

import warnings

import numpy as np

N_MARKET_FEAT = 4


def _cross_sectional(pk: np.ndarray, vshock: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-date cross-sectional aggregates over stocks, ignoring off-date NaNs.

    ``pk``/``vshock`` are ``[T, N]``. Rows with no valid stock (all-NaN) yield 0.0 (a neutral market
    state) instead of a NaN that would poison the scaler.
    """
    valid = np.isfinite(pk)
    any_valid = valid.any(axis=1)
    m_mean = np.zeros(pk.shape[0]); m_disp = np.zeros(pk.shape[0]); m_vshock = np.zeros(pk.shape[0])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN rows -> handled explicitly below
        mm = np.nanmean(np.where(valid, pk, np.nan), axis=1)
        md = np.nanstd(np.where(valid, pk, np.nan), axis=1)
        mv = np.nanmean(np.where(np.isfinite(vshock), vshock, np.nan), axis=1)
    m_mean[any_valid] = mm[any_valid]
    m_disp[any_valid] = md[any_valid]
    mv_valid = np.isfinite(vshock).any(axis=1)
    m_vshock[mv_valid] = mv[mv_valid]
    return m_mean, m_disp, m_vshock


def causal_rolling_z(series: np.ndarray, window: int) -> np.ndarray:
    """Causal trailing z-score: ``(v_t - mean) / std`` over ``[max(0, t-window+1) .. t]`` (min_periods=1).

    Element t uses ONLY values up to t (no look-ahead). std==0 (or a single observation) -> z=0.
    """
    v = np.asarray(series, dtype=float)
    n = len(v)
    z = np.zeros(n)
    for t in range(n):
        w = v[max(0, t - window + 1):t + 1]
        s = w.std()
        z[t] = (v[t] - w.mean()) / s if s > 0 else 0.0
    return z


def compute_market_raw(pk: np.ndarray, vshock: np.ndarray, window: int) -> np.ndarray:
    """Return the ``[T, 4]`` raw (unstandardized) causal market-feature matrix."""
    m_mean, m_disp, m_vshock = _cross_sectional(pk, vshock)
    m_volz = causal_rolling_z(m_mean, window)
    return np.column_stack([m_mean, m_disp, m_vshock, m_volz]).astype(np.float64)


def fit_market_scaler(mkt_raw: np.ndarray, train_end_row: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit a per-column StandardScaler on rows ``[0 : train_end_row + 1]`` (TRAIN dates only, no leakage).

    ``train_end_row`` is the date-index of the last TRAIN anchor; every train input window ends at or
    before it, so this range covers all train windows and no val/test date.
    """
    if not 0 <= train_end_row < len(mkt_raw):
        raise ValueError(f"train_end_row {train_end_row} out of range for T={len(mkt_raw)}")
    block = mkt_raw[:train_end_row + 1]
    mean = block.mean(axis=0)
    std = block.std(axis=0) + eps
    return mean, std


def standardize(mkt_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply a fitted scaler; NaN/inf -> 0.0 (neutral) so the gate never sees non-finite input."""
    out = (mkt_raw - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def pack_market(mkt_std: np.ndarray, anchor_rows: np.ndarray, lookback: int) -> np.ndarray:
    """Per-anchor per-timestep market window ``[A, lookback, 4]`` (broadcast across stocks by the caller).

    For anchor at date-index ``t`` the window is ``mkt_std[t - lookback + 1 : t + 1]`` -- all dates <= t,
    so it is causal. Requires ``t >= lookback - 1`` (guaranteed by the panel's anchor construction).
    """
    a = len(anchor_rows)
    out = np.zeros((a, lookback, mkt_std.shape[1]), dtype=np.float32)
    for i, t in enumerate(anchor_rows):
        if t - lookback + 1 < 0:
            raise ValueError(f"anchor row {t} < lookback-1 ({lookback - 1}); window would look before t=0")
        out[i] = mkt_std[t - lookback + 1:t + 1]
    return out
