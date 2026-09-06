"""Leakage-safe chronological out-of-fold (OOF) HAR-X forecasts for the residual-training target.

The residual learner must NOT be trained against ordinary in-sample HAR-X fitted values (that would let
it memorise the OLS residuals). Instead ``oof_harx`` produces expanding-window HAR-X forecasts over the
fold's TRAIN anchors: after a warm-up prefix, the remaining train anchors are split into contiguous
chronological blocks and each block is predicted by an OLS fitted **strictly on earlier train anchors**.

All arrays are indexed ``[anchor, node]``; anchors are already in chronological order (``wf_folds``).
"""
from __future__ import annotations

import numpy as np

import xgb_config as C


def _ols_fit(har5_rows, y_rows):
    """Least-squares HAR-X coefficients (intercept + 5 features) on flat valid rows."""
    a = np.column_stack([np.ones(len(har5_rows)), har5_rows])
    return np.linalg.lstsq(a, y_rows, rcond=None)[0]


def _ols_predict(coef, har5, floor, nfloor):
    """Floored HAR-X forecast for a ``[m,5]`` design; ``nfloor`` is a scalar or ``[m]`` per-row floor."""
    a = np.column_stack([np.ones(len(har5)), har5])
    return np.maximum(np.maximum(a @ coef, floor), nfloor)


def oof_harx(har5, y, mask, floor, nfloor):
    """Expanding-window OOF HAR-X forecasts over TRAIN anchors -> ``oof [A,N]`` (NaN where not predicted).

    ``har5`` is ``[A,N,5]``, ``y`` ``[A,N]``, ``mask`` ``[A,N]`` bool (valid train cells), ``nfloor`` a
    per-node ``[N]`` positivity floor. Warm-up rows (the first ``OOF_WARMUP_FRAC`` of anchors) get no OOF
    prediction; each later block is predicted by an OLS fitted only on anchors strictly before it.
    """
    a_count, n, _ = har5.shape
    oof = np.full((a_count, n), np.nan)
    warmup = max(1, int(a_count * C.OOF_WARMUP_FRAC))
    if a_count - warmup < C.OOF_SPLITS:
        return oof                                          # too few train anchors for a valid OOF -> leave NaN
    bounds = np.linspace(warmup, a_count, C.OOF_SPLITS + 1).astype(int)
    for b in range(C.OOF_SPLITS):
        s, e = bounds[b], bounds[b + 1]
        if e <= s:               # pragma: no cover - degenerate linspace bound (guarded by warmup>=1, a-warmup>=SPLITS)
            continue
        fit_m = mask[:s]                                    # anchors strictly before this block
        if fit_m.sum() < 6:                                 # need >= (#coef) rows for a determined OLS
            continue
        coef = _ols_fit(har5[:s][fit_m], y[:s][fit_m])
        blk_m = mask[s:e]
        if not blk_m.any():
            continue
        # per-cell node index for the block's valid cells -> per-row nfloor
        rows_node = np.broadcast_to(np.arange(n), (e - s, n))[blk_m]
        pred = _ols_predict(coef, har5[s:e][blk_m], floor, nfloor[rows_node])
        block = np.full((e - s, n), np.nan)
        block[blk_m] = pred
        oof[s:e] = block
    return oof


def residual_target(y, harx_oof):
    """Log residual-ratio target ``z = log((y+eps)/(harx_oof+eps))`` (NaN where ``harx_oof`` is NaN)."""
    eps = C.RESIDUAL_EPS
    return np.log((np.maximum(y, 0.0) + eps) / (harx_oof + eps))


def reconstruct(harx_forecast, zhat, alpha, clip):
    """Final forecast ``yhat = HARX_forecast * exp(alpha * clip(zhat, -c, c))`` (elementwise)."""
    return harx_forecast * np.exp(alpha * np.clip(zhat, -clip, clip))
