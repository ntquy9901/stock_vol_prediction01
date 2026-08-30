"""NEW ETL cleaning functions for the dirty-data classes. Each is pure: it takes a raw OHLCV frame and
returns ``(cleaned_df, info)`` WITHOUT mutating the input. Independently tested against the intended
behaviour (not by reusing the implementation).

Estimator impact (see the consolidated spec):
  * widen_range / clip_oc / swap_or_drop_high_low / reconstruct_nonpositive / backadjust_splits change
    high/low and therefore CAN move the Parkinson target.
  * clip_oc / widen_range specifically fix the open/close-outside class, which matters for the O/C-using
    estimators (GK/RS/YZ); Parkinson (H/L only) is immune to O/C values themselves.
  * flag_* KEEP the rows (a liquidity screen / vol floor is the right handling, not deletion).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

OHLC_RTOL = 1e-5
SPLIT_THRESH = 0.50
_OHLC = ["open", "high", "low", "close"]


def _arrs(df: pd.DataFrame):
    return {k: pd.to_numeric(df[k], errors="coerce").to_numpy(float) for k in _OHLC}


def widen_range(df: pd.DataFrame) -> tuple:
    """RECOMMENDED fix for open/close-outside: treat open/close as real trades the recorded high/low missed,
    so set high=max(high,open,close), low=min(low,open,close). Makes OHLC internally consistent and preserves
    the trade information. n_widened = rows whose high or low actually moved."""
    out = df.copy()
    a = _arrs(df)
    new_h = np.fmax(a["high"], np.fmax(a["open"], a["close"]))
    new_l = np.fmin(a["low"], np.fmin(a["open"], a["close"]))
    moved = (~np.isclose(new_h, a["high"], equal_nan=True)) | (~np.isclose(new_l, a["low"], equal_nan=True))
    out["high"] = new_h
    out["low"] = new_l
    return out, {"n_widened": int(np.sum(moved & np.isfinite(new_h) & np.isfinite(new_l)))}


def clip_oc(df: pd.DataFrame) -> tuple:
    """ALTERNATIVE fix for open/close-outside: clip open/close into [low, high] (discards the out-of-range
    trade info). n_clipped = rows whose open or close moved."""
    out = df.copy()
    a = _arrs(df)
    new_o = np.clip(a["open"], a["low"], a["high"])
    new_c = np.clip(a["close"], a["low"], a["high"])
    moved = (~np.isclose(new_o, a["open"], equal_nan=True)) | (~np.isclose(new_c, a["close"], equal_nan=True))
    out["open"] = new_o
    out["close"] = new_c
    return out, {"n_clipped": int(np.sum(moved & np.isfinite(new_o) & np.isfinite(new_c)))}


def swap_or_drop_high_low(df: pd.DataFrame) -> tuple:
    """For high<low rows: if swapping high<->low yields a valid bar (open/close inside the swapped range,
    within tolerance) it is a transposition -> swap and keep; otherwise the bar is unrecoverable -> drop."""
    a = _arrs(df)
    o, h, lo, c = a["open"], a["high"], a["low"], a["close"]
    finite = np.isfinite([o, h, lo, c]).all(0)
    bad = finite & (h < lo)
    hi_oc, lo_oc = np.maximum(o, c), np.minimum(o, c)
    # swapped range would be [h, lo] (since h<lo); valid if o/c inside it within tolerance
    swap_ok = bad & (lo >= hi_oc * (1 - OHLC_RTOL)) & (h <= lo_oc * (1 + OHLC_RTOL))
    drop = bad & ~swap_ok
    out = df.copy()
    new_h, new_l = h.copy(), lo.copy()
    new_h[swap_ok] = lo[swap_ok]
    new_l[swap_ok] = h[swap_ok]
    out["high"] = new_h
    out["low"] = new_l
    out = out.loc[~drop].reset_index(drop=True)
    return out, {"n_swapped": int(np.sum(swap_ok)), "n_dropped": int(np.sum(drop))}


def reconstruct_nonpositive(df: pd.DataFrame) -> tuple:
    """For rows with any nonpositive OHLC: rebuild high=max / low=min over the POSITIVE OHLC values and clamp
    open/close into [low, high] (CLAUDE.md rule). If fewer than 2 positive OHLC values exist the bar cannot be
    reconstructed -> drop it."""
    a = _arrs(df)
    stack = np.vstack([a[k] for k in _OHLC])            # 4 x n
    finite = np.isfinite(stack).all(0)
    nonpos = finite & (stack <= 0).any(0)
    pos = np.where(stack > 0, stack, np.nan)
    n_pos = np.sum(np.isfinite(pos), axis=0)
    # nanmax/nanmin over an all-nonpositive bar (n_pos==0) is an unused NaN (gated out by fixable=n_pos>=2);
    # suppress the "All-NaN slice" RuntimeWarning it would emit rather than let it pollute the log.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hi = np.nanmax(pos, axis=0)
        lo = np.nanmin(pos, axis=0)
    fixable = nonpos & (n_pos >= 2)
    drop = nonpos & (n_pos < 2)
    out = df.copy()
    o, h, l_, c = a["open"], a["high"], a["low"], a["close"]
    h = np.where(fixable, hi, h)
    l_ = np.where(fixable, lo, l_)
    o = np.where(fixable, np.clip(np.where(o > 0, o, hi), lo, hi), o)
    c = np.where(fixable, np.clip(np.where(c > 0, c, hi), lo, hi), c)
    out["open"], out["high"], out["low"], out["close"] = o, h, l_, c
    out = out.loc[~drop].reset_index(drop=True)
    return out, {"n_reconstructed": int(np.sum(fixable)), "n_dropped": int(np.sum(drop))}


def backadjust_splits(df: pd.DataFrame, thresh: float = SPLIT_THRESH) -> tuple:
    """Back-adjust candidate unadjusted splits: on each day with |simple return|>thresh, multiply ALL prior
    rows' OHLC by factor = close_t / close_{t-1} so the price level is continuous (the jump is removed).
    Volume is left unchanged. Processes the most recent jump last so cumulative factors compose correctly."""
    out = df.copy()
    o = out["open"].to_numpy(float).copy()
    h = out["high"].to_numpy(float).copy()
    lo = out["low"].to_numpy(float).copy()
    c = out["close"].to_numpy(float).copy()
    prev = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(prev > 0, c / prev - 1.0, np.nan)
    jump_idx = [i for i in range(len(c)) if np.isfinite(r[i]) and abs(r[i]) > thresh]
    n_adj = 0
    for i in jump_idx:
        if not (prev[i] > 0 and c[i] > 0):
            continue
        factor = c[i] / prev[i]                        # >1 for an up-jump (e.g. 2:1 split doubling)
        for arr in (o, h, lo, c):
            arr[:i] *= factor
        prev = np.concatenate([[np.nan], c[:-1]])      # recompute after adjusting earlier levels
        n_adj += 1
    out["open"], out["high"], out["low"], out["close"] = o, h, lo, c
    return out, {"n_adjusted": n_adj}


def _leading_backfill_len(df: pd.DataFrame) -> int:
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)  # noqa: F841 - symmetry with detectors
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    # No silent degradation (CLAUDE.md): absent volume -> NaN (neutral), not zero-volume, so a flat-close
    # start with a real H-L range is NOT mistaken for pre-listing backfill and cut.
    v = pd.to_numeric(df["volume"], errors="coerce").to_numpy(float) if "volume" in df.columns \
        else np.full(len(df), np.nan)
    n = len(c)
    if n == 0:
        return 0
    c0, k = c[0], 0
    while k < n:
        constant = np.isfinite(c[k]) and c[k] == c0
        nontrading = (np.isfinite(v[k]) and v[k] == 0.0) or (np.isfinite(h[k]) and np.isfinite(lo[k])
                                                             and h[k] == lo[k])
        if constant and nontrading:
            k += 1
        else:
            break
    return 0 if k >= n else k


def cut_to_listing(df: pd.DataFrame) -> tuple:
    """Drop the leading pre-listing / backfill run (constant close + non-trading) so the series starts at the
    true first-trade date."""
    k = _leading_backfill_len(df)
    out = df.iloc[k:].reset_index(drop=True)
    return out, {"n_cut": int(k)}


def drop_naninf(df: pd.DataFrame) -> tuple:
    """Drop rows whose open/high/low/close/volume contains a non-finite value (must never reach the model)."""
    cols = _OHLC + (["volume"] if "volume" in df.columns else [])
    vals = np.vstack([pd.to_numeric(df[k], errors="coerce").to_numpy(float) for k in cols])
    finite = np.isfinite(vals).all(0)
    out = df.loc[finite].reset_index(drop=True)
    return out, {"n_dropped": int(np.sum(~finite))}


def flag_zero_range(df: pd.DataFrame) -> tuple:
    """KEEP all rows; add a boolean ``zero_range_flag`` marking finite-positive high==low (limit/no-trade)."""
    out = df.copy()
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    flag = np.isfinite(h) & np.isfinite(lo) & (h == lo) & (h > 0)
    out["zero_range_flag"] = flag
    return out, {"n_flagged": int(np.sum(flag))}


def flag_zero_volume(df: pd.DataFrame) -> tuple:
    """KEEP all rows; add a boolean ``zero_volume_flag`` marking finite volume == 0 (illiquidity)."""
    out = df.copy()
    v = pd.to_numeric(df["volume"], errors="coerce").to_numpy(float) if "volume" in df.columns \
        else np.full(len(df), np.nan)
    flag = np.isfinite(v) & (v == 0.0)
    out["zero_volume_flag"] = flag
    return out, {"n_flagged": int(np.sum(flag))}
