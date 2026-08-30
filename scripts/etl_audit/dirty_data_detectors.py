"""Per-(ticker, date) dirty-data LOCATE detectors on a single raw OHLCV frame.

The existing ``vnmarkets_eda`` / ``sp500_eda`` detectors return COUNTS only; the audit needs the exact
offending dates (for the per-stock drill-down) and magnitudes. These functions are pure (no torch, no IO)
and use the SAME relative tolerance as the raw-data quality gate so float32 storage noise is not flagged.

Estimator-impact note (used by the ETL spec): Parkinson = ln(H/L)^2/(4 ln2) uses only high/low, so
open/close-outside and stale-close are COSMETIC for the delivered Parkinson target but REAL for the
open/close-using estimators (Garman-Klass, Rogers-Satchell, Yang-Zhang).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLC_RTOL = 1e-5                    # matches tests/test_raw_prices_quality.py + vnmarkets_eda
SPLIT_THRESH = 0.50                 # |1-day simple return| > 50% -> candidate unadjusted split/dividend
STALE_MIN_RUN = 5                   # >= N identical consecutive closes -> stale run
_EXAMPLE_CAP = 25                   # cap example lists stored per class (drill-down only needs the worst)


def _cols(df: pd.DataFrame):
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)
    return o, h, lo, c


def _dates(df: pd.DataFrame) -> np.ndarray:
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d").to_numpy()
    return np.arange(len(df)).astype(str)


def high_lt_low(df: pd.DataFrame) -> list:
    """Dates where high < low (impossible geometry)."""
    o, h, lo, c = _cols(df)
    finite = np.isfinite([o, h, lo, c]).all(0)
    mask = finite & (h < lo)
    return _dates(df)[mask].tolist()


def open_close_outside(df: pd.DataFrame) -> list:
    """(date, rel_violation) where open/close fall outside [low, high] beyond the float tolerance.
    rel_violation = the larger of the high-side / low-side relative overshoot (magnitude for reporting)."""
    o, h, lo, c = _cols(df)
    finite = np.isfinite([o, h, lo, c]).all(0)
    hi_oc = np.maximum(o, c)
    lo_oc = np.minimum(o, c)
    over_hi = finite & (h < hi_oc * (1 - OHLC_RTOL))
    over_lo = finite & (lo > lo_oc * (1 + OHLC_RTOL))
    mask = over_hi | over_lo
    dates = _dates(df)
    out = []
    for i in np.where(mask)[0]:
        hi_v = (hi_oc[i] - h[i]) / h[i] if (over_hi[i] and h[i] > 0) else 0.0
        lo_v = (lo[i] - lo_oc[i]) / lo[i] if (over_lo[i] and lo[i] > 0) else 0.0
        out.append((dates[i], float(max(hi_v, lo_v))))
    return out


def nonpositive(df: pd.DataFrame) -> list:
    """Dates where any finite OHLC value is <= 0."""
    o, h, lo, c = _cols(df)
    finite = np.isfinite([o, h, lo, c]).all(0)
    mask = finite & ((o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0))
    return _dates(df)[mask].tolist()


def zero_range(df: pd.DataFrame) -> list:
    """Dates with a finite POSITIVE high == low (limit-lock / no-trade -> Parkinson == 0)."""
    o, h, lo, c = _cols(df)
    finite = np.isfinite([o, h, lo, c]).all(0)
    mask = finite & (h == lo) & (h > 0)
    return _dates(df)[mask].tolist()


def _simple_returns(c: np.ndarray) -> np.ndarray:
    prev = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(prev > 0, c / prev - 1.0, np.nan)


def split_jumps(df: pd.DataFrame, thresh: float = SPLIT_THRESH) -> list:
    """(date, simple_return) where |1-day simple return| > thresh, sorted by |return| descending."""
    _o, _h, _lo, c = _cols(df)
    r = _simple_returns(c)
    dates = _dates(df)
    idx = np.where(np.isfinite(r) & (np.abs(r) > thresh))[0]
    out = [(dates[i], float(r[i])) for i in idx]
    out.sort(key=lambda t: abs(t[1]), reverse=True)
    return out


def stale_runs(df: pd.DataFrame, min_run: int = STALE_MIN_RUN) -> list:
    """(start_date, end_date, length) maximal runs of >= min_run identical finite-positive closes."""
    _o, _h, _lo, c = _cols(df)
    dates = _dates(df)
    runs = []
    i, n = 0, len(c)
    while i < n:
        j = i
        while j + 1 < n and np.isfinite(c[j]) and c[j] > 0 and c[j + 1] == c[i]:
            j += 1
        length = j - i + 1
        if length >= min_run and np.isfinite(c[i]) and c[i] > 0:
            runs.append((dates[i], dates[j], int(length)))
        i = j + 1
    return runs


def naninf(df: pd.DataFrame) -> list:
    """Dates whose open/high/low/close/volume contains a non-finite (NaN/inf) value."""
    o, h, lo, c = _cols(df)
    v = pd.to_numeric(df["volume"], errors="coerce").to_numpy(float) if "volume" in df.columns \
        else np.zeros(len(df))
    mask = ~np.isfinite([o, h, lo, c, v]).all(0)
    return _dates(df)[mask].tolist()


def zero_volume(df: pd.DataFrame) -> list:
    """Dates with a finite volume == 0 (illiquidity flag)."""
    if "volume" not in df.columns:
        return []
    v = pd.to_numeric(df["volume"], errors="coerce").to_numpy(float)
    mask = np.isfinite(v) & (v == 0.0)
    return _dates(df)[mask].tolist()


def leading_backfill(df: pd.DataFrame) -> dict:
    """Detect a leading pre-listing / backfill run: the initial block of rows whose close is constant AND
    which look non-trading (zero volume OR zero range high==low). Returns n_leading + the first genuine
    trade date. n_leading==0 when the series trades from the first row."""
    _o, h, lo, c = _cols(df)
    # No silent degradation (CLAUDE.md): a MISSING volume column must NOT be read as zero-volume (that would
    # flag every flat-close start as backfill and cut real rows). Absent volume -> NaN -> contributes nothing;
    # the non-trading test then rests on zero-range (high==low) only.
    v = pd.to_numeric(df["volume"], errors="coerce").to_numpy(float) if "volume" in df.columns \
        else np.full(len(df), np.nan)
    dates = _dates(df)
    n = len(c)
    if n == 0:
        return {"n_leading": 0, "first_trade_date": None}
    c0 = c[0]
    k = 0
    while k < n:
        constant = np.isfinite(c[k]) and c[k] == c0
        nontrading = (np.isfinite(v[k]) and v[k] == 0.0) or (np.isfinite(h[k]) and np.isfinite(lo[k])
                                                             and h[k] == lo[k])
        if constant and nontrading:
            k += 1
        else:
            break
    if k >= n:              # entire series is a flat non-trading block -> not a "listing", flag nothing
        return {"n_leading": 0, "first_trade_date": None}
    if k == 0:
        return {"n_leading": 0, "first_trade_date": str(dates[0])}
    return {"n_leading": int(k), "first_trade_date": str(dates[k])}


def detect_all(df: pd.DataFrame) -> dict:
    """Run every detector; return per-class counts + capped example lists."""
    hl = high_lt_low(df)
    oc = open_close_outside(df)
    npv = nonpositive(df)
    zr = zero_range(df)
    sj = split_jumps(df)
    st = stale_runs(df)
    ni = naninf(df)
    zv = zero_volume(df)
    lb = leading_backfill(df)
    counts = {
        "high_lt_low": len(hl), "open_close_outside": len(oc), "nonpositive": len(npv),
        # stale_runs is counted in stale DAYS (sum of run lengths) so its "% of rows" is comparable to the
        # other day-based classes (a run count would be a different unit; code review 2026-08-30).
        "zero_range": len(zr), "split_jumps": len(sj), "stale_runs": sum(length for _s, _e, length in st),
        "naninf": len(ni), "zero_volume": len(zv), "leading_backfill": lb["n_leading"],
    }
    oc_mag = float(np.median([m for _d, m in oc])) if oc else 0.0
    oc_worst = sorted(oc, key=lambda t: t[1], reverse=True)   # drill-down highlights the LARGEST violations
    examples = {
        "high_lt_low": hl[:_EXAMPLE_CAP], "open_close_outside": oc_worst[:_EXAMPLE_CAP],
        "nonpositive": npv[:_EXAMPLE_CAP], "zero_range": zr[:_EXAMPLE_CAP],
        "split_jumps": sj[:_EXAMPLE_CAP], "stale_runs": st[:_EXAMPLE_CAP],
        "naninf": ni[:_EXAMPLE_CAP], "zero_volume": zv[:_EXAMPLE_CAP],
        "leading_backfill": lb,
    }
    return {"counts": counts, "examples": examples, "oc_median_rel_violation": oc_mag}


def per_ticker_summary(ticker: str, df: pd.DataFrame) -> dict:
    """One flat row for the sortable per-ticker table: counts of each class + coverage + illiquidity rates."""
    res = detect_all(df)
    counts = res["counts"]
    _o, _h, _lo, c = _cols(df)
    n = len(df)
    zv_frac = counts["zero_volume"] / n if n else float("nan")
    zr_frac = counts["zero_range"] / n if n else float("nan")
    dates = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else None
    row = {
        "ticker": ticker, "rows": n,
        "first": str(dates.min().date()) if dates is not None and dates.notna().any() else "",
        "last": str(dates.max().date()) if dates is not None and dates.notna().any() else "",
        "zero_range_frac": float(zr_frac), "zero_volume_frac": float(zv_frac),
        "oc_median_rel_violation": res["oc_median_rel_violation"],
    }
    row.update(counts)
    return row
