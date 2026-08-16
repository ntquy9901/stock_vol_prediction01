"""Column-level verification for raw ``*_ohlcv.csv`` price files (any universe: VN30/VN100/HOSE/HNX).

Reports, per ticker, the health of every column and surfaces ``volume == 0`` days explicitly.

HARD checks (integrity that should never fail — a True means a real defect):
  * schema  : columns are exactly ``date,open,high,low,close,volume``
  * dates   : parseable (after normalizing tz-aware to plain date), strictly increasing, unique,
              weekday-only (no Sat/Sun)
  * ohlc    : open/high/low/close finite and > 0; ``high >= low``; ``high >= max(open,close)``;
              ``low <= min(open,close)``
  * volume  : finite and >= 0

DIAGNOSTICS (reported, not necessarily defects):
  * zero-volume days: count + fraction. A zero-volume day is usually a legitimate no-trade / halted
    session (common for less-liquid mid-caps) — Parkinson is still well-defined from H/L.
  * ``zero_vol_price_moved`` = rows with ``volume == 0`` BUT ``high != low`` — SUSPICIOUS: price moved
    with no reported volume, i.e. a feed glitch worth review.
  * ``zero_vol_flat`` = rows with ``volume == 0`` AND ``high == low`` — a flat no-trade day (benign) or
    pre-listing backfill.
  * ``leading_zero_vol_run`` = length of the leading ``volume == 0`` run (a backfill signature).

CLI:
  python -m src.data.verify_raw_prices data/raw/prices            # VN30
  python -m src.data.verify_raw_prices data/raw/prices/vn100      # VN100
Exit code 0 if no HARD defect in any ticker, else 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "volume"]
_OHLC = ["open", "high", "low", "close"]
_RTOL = 1e-5


def verify_frame(df: pd.DataFrame) -> dict:
    """Return a dict of per-ticker check results and diagnostics for one raw frame."""

    res: dict = {"rows": len(df)}
    res["schema_ok"] = list(df.columns) == COLUMNS
    if not res["schema_ok"]:
        res["hard_ok"] = False
        return res

    dates = pd.to_datetime(df["date"].astype(str).str.split(" ").str[0], errors="coerce")
    o, h, low, c = (df[x].astype(float) for x in _OHLC)
    vol = df["volume"].astype(float)

    res["first_date"] = str(df["date"].iloc[0]).split(" ")[0]
    res["last_date"] = str(df["date"].iloc[-1]).split(" ")[0]
    res["date_unparseable"] = int(dates.isna().sum())
    res["date_non_monotonic"] = int((dates.diff().dropna() <= pd.Timedelta(0)).sum())
    res["date_duplicated"] = int(dates.duplicated().sum())
    res["date_weekend"] = int((dates.dt.dayofweek >= 5).sum())

    ohlc = df[_OHLC].astype(float)
    res["nonpositive"] = int((ohlc <= 0).any(axis=1).sum())
    res["nan_inf"] = int((~np.isfinite(ohlc.to_numpy())).sum())
    res["high_lt_low"] = int((h < low * (1 - _RTOL)).sum())
    res["high_lt_max_oc"] = int((h < np.maximum(o, c) * (1 - _RTOL)).sum())
    res["low_gt_min_oc"] = int((low > np.minimum(o, c) * (1 + _RTOL)).sum())
    res["volume_negative"] = int((vol < 0).sum())
    res["volume_nan"] = int((~np.isfinite(vol.to_numpy())).sum())

    # volume == 0 diagnostics. "flat" uses a relative tolerance so float-precision equal highs/lows
    # (~1e-15, genuine no-trade days) count as flat, not as a spurious "price moved" glitch.
    zero = vol == 0
    flat = (h - low).abs() <= _RTOL * np.maximum(h.abs(), 1e-9)
    res["zero_vol"] = int(zero.sum())
    res["zero_vol_frac"] = round(float(zero.mean()), 4)
    res["zero_vol_price_moved"] = int((zero & ~flat).sum())   # suspicious: price moved, no volume
    res["zero_vol_flat"] = int((zero & flat).sum())           # benign no-trade / backfill
    run = 0
    for z in zero.to_numpy():
        if z:
            run += 1
        else:
            break
    res["leading_zero_vol_run"] = run

    res["hard_ok"] = all(res[k] == 0 for k in (
        "date_unparseable", "date_non_monotonic", "date_duplicated", "date_weekend",
        "nonpositive", "nan_inf", "high_lt_low", "high_lt_max_oc", "low_gt_min_oc",
        "volume_negative", "volume_nan",
    ))
    return res


def verify_dir(directory: Path) -> dict[str, dict]:
    """Verify every ``*_ohlcv.csv`` in ``directory``; returns ticker -> result dict."""

    out: dict[str, dict] = {}
    for p in sorted(Path(directory).glob("*_ohlcv.csv")):
        out[p.stem.replace("_ohlcv", "")] = verify_frame(pd.read_csv(p, dtype={"date": str}))
    return out


def main(directory: str) -> int:
    results = verify_dir(Path(directory))
    if not results:
        print(f"[verify] no *_ohlcv.csv under {directory}")
        return 1
    hard_fail = [t for t, r in results.items() if not r.get("hard_ok", False)]
    suspicious = sorted(
        ((t, r["zero_vol_price_moved"]) for t, r in results.items() if r.get("zero_vol_price_moved", 0)),
        key=lambda x: -x[1],
    )
    backfill = sorted(
        ((t, r["leading_zero_vol_run"]) for t, r in results.items() if r.get("leading_zero_vol_run", 0) > 20),
        key=lambda x: -x[1],
    )
    top_zero = sorted(results.items(), key=lambda kv: -kv[1].get("zero_vol_frac", 0))[:8]

    print(f"[verify] {directory}: {len(results)} tickers")
    print(f"  HARD defects: {len(hard_fail)} ticker(s)" + (f" -> {hard_fail}" if hard_fail else ""))
    print(f"  zero-volume WITH price move (SUSPICIOUS glitch): {len(suspicious)} ticker(s)"
          + (f" -> {suspicious[:8]}" if suspicious else ""))
    print(f"  leading zero-volume backfill run >20: {len(backfill)} ticker(s)"
          + (f" -> {backfill}" if backfill else ""))
    print("  highest zero-volume fraction: "
          + ", ".join(f"{t}={r['zero_vol_frac'] * 100:.1f}%" for t, r in top_zero))
    return 1 if hard_fail else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/prices"))
