"""Raw-review + ETL (raw -> Parkinson-variance processed) for HOSE/HNX crawls.

CPU/pandas only. Reuses the Parkinson-variance formula from
``src.common.parkinson_utils.calculate_parkinson_variance`` (does NOT
re-derive the math). Cleans raw OHLCV per the project CLAUDE.md rules
(fix non-positive/invalid OHLC via per-row max/min over positive prices,
trim leading backfill to the first real trading day, drop duplicate dates,
enforce weekday-only monotonic dates) rather than propagating bad rows.

Usage:
    python scripts/etl_hose_hnx.py scan  --raw data/raw/prices/hose_vnstock --out /tmp/hose_scan.json
    python scripts/etl_hose_hnx.py etl   --raw data/raw/prices/hose_vnstock --processed data/processed/hose --out /tmp/hose_etl.json

Writes only under the given --processed dir and the --out json path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.parkinson_utils import calculate_parkinson_variance  # noqa: E402

PRICE_COLS = ["open", "high", "low", "close"]
NUMERIC_COLS = ["open", "high", "low", "close", "volume"]
CLIP_CEILING = 0.1


def _ticker_of(path: Path) -> str:
    return path.name[: -len("_ohlcv.csv")]


def _discover(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("*_ohlcv.csv"))


def _load_raw(path: Path) -> pd.DataFrame:
    """Load one raw CSV; parse date (strip tz time part) and coerce numerics."""
    raw = pd.read_csv(path, dtype=str)
    df = pd.DataFrame()
    df["date_str"] = raw["date"].astype(str).str.split(" ").str[0]
    df["date"] = pd.to_datetime(df["date_str"], format="%Y-%m-%d", errors="coerce")
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(raw[col], errors="coerce")
    return df


def _leading_flat_run(high: np.ndarray, low: np.ndarray) -> int:
    """Length of the leading contiguous run of flat rows (high == low)."""
    flat = high == low
    if flat.size == 0 or not flat[0]:
        return 0
    if flat.all():
        return int(flat.size)
    return int(np.argmax(~flat))


# --------------------------------------------------------------------------- #
# Stage 1: raw review diagnostics (read-only)
# --------------------------------------------------------------------------- #

def scan_ticker(path: Path) -> dict:
    df = _load_raw(path)
    n = len(df)
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in PRICE_COLS)
    v = df["volume"].to_numpy(dtype=float)
    dates = df["date"]

    price_arr = df[PRICE_COLS].to_numpy(dtype=float)
    nan_any = np.isnan(df[NUMERIC_COLS].to_numpy(dtype=float)).any(axis=1)
    inf_any = np.isinf(df[NUMERIC_COLS].to_numpy(dtype=float)).any(axis=1)

    nonpositive = ~np.all(price_arr > 0, axis=1)  # any price <= 0 or NaN
    hi_oc = np.maximum(o, c)
    lo_oc = np.minimum(o, c)
    with np.errstate(invalid="ignore"):
        high_lt_low = h < l
        oc_outside = (h < hi_oc) | (l > lo_oc)
        neg_vol = v < 0

    dup_dates = int(dates.duplicated().sum())
    parseable = dates.notna()
    non_monotonic = 0
    if parseable.all():
        deltas = dates.diff().dropna()
        non_monotonic = int((deltas <= pd.Timedelta(0)).sum())
    weekend = int((dates.dt.dayofweek >= 5).sum()) if parseable.all() else -1

    lead_flat = _leading_flat_run(h, l)
    first_real_date = ""
    if 0 < lead_flat < n:
        first_real_date = dates.iloc[lead_flat].strftime("%Y-%m-%d")

    return {
        "ticker": _ticker_of(path),
        "rows": n,
        "first_date": dates.iloc[0].strftime("%Y-%m-%d") if parseable.iloc[0] else "NaT",
        "last_date": dates.iloc[-1].strftime("%Y-%m-%d") if parseable.iloc[-1] else "NaT",
        "unparseable_dates": int((~parseable).sum()),
        "nan_rows": int(nan_any.sum()),
        "inf_rows": int(inf_any.sum()),
        "nonpositive_rows": int(np.nansum(nonpositive) if nonpositive.dtype != bool else nonpositive.sum()),
        "high_lt_low_rows": int(np.nansum(high_lt_low)),
        "oc_outside_rows": int(np.nansum(oc_outside)),
        "neg_volume_rows": int(np.nansum(neg_vol)),
        "dup_dates": dup_dates,
        "non_monotonic": non_monotonic,
        "weekend_rows": weekend,
        "leading_flat_run": lead_flat,
        "first_real_date": first_real_date,
        "all_zero_volume": bool(np.nan_to_num(v).sum() == 0),
        "zero_volume_rows": int((v == 0).sum()),
    }


# --------------------------------------------------------------------------- #
# Stage 2: clean + process to Parkinson variance
# --------------------------------------------------------------------------- #

def clean_and_process(path: Path) -> tuple[pd.DataFrame, dict]:
    """Return (processed_df[date,parkinson_variance], corrections)."""
    df = _load_raw(path)
    ticker = _ticker_of(path)
    n0 = len(df)

    corr = {
        "ticker": ticker,
        "raw_rows": n0,
        "dropped_unparseable_date": 0,
        "dropped_dup_date": 0,
        "dropped_weekend": 0,
        "dropped_no_positive_price": 0,
        "ohlc_rows_corrected": 0,
        "leading_backfill_trimmed": 0,
        "trim_first_real_date": "",
        "dropped_nan_inf_parkinson": 0,
        "clip_hits": 0,
        "out_rows": 0,
    }

    # (a) drop unparseable dates
    mask = df["date"].notna()
    corr["dropped_unparseable_date"] = int((~mask).sum())
    df = df[mask].copy()

    # (b) drop exact-duplicate dates, keep last; (c) sort ascending
    before = len(df)
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    corr["dropped_dup_date"] = before - len(df)
    df = df.reset_index(drop=True)

    # (d) weekday-only
    wk = df["date"].dt.dayofweek < 5
    corr["dropped_weekend"] = int((~wk).sum())
    df = df[wk].reset_index(drop=True)

    # (e) fix only rows that are Parkinson-invalid (high/low non-positive, NaN,
    # or high < low) via per-row max/min over the POSITIVE prices of that row.
    # Parkinson variance uses only high/low, so rows where high >= low > 0 are
    # left untouched even if open/close fall outside [low, high] (matches how the
    # delivered vn30/vn100 processed data was produced -- no invented data).
    old_high = df["high"].to_numpy(dtype=float).copy()
    old_low = df["low"].to_numpy(dtype=float).copy()
    invalid = (
        ~(old_high > 0) | ~(old_low > 0) | (old_high < old_low)
    )  # NaN high/low -> ~(>0) is True -> invalid

    prices = df[PRICE_COLS].to_numpy(dtype=float)
    pos = np.where(prices > 0, prices, np.nan)  # keep only positive prices
    has_pos = np.sum(~np.isnan(pos), axis=1) > 0

    # Rows to drop: invalid AND no positive price to recover from.
    drop_mask = invalid & ~has_pos
    corr["dropped_no_positive_price"] = int(drop_mask.sum())

    fix_mask = invalid & has_pos
    corr["ohlc_rows_corrected"] = int(fix_mask.sum())
    if fix_mask.any():
        old_high[fix_mask] = np.nanmax(pos[fix_mask], axis=1)
        old_low[fix_mask] = np.nanmin(pos[fix_mask], axis=1)
    df = df.assign(high=old_high, low=old_low)
    df = df[~drop_mask].reset_index(drop=True)

    # (f) trim leading backfill (leading flat run high==low) to first real day
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    lead = _leading_flat_run(h, l)
    if 0 < lead < len(df):
        corr["leading_backfill_trimmed"] = lead
        corr["trim_first_real_date"] = df["date"].iloc[lead].strftime("%Y-%m-%d")
        df = df.iloc[lead:].reset_index(drop=True)
    elif lead >= len(df) and len(df) > 0:
        # all rows flat -> keep (degenerate), record
        corr["leading_backfill_trimmed"] = 0

    # compute Parkinson variance (reuse repo formula)
    park = calculate_parkinson_variance(df.rename(columns={"high": "high", "low": "low"}))
    out = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "parkinson_variance": park.to_numpy(dtype=float),
    })
    before = len(out)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    corr["dropped_nan_inf_parkinson"] = before - len(out)
    corr["clip_hits"] = int((out["parkinson_variance"] > CLIP_CEILING).sum())
    out["parkinson_variance"] = out["parkinson_variance"].clip(upper=CLIP_CEILING)
    corr["out_rows"] = len(out)
    return out, corr


def run_scan(raw_dir: Path, out_json: Path) -> None:
    paths = _discover(raw_dir)
    rows = [scan_ticker(p) for p in paths]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[scan] {len(rows)} tickers -> {out_json}")


def run_etl(raw_dir: Path, processed_dir: Path, out_json: Path) -> None:
    paths = _discover(raw_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    corrections = []
    for p in paths:
        out, corr = clean_and_process(p)
        out.to_csv(processed_dir / f"{corr['ticker']}_processed.csv", index=False)
        corrections.append(corr)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(corrections, indent=2), encoding="utf-8")
    total_out = sum(c["out_rows"] for c in corrections)
    print(f"[etl] {len(corrections)} tickers -> {processed_dir} ({total_out} rows) ; log -> {out_json}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan", "etl"])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--processed", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    raw_dir = Path(args.raw)
    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir
    out_json = Path(args.out)

    if args.mode == "scan":
        run_scan(raw_dir, out_json)
    else:
        processed = Path(args.processed)
        if not processed.is_absolute():
            processed = PROJECT_ROOT / processed
        run_etl(raw_dir, processed, out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
