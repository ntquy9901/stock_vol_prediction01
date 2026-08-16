"""Positive-aware OHLC sanitization for raw ``*_ohlcv.csv`` price files.

Makes each row internally consistent in one idempotent pass:
  * ``high`` = max of the row's POSITIVE OHLC values,
  * ``low``  = min of the row's POSITIVE OHLC values,
  * any nonpositive ``open``/``close`` is clamped into ``[low, high]``.

This fixes the three raw-feed defects that data-quality tests flag — ``high < low`` (Parkinson
variance is driven by H/L, so this is the critical one), nonpositive prices, and ``open``/``close``
falling outside ``[low, high]`` — without dropping any row. Nonpositive values are excluded from the
max/min so a spurious ``0`` is never propagated into ``low``. Rows whose OHLC are ALL nonpositive
cannot be reconstructed and are left untouched (reported by the caller, not silently zeroed).

Parkinson variance uses only H/L, so after cleaning ``high >= low`` holds and the target is
well-defined. Idempotent: re-running on cleaned data changes nothing.

CLI:  python -m src.data.clean_ohlc <dir>   # cleans every *_ohlcv.csv in <dir> in place
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]
_TOL = 1e-9


def clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return ``(cleaned_df, n_cells_changed)``.

    A row with at least one positive OHLC value is reconstructed; a row with all-nonpositive OHLC is
    left as-is (it cannot be reconstructed from itself).
    """

    out = df.copy()
    vals = out[OHLC].astype(float)
    pos = vals.where(vals > 0)              # nonpositive -> NaN, excluded from max/min
    hi = pos.max(axis=1)
    lo = pos.min(axis=1)
    valid = hi.notna() & lo.notna()         # at least one positive value present
    new_high = np.where(valid, hi, vals["high"])
    new_low = np.where(valid, lo, vals["low"])
    new_open = np.where(valid & (vals["open"] <= 0), new_low, vals["open"])
    new_close = np.where(valid & (vals["close"] <= 0), new_low, vals["close"])
    changed = int(
        (np.abs(new_high - vals["high"].to_numpy()) > _TOL).sum()
        + (np.abs(new_low - vals["low"].to_numpy()) > _TOL).sum()
        + (np.abs(new_open - vals["open"].to_numpy()) > _TOL).sum()
        + (np.abs(new_close - vals["close"].to_numpy()) > _TOL).sum()
    )
    out["high"] = new_high
    out["low"] = new_low
    out["open"] = new_open
    out["close"] = new_close
    return out, changed


def clean_file(path: Path) -> int:
    """Clean one ``*_ohlcv.csv`` in place; returns the number of cells corrected."""

    df = pd.read_csv(path, dtype={"date": str})   # keep the date column verbatim (tz-aware or plain)
    cleaned, changed = clean_frame(df)
    if changed:
        cleaned.to_csv(path, index=False)
    return changed


def clean_dir(directory: Path) -> dict[str, int]:
    """Clean every ``*_ohlcv.csv`` in ``directory`` in place; returns ticker -> cells corrected."""

    return {p.stem.replace("_ohlcv", ""): clean_file(p)
            for p in sorted(Path(directory).glob("*_ohlcv.csv"))}


def main(directory: str) -> None:
    res = clean_dir(Path(directory))
    changed = {k: v for k, v in res.items() if v}
    for ticker, n in changed.items():
        print(f"{ticker}: {n} cells corrected")
    print(f"cleaned {len(res)} files; {len(changed)} changed; {sum(res.values())} cells total")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/prices")
