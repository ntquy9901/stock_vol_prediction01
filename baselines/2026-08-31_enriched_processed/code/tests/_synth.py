"""Synthetic OHLCV builders for the enriched-processed tests (not a test module)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:  # pragma: no cover - test bootstrap
    sys.path.insert(0, str(_CODE))


def clean_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """A fully valid (geometry-clean, positive) OHLCV frame with weekday dates and positive volume."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.01, n)
    close = 100.0 * np.cumprod(1 + ret)
    openp = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "open": openp, "high": high, "low": low,
                         "close": close, "volume": vol})


def dirty_frame() -> pd.DataFrame:
    """A frame seeded with every dirty class so the ETL + flags + rejections are exercised."""
    base = clean_frame(n=40, seed=1).copy()
    # leading backfill: 2 constant-close zero-range, zero-volume rows at the top
    base.loc[0, ["open", "high", "low", "close"]] = 50.0
    base.loc[0, "volume"] = 0.0
    base.loc[1, ["open", "high", "low", "close"]] = 50.0
    base.loc[1, "volume"] = 0.0
    # NaN/inf row
    base.loc[5, "high"] = np.nan
    # nonpositive but reconstructable (>=2 positive) -> reconstruct
    base.loc[8, "low"] = -1.0
    # nonpositive unrecoverable (<2 positive) -> drop
    base.loc[9, ["open", "high", "low", "close"]] = [-1.0, -2.0, -3.0, 5.0]
    # high<low transposition -> swap
    o, c = base.loc[12, "open"], base.loc[12, "close"]
    base.loc[12, "high"] = min(o, c) - 0.01
    base.loc[12, "low"] = max(o, c) + 0.01
    # high<low unrecoverable -> drop
    base.loc[15, ["open", "high", "low", "close"]] = [1.0, 5.0, 10.0, 20.0]
    # open/close-outside -> widen_range (high dropped below max(open,close) but kept above low)
    base.loc[18, "high"] = min(base.loc[18, "open"], base.loc[18, "close"])
    # split jump (>50% up) -> backadjust
    base.loc[25, "close"] = base.loc[24, "close"] * 2.0
    base.loc[25, "high"] = base.loc[25, "close"] * 1.01
    base.loc[25, "low"] = base.loc[25, "close"] * 0.99
    base.loc[25, "open"] = base.loc[25, "close"]
    # zero-range + zero-volume flag row (valid, positive)
    base.loc[30, ["open", "high", "low", "close"]] = 60.0
    base.loc[30, "volume"] = 0.0
    return base


def write_market(tmp: Path, frames: dict) -> Path:
    """Write ``{ticker: raw_df}`` as ``<ticker>_ohlcv.csv`` into a tmp price dir; return the dir."""
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for tk, df in frames.items():
        df.to_csv(tmp / f"{tk}_ohlcv.csv", index=False)
    return tmp
