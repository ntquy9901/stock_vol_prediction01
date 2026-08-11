"""Robust raw-OHLCV loading, cross-stock alignment, and chronological split.

Handles the two heterogeneous date/scale conventions found in ``data/raw/prices``:
- plain ``YYYY-MM-DD`` dates in thousands-VND (most tickers), and
- tz-aware ``YYYY-MM-DD 00:00:00+07:00`` datetimes in absolute VND (VPB, VRE).

Price SCALE differs across tickers but every downstream quantity is a log-ratio
(log return, Parkinson log(high/low)), which is invariant to a constant multiplicative
scale, so no scale harmonisation is applied (see ``data/raw/prices/LPB_SOURCE_NOTE.md``).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def load_ticker(path: str) -> pd.DataFrame:
    """Load one OHLCV csv, normalising heterogeneous date formats to naive dates.

    Returns a frame sorted by date with duplicate dates dropped (first kept) and a
    ``ticker`` column. Dates are normalised to the Asia/Ho_Chi_Minh calendar day so
    tz-aware and tz-naive sources align.
    """
    ticker = os.path.basename(path).split("_")[0]
    df = pd.read_csv(path)
    dt = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["date"] = dt.dt.tz_convert("Asia/Ho_Chi_Minh").dt.normalize().dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset="date", keep="first").reset_index(drop=True)
    df["ticker"] = ticker
    return df[["date", *OHLCV_COLS, "ticker"]]


def load_universe(price_dir: str) -> dict[str, pd.DataFrame]:
    """Load every ``*_ohlcv.csv`` in ``price_dir`` keyed by ticker."""
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob(os.path.join(price_dir, "*_ohlcv.csv"))):
        d = load_ticker(path)
        out[d["ticker"].iloc[0]] = d
    return out


def build_common_panel(
    frames: dict[str, pd.DataFrame], column: str
) -> pd.DataFrame:
    """Wide ``date x ticker`` matrix for one column, on dates common to ALL tickers.

    No forward-fill: only dates present for every ticker are retained (plan section 6
    default — do not fill prices through non-trading observations).
    """
    common: set | None = None
    for d in frames.values():
        s = set(d["date"])
        common = s if common is None else (common & s)
    dates = sorted(common or set())
    wide = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    for t, d in frames.items():
        s = d.set_index("date")[column]
        wide[t] = s.reindex(dates)
    return wide.sort_index()


@dataclass(frozen=True)
class ChronoSplit:
    """Chronological (leakage-safe) split boundaries over an ordered date index."""

    train_end: pd.Timestamp
    val_end: pd.Timestamp
    dates: pd.DatetimeIndex

    def mask(self, index: pd.DatetimeIndex, part: str) -> np.ndarray:
        if part == "train":
            return index <= self.train_end
        if part == "val":
            return (index > self.train_end) & (index <= self.val_end)
        if part == "test":
            return index > self.val_end
        raise ValueError(f"unknown part {part!r}")


def chrono_split(
    dates: pd.DatetimeIndex, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> ChronoSplit:
    """Positional chronological split (earliest -> train, latest -> test).

    Boundaries are inclusive train_end / val_end timestamps so any date index can be
    masked without re-deriving positions.
    """
    dates = pd.DatetimeIndex(sorted(dates))
    n = len(dates)
    i_train = int(round(n * train_ratio)) - 1
    i_val = int(round(n * (train_ratio + val_ratio))) - 1
    i_train = max(0, min(i_train, n - 1))
    i_val = max(i_train, min(i_val, n - 1))
    return ChronoSplit(train_end=dates[i_train], val_end=dates[i_val], dates=dates)
