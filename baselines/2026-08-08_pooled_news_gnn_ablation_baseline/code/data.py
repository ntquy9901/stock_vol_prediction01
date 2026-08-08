"""Validated raw price records and chronological per-ticker splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SampleKey:
    """Stable identity for one pooled forecast sample."""

    ticker_id: int
    ticker: str
    target_date: str


@dataclass(frozen=True)
class PooledSample:
    """Raw values retained before Task 2 preprocessing and window creation."""

    key: SampleKey
    x_price_raw: np.ndarray
    x_news: np.ndarray
    news_mask: np.ndarray
    y_raw: float


@dataclass(frozen=True)
class SplitFrames:
    """Validated raw split frames and the deterministic ticker vocabulary."""

    frames: dict[str, dict[str, pd.DataFrame]]
    ticker_to_id: dict[str, int]


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != len(_SPLIT_NAMES) or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must contain three positive values")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to 1.0")


def _validated_dates(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns:
        raise ValueError("price data must contain a date column")
    if df.empty:
        raise ValueError("price data must not be empty")

    dates = pd.to_datetime(df["date"], errors="raise")
    if dates.duplicated().any():
        raise ValueError("duplicate dates are not allowed")
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be monotonic increasing")
    return dates


def chronological_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, pd.DataFrame]:
    """Validate and split one ticker's rows in their supplied chronological order."""

    _validate_ratios(ratios)
    dates = _validated_dates(df)
    train_end = int(len(df) * ratios[0])
    val_end = train_end + int(len(df) * ratios[1])
    boundaries = (0, train_end, val_end, len(df))
    parts = {
        name: df.iloc[boundaries[index] : boundaries[index + 1]].copy()
        for index, name in enumerate(_SPLIT_NAMES)
    }
    if any(part.empty for part in parts.values()):
        raise ValueError("each chronological split must contain at least one row")

    for part in parts.values():
        part["date"] = dates.loc[part.index].to_numpy()
    return parts


def load_and_split_price_data(
    data_dir: Path | str,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> SplitFrames:
    """Load all processed ticker files and split each ticker without reordering rows."""

    _validate_ratios(ratios)
    directory = Path(data_dir)
    paths = sorted(directory.glob("*_processed.csv"))
    if not paths:
        raise ValueError(f"no *_processed.csv price files found under {directory}")

    tickers = [path.stem.removesuffix("_processed") for path in paths]
    if len(set(tickers)) != len(tickers):
        raise ValueError("ticker file stems must be unique")

    frames = {
        ticker: chronological_split(pd.read_csv(path), ratios)
        for ticker, path in zip(tickers, paths, strict=True)
    }
    return SplitFrames(
        frames=frames,
        ticker_to_id={ticker: ticker_id for ticker_id, ticker in enumerate(tickers)},
    )
