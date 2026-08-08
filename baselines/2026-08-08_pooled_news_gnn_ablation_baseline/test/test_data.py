"""Contract tests for validated raw per-ticker price splits."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
for _path in (str(_ROOT), str(_CODE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data import (  # noqa: E402
    PooledSample,
    SampleKey,
    chronological_split,
    load_and_split_price_data,
)


def _frame(size: int, ticker: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=size, freq="B").strftime("%Y-%m-%d"),
            "parkinson_volatility": np.linspace(0.01, 0.02, size),
            "ticker": ticker,
        }
    )


def _frame_with_duplicate_date() -> pd.DataFrame:
    frame = _frame(100)
    frame.loc[99, "date"] = frame.loc[98, "date"]
    return frame


def test_split_is_per_ticker_chronological_and_disjoint() -> None:
    parts = chronological_split(_frame(100))

    assert [len(parts[key]) for key in ("train", "val", "test")] == [70, 15, 15]
    assert parts["train"].date.max() < parts["val"].date.min()
    assert parts["val"].date.max() < parts["test"].date.min()


def test_invalid_duplicate_date_fails_before_split() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        chronological_split(_frame_with_duplicate_date())


def test_non_monotonic_dates_fail_before_split() -> None:
    frame = _frame(100)
    frame.loc[[1, 2], "date"] = frame.loc[[2, 1], "date"].to_numpy()

    with pytest.raises(ValueError, match="monotonic"):
        chronological_split(frame)


def test_raw_records_are_frozen_and_preserve_values() -> None:
    key = SampleKey(ticker_id=4, ticker="AAA", target_date="2021-01-08")
    sample = PooledSample(
        key=key,
        x_price_raw=np.array([[0.1]]),
        x_news=np.array([[0.0]]),
        news_mask=np.array([0]),
        y_raw=0.2,
    )

    assert sample.key == key
    assert sample.y_raw == 0.2
    with pytest.raises(Exception):
        key.ticker = "ZZZ"  # type: ignore[misc]


def test_ticker_ids_are_sorted_and_stable(tmp_path: Path) -> None:
    _frame(100, ticker="ZZZ").drop(columns="ticker").to_csv(tmp_path / "ZZZ_processed.csv", index=False)
    _frame(100, ticker="AAA").drop(columns="ticker").to_csv(tmp_path / "AAA_processed.csv", index=False)

    splits = load_and_split_price_data(tmp_path)

    assert splits.ticker_to_id == {"AAA": 0, "ZZZ": 1}
    assert set(splits.frames) == {"AAA", "ZZZ"}
    assert [len(splits.frames["AAA"][key]) for key in ("train", "val", "test")] == [70, 15, 15]
