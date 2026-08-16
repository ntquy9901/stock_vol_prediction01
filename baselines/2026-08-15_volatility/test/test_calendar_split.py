"""Configurable train/val/test split: RATIO mode (70/15/15, default, unchanged) and CALENDAR mode
(fixed start/end dates per block, leak-safe non-overlap, per-ticker availability for heterogeneous
listing dates)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
PILOT = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
if str(PILOT) not in sys.path:
    sys.path.insert(0, str(PILOT))

import data as d  # noqa: E402

CAL = (("2020-01-01", "2020-12-31"), ("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"))


def _frame(start, periods):
    dates = pd.bdate_range(start, periods=periods)
    return pd.DataFrame({"date": dates.astype(str),
                         "parkinson_volatility": [0.01] * periods})


def test_calendar_split_partitions_by_date():
    df = _frame("2020-01-01", 780)                       # ~3 business years
    parts = d.calendar_split(df, CAL)
    assert set(parts) == {"train", "val", "test"}
    assert (pd.to_datetime(parts["train"]["date"]).dt.year == 2020).all()
    assert (pd.to_datetime(parts["val"]["date"]).dt.year == 2021).all()
    assert (pd.to_datetime(parts["test"]["date"]).dt.year == 2022).all()
    assert len(parts["train"]) > 200 and len(parts["val"]) > 200


def test_calendar_rejects_overlapping_blocks():
    bad = (("2020-01-01", "2021-06-30"), ("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"))
    with pytest.raises(ValueError, match="ordered/non-overlapping"):
        d.calendar_split(_frame("2020-01-01", 780), bad)


def test_calendar_rejects_misordered_blocks():
    bad = (("2022-01-01", "2022-12-31"), ("2021-01-01", "2021-12-31"), ("2020-01-01", "2020-12-31"))
    with pytest.raises(ValueError, match="ordered/non-overlapping"):
        d.calendar_split(_frame("2020-01-01", 780), bad)


def test_calendar_allows_empty_block_for_late_listing():
    df = _frame("2021-02-01", 400)                        # listed AFTER the 2020 train window
    parts = d.calendar_split(df, CAL)
    assert len(parts["train"]) == 0                       # empty, but no raise
    assert len(parts["val"]) > 0


def test_load_calendar_excludes_insufficient_train(tmp_path):
    _frame("2020-01-01", 780).to_csv(tmp_path / "OLD_processed.csv", index=False)   # full history
    _frame("2021-06-01", 300).to_csv(tmp_path / "NEW_processed.csv", index=False)   # listed 2021 -> no train
    sf = d.load_and_split_price_data(tmp_path, calendar=CAL, min_train_rows=60)
    assert set(sf.ticker_to_id) == {"OLD"}               # NEW excluded (empty train)
    assert list(sf.ticker_to_id.values()) == [0]         # ids re-indexed over kept tickers


def test_load_ratio_mode_unchanged(tmp_path):
    _frame("2020-01-01", 100).to_csv(tmp_path / "AAA_processed.csv", index=False)
    sf = d.load_and_split_price_data(tmp_path)            # default ratio mode
    parts = sf.frames["AAA"]
    assert len(parts["train"]) == 70 and len(parts["val"]) == 15 and len(parts["test"]) == 15
