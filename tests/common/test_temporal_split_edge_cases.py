"""Tests for the AUD-018 hardening of src/common/temporal_split.py.

temporal_split_dataframe is the entry point most exposed to raw/less-trusted
input (a single-ticker date column, per every real caller in this repo --
see src/lstm_har_enhanced/train_with_overfitting_prevention.py and friends).
It must reject, rather than silently mishandle:
  - unparseable / NaT dates (including the mixed tz-aware/tz-naive class of
    bug this project already hit once for VPB/VRE, see
    src/common/parkinson_utils.py)
  - duplicate dates (a real per-ticker series should have exactly one row
    per date)
  - ratio/size combinations that would produce an empty train/val/test split
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.temporal_split import temporal_split_dataframe

pytestmark = pytest.mark.smoke


def _make_df(n=20):
    # Dates as ISO strings (object dtype), matching how real callers load
    # dates from CSV (e.g. tests/lstm_gat_hybrid/test_date_alignment_fix.py) --
    # this also lets tests assign an unparseable string into the column
    # without pandas' datetime64 dtype rejecting the assignment outright.
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "value": np.arange(n)})


class TestValidInputStillWorks:
    def test_splits_into_expected_sizes_and_chronological_order(self):
        df = _make_df(20)
        train, val, test = temporal_split_dataframe(df, 0.7, 0.15, 0.15)

        assert len(train) == 14
        assert len(val) == 3
        assert len(test) == 3
        assert train["date"].max() < val["date"].min()
        assert val["date"].max() < test["date"].min()


class TestNaTAndUnparseableDates:
    def test_raises_on_unparseable_date_string(self):
        df = _make_df(20)
        df.loc[5, "date"] = "not a date"
        with pytest.raises(ValueError, match="unparseable/NaT"):
            temporal_split_dataframe(df, 0.7, 0.15, 0.15)

    def test_raises_on_explicit_nat(self):
        df = _make_df(20)
        df.loc[3, "date"] = pd.NaT
        with pytest.raises(ValueError, match="unparseable/NaT"):
            temporal_split_dataframe(df, 0.7, 0.15, 0.15)

    def test_raises_on_mixed_tz_aware_and_naive_dates(self):
        """Mirrors the VPB/VRE bug class: one row tz-aware, the rest tz-naive
        date strings. pd.to_datetime(..., errors='coerce') -- called as the
        first line of our new validation -- itself raises ValueError for this
        exact shape ("Mixed timezones detected...") rather than coercing to
        NaT. That is pandas' own fail-loud behavior propagating through our
        function; per AUD-018 we don't add tz-normalization logic here (that's
        parkinson_utils.py's job), we just confirm this doesn't silently
        misbehave -- and it doesn't: it raises ValueError, unprompted."""
        df = _make_df(20)
        df["date"] = df["date"].astype(object)
        df.loc[7, "date"] = pd.Timestamp("2024-01-10", tz="Asia/Ho_Chi_Minh")
        with pytest.raises(ValueError, match="[Mm]ixed timezone"):
            temporal_split_dataframe(df, 0.7, 0.15, 0.15)


class TestDuplicateDates:
    def test_raises_on_duplicate_dates(self):
        df = _make_df(20)
        df.loc[5, "date"] = df.loc[4, "date"]
        with pytest.raises(ValueError, match="duplicate date"):
            temporal_split_dataframe(df, 0.7, 0.15, 0.15)


class TestEmptyPartition:
    def test_raises_when_dataset_too_small_for_ratios(self):
        """3 rows at 70/15/15 -> train=2, val=0, test=0: val/test would be empty."""
        df = _make_df(3)
        with pytest.raises(ValueError, match="empty partition"):
            temporal_split_dataframe(df, 0.7, 0.15, 0.15)

    def test_raises_when_val_ratio_rounds_to_zero(self):
        df = _make_df(10)
        with pytest.raises(ValueError, match="empty partition"):
            temporal_split_dataframe(df, 0.95, 0.02, 0.03)
