"""Plan sections 6/32/65: alignment, chronological split, date-order (real-data smoke)."""

import pandas as pd
import pytest

from graph_eda.io_data import (
    build_common_panel,
    chrono_split,
    load_ticker,
    load_universe,
)
from graph_eda.leakage import assert_dates_sorted


def test_load_ticker_normalises_tzaware_dates():
    # VPB uses tz-aware "YYYY-MM-DD 00:00:00+07:00"; must become naive dates.
    d = load_ticker("data/raw/prices/VPB_ohlcv.csv")
    assert d["date"].dt.tz is None
    assert_dates_sorted(d)
    assert d["date"].is_unique


def test_load_ticker_sorted_plain_dates():
    d = load_ticker("data/raw/prices/ACB_ohlcv.csv")
    assert_dates_sorted(d)


@pytest.mark.smoke
def test_common_panel_and_split_real_universe():
    frames = load_universe("data/raw/prices")
    assert len(frames) == 33
    wide = build_common_panel(frames, "close")
    # heterogeneous VPB/VRE dates must still align into the common panel
    assert wide.shape[1] == 33
    assert wide.shape[0] > 1000
    assert wide.index.is_monotonic_increasing
    split = chrono_split(wide.index)
    assert split.train_end < split.val_end < wide.index.max()
    tr = split.mask(wide.index, "train")
    te = split.mask(wide.index, "test")
    # chronological: every train date strictly precedes every test date
    assert wide.index[tr].max() < wide.index[te].min()
    # partitions are disjoint and cover the panel
    assert tr.sum() + split.mask(wide.index, "val").sum() + te.sum() == len(wide.index)


def test_split_boundaries_ratio():
    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=100, freq="D"))
    s = chrono_split(idx, 0.70, 0.15)
    assert (idx <= s.train_end).sum() == 70
    assert ((idx > s.train_end) & (idx <= s.val_end)).sum() == 15
    assert (idx > s.val_end).sum() == 15
