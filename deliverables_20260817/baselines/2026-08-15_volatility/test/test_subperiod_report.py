"""FU3b: temporal-stability robustness -- split the held-out test into sequential time blocks and
DM FULL vs HAR per block. Answers whether an advantage is stable over time or driven by one period.
(This is a temporal-stability check on the fixed test window, NOT full rolling recalibration.)
"""
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import subperiod_report as sp  # noqa: E402


def test_split_subperiods_sequential_by_date():
    dates = np.array(["2020-01-01", "2020-01-01", "2020-02-01", "2020-03-01",
                      "2020-04-01", "2020-05-01"])
    masks = sp.split_subperiods(dates, n_blocks=3)
    assert len(masks) == 3
    # 4 unique dates over 3 blocks -> block sizes by unique date: [2,1,1] unique -> obs grouping
    covered = np.zeros(len(dates), bool)
    for m in masks:
        assert not (covered & m).any()          # disjoint
        covered |= m
    assert covered.all()                        # partition
    # first block holds the earliest dates
    assert masks[0][0] and masks[0][1]          # both 2020-01-01 obs in block 0


def test_split_subperiods_keeps_same_date_together():
    dates = np.array(["d1", "d1", "d2", "d2", "d3", "d3"])
    masks = sp.split_subperiods(dates, n_blocks=3)
    # each unique date fully inside one block (no date split across blocks)
    for d in ("d1", "d2", "d3"):
        idx = np.where(dates == d)[0]
        in_block = [bool(m[idx].all()) for m in masks]
        assert sum(in_block) == 1               # exactly one block contains all of this date


def test_split_subperiods_validates_n_blocks():
    dates = np.array(["d1", "d2"])
    for bad in (0, -1):
        try:
            sp.split_subperiods(dates, n_blocks=bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
