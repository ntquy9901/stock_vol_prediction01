"""Fold construction + leakage-guard coverage (pure, no torch/data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

from wf_folds import Fold, assert_no_leakage, make_folds  # noqa: E402


def test_make_folds_tiles_and_expands():
    folds = make_folds(n=1000, test_start=900, K=66, val=66, horizon=1)
    assert len(folds) == 2
    # contiguous forecast tiling, last block short
    assert folds[0].forecast == slice(900, 966)
    assert folds[1].forecast == slice(966, 1000)
    # expanding train, purge == horizon, val length == 66
    assert folds[0].train == slice(0, 833) and folds[0].val == slice(833, 899)
    assert folds[1].train.stop > folds[0].train.stop
    assert folds[0].purge == slice(899, 900)


@pytest.mark.parametrize("kw", [
    {"test_start": 0}, {"test_start": 1000},                    # out of (0, n)
    {"K": 0}, {"val": 0}, {"horizon": 0},                       # non-positive params
])
def test_make_folds_rejects_bad_params(kw):
    base = {"n": 1000, "test_start": 900, "K": 66, "val": 66, "horizon": 1}
    base.update(kw)
    with pytest.raises(ValueError):
        make_folds(**base)


def test_make_folds_rejects_empty_train_window():
    # r=50 but horizon+val=61 -> tr_stop <= 0
    with pytest.raises(ValueError):
        make_folds(n=200, test_start=50, K=40, val=60, horizon=1)


def _valid_folds():
    return make_folds(n=1000, test_start=900, K=66, val=66, horizon=1)


def test_assert_no_leakage_passes_on_valid_folds():
    td = np.arange(1000).astype("datetime64[D]")
    assert_no_leakage(_valid_folds(), td, horizon=1)   # no raise


def test_assert_no_leakage_empty_folds_raises():
    td = np.arange(1000).astype("datetime64[D]")
    with pytest.raises(AssertionError, match="no folds"):
        assert_no_leakage([], td, horizon=1)


def test_assert_no_leakage_train_val_gap_raises():
    td = np.arange(1000).astype("datetime64[D]")
    bad = [Fold(0, slice(0, 800), slice(833, 899), slice(900, 966), slice(899, 900))]
    with pytest.raises(AssertionError, match="train.stop"):
        assert_no_leakage(bad, td, horizon=1)


def test_assert_no_leakage_wrong_purge_raises():
    td = np.arange(1000).astype("datetime64[D]")
    # val.stop (899) != forecast.start(966) - horizon(1) -> purge != horizon
    bad = [Fold(0, slice(0, 833), slice(833, 899), slice(966, 1000), slice(899, 966))]
    with pytest.raises(AssertionError, match="purge != horizon"):
        assert_no_leakage(bad, td, horizon=1)


def test_assert_no_leakage_inconsistent_purge_slice_raises():
    td = np.arange(1000).astype("datetime64[D]")
    # val.stop == forecast.start - horizon holds, but the purge slice itself is wrong
    bad = [Fold(0, slice(0, 833), slice(833, 899), slice(900, 966), slice(898, 900))]
    with pytest.raises(AssertionError, match="purge slice"):
        assert_no_leakage(bad, td, horizon=1)


def test_assert_no_leakage_forecast_overlap_raises():
    td = np.arange(1000).astype("datetime64[D]")
    # forecast reaches back into the val range -> positional overlap (checked before the purge structure)
    bad = [Fold(0, slice(0, 833), slice(833, 899), slice(880, 966), slice(879, 880))]
    with pytest.raises(AssertionError, match="overlaps"):
        assert_no_leakage(bad, td, horizon=1)


def test_assert_no_leakage_non_expanding_raises():
    td = np.arange(2000).astype("datetime64[D]")
    f0 = Fold(0, slice(0, 833), slice(833, 899), slice(900, 966), slice(899, 900))
    f1 = Fold(1, slice(0, 833), slice(833, 899), slice(900, 966), slice(899, 900))  # same train.stop
    with pytest.raises(AssertionError, match="not expanding"):
        assert_no_leakage([f0, f1], td, horizon=1)


def test_assert_no_leakage_empty_forecast_slice_skips_date_check():
    # a degenerate fold with an empty forecast block: the date-space check is skipped, no crash
    td = np.arange(1000).astype("datetime64[D]")
    ok = [Fold(0, slice(0, 899), slice(899, 965), slice(966, 966), slice(965, 966))]
    assert_no_leakage(ok, td, horizon=1)   # no raise


def test_assert_no_leakage_date_overlap_raises():
    # positions are fine, but the target-date array places a train date at/after the forecast date
    td = np.arange(1000).astype("datetime64[D]")
    td[832] = td[965]                                  # a train target date == a forecast target date
    with pytest.raises(AssertionError, match="reaches the forecast region"):
        assert_no_leakage(_valid_folds(), td, horizon=1)
