"""TDD: leakage-safe purged split. The core invariant is that no earlier split's target date
(anchor + horizon) can fall on or after the first anchor of a later split."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "code"))

import folds  # noqa: E402


@pytest.mark.parametrize("horizon", [1, 5, 10, 22])
def test_purge_no_target_overlap(horizon):
    anchors = np.arange(100, 1100)          # 1000 consecutive trading-day anchors
    tr, va, te = folds.purged_split(anchors, horizon, train_frac=0.8, val_frac=0.1)
    assert len(tr) and len(va) and len(te)
    # every train target day is strictly before the first val anchor day
    assert tr.max() + horizon < va.min()
    # every val target day is strictly before the first test anchor day
    assert va.max() + horizon < te.min()
    # test set is unchanged by purging (only train/val shrink)
    n = len(anchors); i_va = int(n * 0.9)
    assert np.array_equal(te, anchors[i_va:])


def test_purge_reduces_train_and_val_by_horizon():
    anchors = np.arange(0, 1000)
    tr0 = anchors[:800]; va0 = anchors[800:900]
    tr, va, te = folds.purged_split(anchors, horizon=5, train_frac=0.8, val_frac=0.1)
    assert len(tr) == len(tr0) - 5
    assert len(va) == len(va0) - 5


def test_purge_contiguous_and_ordered():
    anchors = np.arange(0, 500)
    tr, va, te = folds.purged_split(anchors, horizon=3, train_frac=0.8, val_frac=0.1)
    for part in (tr, va, te):
        assert np.array_equal(part, np.arange(part[0], part[-1] + 1))
    assert tr.max() < va.min() <= va.max() < te.min()


def test_empty_when_horizon_too_large_for_split():
    anchors = np.arange(0, 100)             # val block ~10 anchors; horizon 20 wipes it
    tr, va, te = folds.purged_split(anchors, horizon=20, train_frac=0.8, val_frac=0.1)
    assert len(va) == 0                      # val fully purged -> caller must skip this ticker/horizon
