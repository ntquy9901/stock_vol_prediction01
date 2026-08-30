"""Expanding-window walk-forward fold construction + leakage guards (anchor-position space).

A fold's boundaries are expressed as slices into the panel's anchor array. For a retrain point ``r``
(an anchor position) with purge gap ``horizon`` and validation-tail length ``val``:

    train    = [0 .. r-horizon-val)      (expanding: grows every fold)
    val       = [r-horizon-val .. r-horizon)
    purge     = [r-horizon .. r)         (length == horizon)
    forecast  = [r .. r+K)               (the OOS block scored this fold)

``assert_no_leakage`` is the executable form of the three requirements acceptance criteria; it raises
on any violation (overlap, wrong purge, non-expanding train, or a train/val target date that reaches
into the forecast region).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    idx: int
    train: slice
    val: slice
    forecast: slice
    purge: slice


def make_folds(n: int, test_start: int, K: int, val: int, horizon: int) -> list[Fold]:
    """Tile ``[test_start, n)`` into contiguous forecast blocks of length ``K`` (last block short),
    each with an expanding train window and a ``val``-length validation tail behind a ``horizon`` purge."""
    if not 0 < test_start < n:
        raise ValueError(f"test_start must be in (0, n); got test_start={test_start}, n={n}")
    if K < 1 or val < 1 or horizon < 1:
        raise ValueError(f"K, val, horizon must be >= 1; got K={K}, val={val}, horizon={horizon}")
    folds: list[Fold] = []
    for i, r in enumerate(range(test_start, n, K)):
        tr_stop = r - horizon - val
        if tr_stop <= 0:
            raise ValueError(f"fold {i}: empty train window (r={r} <= horizon+val={horizon + val})")
        folds.append(Fold(i, slice(0, tr_stop), slice(tr_stop, r - horizon),
                          slice(r, min(r + K, n)), slice(r - horizon, r)))
    return folds


def _positions(sl: slice, n: int) -> range:
    return range(*sl.indices(n))


def assert_no_leakage(folds: list[Fold], target_dates: np.ndarray, horizon: int) -> None:
    """Raise ``AssertionError`` unless every fold is leakage-free and the train windows expand.

    ``target_dates[p]`` is the forecast-target date of anchor position ``p`` (i.e. ``dates[anchor_p +
    horizon]``); the date-space check pins that no train/val target date reaches the forecast region.
    """
    if not folds:
        raise AssertionError("no folds to validate")
    n = len(target_dates)
    prev_train_stop = -1
    for f in folds:
        tr = set(_positions(f.train, n))
        va = set(_positions(f.val, n))
        fc = set(_positions(f.forecast, n))
        # (a) forecast positions disjoint from train and val (primary leakage guard)
        if fc & tr or fc & va:
            raise AssertionError(f"fold {f.idx}: forecast overlaps train/val positions")
        # (b) contiguous train|val, purge == horizon
        if f.train.stop != f.val.start:
            raise AssertionError(f"fold {f.idx}: train.stop {f.train.stop} != val.start {f.val.start}")
        if f.val.stop != f.forecast.start - horizon:
            raise AssertionError(
                f"fold {f.idx}: purge != horizon (val.stop {f.val.stop}, forecast.start {f.forecast.start})")
        if (f.purge.start, f.purge.stop) != (f.forecast.start - horizon, f.forecast.start):
            raise AssertionError(f"fold {f.idx}: purge slice {f.purge} inconsistent with forecast/horizon")
        # (c) expanding train window across folds
        if f.train.stop <= prev_train_stop:
            raise AssertionError(f"fold {f.idx}: train not expanding (stop {f.train.stop} <= {prev_train_stop})")
        prev_train_stop = f.train.stop
        # date-space purge: every train/val target date strictly precedes every forecast target date
        trva = sorted(tr | va)
        if trva and fc:
            if target_dates[trva].max() >= target_dates[sorted(fc)].min():
                raise AssertionError(
                    f"fold {f.idx}: a train/val target date reaches the forecast region (leakage)")
