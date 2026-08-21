"""Leakage-safe chronological split with target-overlap purge (fixes leakage finding F1).

The existing ``data_utils.per_stock_split`` cuts contiguous anchor ranges with zero gap, so the last
(h-1) train anchors have target dates ``t+h`` that fall inside the validation date range (and likewise
val->test). Plan section 19 mandates purging those rows, with the purge derived from the exact target
interval. For a terminal-value target at ``t+h`` the purge gap is exactly ``horizon`` anchors.
"""
from __future__ import annotations

import numpy as np


def purged_split(
    anchors: np.ndarray, horizon: int, train_frac: float = 0.8, val_frac: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological 80/10/10 split with the last ``horizon`` anchors of train and of val purged.

    Purging the final ``horizon`` anchors of a split guarantees that split's target dates
    (anchor + horizon) all land strictly before the first anchor of the next split, so no target
    interval crosses a split boundary. The test split is never purged (nothing follows it).

    Returns (train, val, test) anchor arrays; ``val`` (or ``train``) may be empty when the horizon is
    large relative to the split size, in which case the caller must skip that ticker/horizon.
    """
    anchors = np.asarray(anchors)
    n = len(anchors)
    i_tr = int(n * train_frac)
    i_va = int(n * (train_frac + val_frac))
    train, val, test = anchors[:i_tr], anchors[i_tr:i_va], anchors[i_va:]
    train = train[:-horizon] if horizon < len(train) else train[:0]
    val = val[:-horizon] if horizon < len(val) else val[:0]
    return train, val, test
