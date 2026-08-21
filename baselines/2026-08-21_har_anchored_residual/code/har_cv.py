"""Cross-fitted HAR predictions on training anchors, for residual-target construction (plan section 10).

Residual experts (E5-E8) must learn the part of the target that HAR does not explain. Building the
residual target from a HAR model fitted on the same row (in-sample) yields optimistic residuals that
differ from deployment. This module produces expanding-window out-of-sample HAR predictions: the train
anchors are cut into ``n_folds`` time-ordered blocks and each block is predicted by a HAR fit on the
strictly-earlier blocks only. The first block has no earlier data and is left as NaN (used for training
the HAR seed, not for residual targets).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
import baselines as B  # noqa: E402  (reuse the tested HAR OLS)


def crossfit_har(
    feats: np.ndarray, pk: np.ndarray, anchors_train: np.ndarray,
    horizon: int, n_folds: int = 5, floor: float = 1e-8,
) -> np.ndarray:
    """Expanding-window OOS HAR predictions for ``anchors_train``.

    Returns an array aligned to ``anchors_train`` with NaN for the first block (training seed) and the
    OOS HAR prediction elsewhere. ``feats`` is the [N,3] raw HAR feature matrix; the target for anchor
    ``t`` is ``pk[t + horizon]``.
    """
    anchors_train = np.asarray(anchors_train)
    n = len(anchors_train)
    oos = np.full(n, np.nan, dtype=np.float64)
    blocks = np.array_split(np.arange(n), n_folds)
    for k in range(1, len(blocks)):
        fit_pos = np.concatenate(blocks[:k])
        a_fit = anchors_train[fit_pos]
        coef = B.har_fit(feats[a_fit], pk[a_fit + horizon])
        a_pred = anchors_train[blocks[k]]
        oos[blocks[k]] = B.har_predict(feats[a_pred], coef, floor=floor)
    return oos
