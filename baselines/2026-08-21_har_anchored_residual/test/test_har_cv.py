"""TDD: expanding-window cross-fitted HAR predictions for residual-target construction (plan section 10).

The key non-leakage property: the OOS HAR prediction for a training block must depend only on EARLIER
blocks, so perturbing a block's own targets cannot change that block's OOS prediction.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "submission" / "soict_lstm_gat"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import data_utils as du  # noqa: E402
import har_cv  # noqa: E402


def _synth(n=600, horizon=5):
    rng = np.random.default_rng(0)
    pk = np.abs(rng.normal(0.02, 0.01, n)) + 1e-4
    feats = du.har_features(pk)
    anchors = du.make_windows(pk, lookback=10, horizon=horizon)
    tr, _, _ = du.per_stock_split(anchors, 0.8, 0.1)
    return pk, feats, tr


def test_shape_and_block0_is_nan():
    pk, feats, tr = _synth()
    oos = har_cv.crossfit_har(feats, pk, tr, horizon=5, n_folds=5)
    assert oos.shape == (len(tr),)
    blocks = np.array_split(np.arange(len(tr)), 5)
    assert np.all(np.isnan(oos[blocks[0]]))          # first block is training-seed only -> no OOS pred
    assert np.all(np.isfinite(oos[np.concatenate(blocks[1:])]))


def test_oos_pred_ignores_own_block_targets():
    pk, feats, tr = _synth()
    oos_a = har_cv.crossfit_har(feats, pk, tr, horizon=5, n_folds=5)
    # perturb the targets of the LAST block only (dates strictly after all earlier blocks)
    blocks = np.array_split(np.arange(len(tr)), 5)
    last = blocks[-1]
    pk2 = pk.copy()
    pk2[tr[last] + 5] += 10.0                         # huge change to the last block's own targets
    oos_b = har_cv.crossfit_har(feats, pk2, tr, horizon=5, n_folds=5)
    # earlier blocks' OOS preds unchanged; last block's OOS pred also unchanged (fit excludes it)
    assert np.allclose(oos_a[last], oos_b[last], equal_nan=True)


def test_residuals_finite_where_predicted():
    pk, feats, tr = _synth()
    oos = har_cv.crossfit_har(feats, pk, tr, horizon=5, n_folds=5)
    mask = np.isfinite(oos)
    resid = pk[tr + 5][mask] - oos[mask]
    assert np.isfinite(resid).all() and resid.shape[0] > 0
