"""Chronological OOF HAR-X: cutoff (no look-ahead), warm-up exclusion, residual/reconstruct identity."""
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))

import xgb_config as C  # noqa: E402
import xgb_oof as O  # noqa: E402


def _synthetic(a=60, n=4, seed=0):
    rng = np.random.default_rng(seed)
    har5 = rng.normal(size=(a, n, 5))
    coef = np.array([0.3, 1.0, -0.5, 0.2, 0.1, 0.4])
    y = np.maximum(coef[0] + (har5 * coef[1:]).sum(2) + 0.01 * rng.normal(size=(a, n)), 1e-4)
    return har5, y


def test_warmup_rows_have_no_oof_prediction():
    har5, y = _synthetic()
    mask = np.ones(y.shape, bool)
    oof = O.oof_harx(har5, y, mask, 1e-8, np.full(y.shape[1], 1e-8))
    warmup = max(1, int(y.shape[0] * C.OOF_WARMUP_FRAC))
    assert np.isnan(oof[:warmup]).all()
    assert np.isfinite(oof[warmup:]).any()


def test_each_block_uses_only_strictly_earlier_anchors():
    har5, y = _synthetic()
    mask = np.ones(y.shape, bool)
    a = y.shape[0]
    nfloor = np.full(y.shape[1], 1e-8)
    oof = O.oof_harx(har5, y, mask, 1e-8, nfloor)
    warmup = max(1, int(a * C.OOF_WARMUP_FRAC))
    bounds = np.linspace(warmup, a, C.OOF_SPLITS + 1).astype(int)
    for b in range(C.OOF_SPLITS):
        s, e = bounds[b], bounds[b + 1]
        if e <= s:
            continue
        coef = O._ols_fit(har5[:s].reshape(-1, 5), y[:s].reshape(-1))       # refit on earlier anchors ONLY
        expect = O._ols_predict(coef, har5[s:e].reshape(-1, 5), 1e-8,
                                np.broadcast_to(nfloor, (e - s, y.shape[1])).reshape(-1))
        np.testing.assert_allclose(oof[s:e].reshape(-1), expect, rtol=1e-9, atol=1e-12)


def test_future_target_change_does_not_move_earlier_oof():
    har5, y = _synthetic()
    mask = np.ones(y.shape, bool)
    nfloor = np.full(y.shape[1], 1e-8)
    oof = O.oof_harx(har5, y, mask, 1e-8, nfloor)
    y2 = y.copy()
    y2[-5:] *= 5.0                                                          # perturb only the last anchors
    oof2 = O.oof_harx(har5, y2, mask, 1e-8, nfloor)
    a = y.shape[0]
    warmup = max(1, int(a * C.OOF_WARMUP_FRAC))
    bounds = np.linspace(warmup, a, C.OOF_SPLITS + 1).astype(int)
    unaffected = slice(warmup, bounds[-2])                                   # blocks fitted before the perturbed tail
    m = np.isfinite(oof[unaffected])
    np.testing.assert_allclose(oof[unaffected][m], oof2[unaffected][m], rtol=1e-9, atol=1e-12)


def test_residual_reconstruct_identity():
    base = np.array([0.01, 0.5, 2.0])
    y = np.array([0.02, 0.3, 2.0])
    z = O.residual_target(y, base)
    back = O.reconstruct(base, z, 1.0, 10.0)                                 # exp(z) large clip -> exact ratio
    np.testing.assert_allclose(back, base * (np.maximum(y, 0) + C.RESIDUAL_EPS) / (base + C.RESIDUAL_EPS), rtol=1e-9)


def test_reconstruct_alpha_zero_returns_base():
    base = np.array([0.1, 0.2, 0.3])
    zhat = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(O.reconstruct(base, zhat, 0.0, 1.0), base)


def test_block_skipped_when_too_few_earlier_fit_rows():
    har5, y = _synthetic(a=60, n=1)
    mask = np.ones(y.shape, bool)
    mask[:16] = False                       # first block's fit window has <6 valid rows -> skipped
    oof = O.oof_harx(har5, y, mask, 1e-8, np.full(1, 1e-8))
    assert np.isnan(oof[18:26]).all()        # block0 (s=18) skipped
    assert np.isfinite(oof[51:60]).any()     # a later block still fits


def test_block_skipped_when_no_valid_target_cells():
    har5, y = _synthetic(a=60, n=1)
    mask = np.ones(y.shape, bool)
    mask[34:43] = False                      # this block has no valid target cells -> skipped
    oof = O.oof_harx(har5, y, mask, 1e-8, np.full(1, 1e-8))
    assert np.isnan(oof[34:43]).all()


def test_too_few_anchors_returns_all_nan():
    har5, y = _synthetic(a=4, n=3)          # warmup=1, remaining 3 < OOF_SPLITS -> no valid OOF block
    oof = O.oof_harx(har5, y, np.ones(y.shape, bool), 1e-8, np.full(3, 1e-8))
    assert np.isnan(oof).all()
