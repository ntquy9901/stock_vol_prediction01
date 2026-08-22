"""TDD: E3/E4 frozen-expert alpha blend + E9 static gate numerics (plan sections 8-9, 12)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import metrics as M  # noqa: E402
import blend  # noqa: E402
import gate  # noqa: E402


def _dicts(y, ha, nn):
    da = {(0, str(i)): (float(y[i]), float(ha[i])) for i in range(len(y))}
    db = {(0, str(i)): (float(y[i]), float(nn[i])) for i in range(len(y))}
    return da, db


def test_alpha_in_range_and_reduces_val_mse():
    rng = np.random.default_rng(1)
    y = np.abs(rng.normal(0.02, 0.005, 400)) + 1e-3
    har = y + rng.normal(0, 0.004, 400)
    nn = y + rng.normal(0, 0.006, 400)
    a = blend.fit_alpha_mse(y, har, nn)
    assert 0.0 <= a <= 1.0
    blended = a * har + (1 - a) * nn
    assert np.mean((y - blended) ** 2) <= min(np.mean((y - har) ** 2), np.mean((y - nn) ** 2)) + 1e-9


def test_qlike_alpha_grid_beats_or_ties_experts():
    rng = np.random.default_rng(2)
    y = np.abs(rng.normal(0.02, 0.005, 400)) + 1e-3
    har = np.maximum(y + rng.normal(0, 0.003, 400), 1e-6)
    nn = np.maximum(y + rng.normal(0, 0.007, 400), 1e-6)
    a = blend.fit_alpha_qlike(y, har, nn, floor=1e-8)
    assert 0.0 <= a <= 1.0
    p = np.maximum(a * har + (1 - a) * nn, 1e-8)
    assert M.qlike(y, p, 1e-8) <= min(M.qlike(y, har, 1e-8), M.qlike(y, nn, 1e-8)) + 1e-9


def test_blend_apply_on_test():
    y = np.array([0.02, 0.03, 0.01]); ha = np.array([0.019, 0.031, 0.011]); nn = np.array([0.03, 0.02, 0.02])
    da, db = _dicts(y, ha, nn)
    alpha, out = blend.blend(da, db, da, db, loss="mse", floor=1e-8)
    assert 0.0 <= alpha <= 1.0 and len(out) == 3


def test_static_gate_collapses_to_har_on_noise():
    rng = np.random.default_rng(3)
    y = np.abs(rng.normal(0.02, 0.005, (200, 5))) + 1e-3
    harp = y + rng.normal(0, 0.002, (200, 5))
    c = rng.normal(0, 1, (200, 5))                     # pure-noise correction, uncorrelated with residual
    scale = np.full(5, 0.01)
    lam = gate.fit_lambda_static(y, harp, c, scale, "additive", eps=1e-8, floor=1e-8)
    assert lam <= 0.2                                   # near-zero: gate should not trust noise


def test_static_gate_uses_helpful_correction():
    rng = np.random.default_rng(4)
    y = np.abs(rng.normal(0.02, 0.005, (200, 5))) + 1e-3
    harp = np.full((200, 5), 0.02)
    scale = np.full(5, 1.0)
    c = (y - harp) / scale                              # correction exactly points to the residual
    lam = gate.fit_lambda_static(y, harp, c, scale, "additive", eps=1e-8, floor=1e-8)
    assert lam > 0.5                                    # gate should apply a strong correction


def test_reconstruct_positivity_multiplicative():
    harp = np.array([[0.02, 0.0]]); c = np.array([[-100.0, 5.0]]); scale = np.array([1.0, 1.0])
    p = gate.reconstruct("mult", harp, c, scale, eps=1e-8, lam=1.0)
    assert np.all(p >= 0)


def test_gate_reconstruct_additive_floor():
    # a large negative additive correction is floored at pred_floor, not clipped to ~0
    p = gate.reconstruct("additive", np.array([[0.02]]), np.array([[-1000.0]]),
                         np.array([1.0]), eps=1e-8, lam=1.0, pred_floor=np.array([0.01]))
    assert np.all(p >= 0.01)
