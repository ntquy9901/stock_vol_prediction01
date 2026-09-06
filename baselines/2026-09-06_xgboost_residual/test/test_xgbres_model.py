"""XGBoost tuning: reproducibility (fixed seed), guardrail grid membership, val-QLIKE selection."""
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))

import xgb_config as C  # noqa: E402
import xgb_model as XM  # noqa: E402


def _data(seed=0, m=400, k=6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(m, k))
    y = np.maximum(0.02 + 0.01 * x[:, 0] - 0.005 * x[:, 1] + 0.002 * rng.normal(size=m), 1e-4)
    return x, y


def test_direct_reproducible_same_seed():
    xtr, ytr = _data(0)
    xva, yva = _data(1, m=150)
    nf_tr = np.full(len(ytr), 1e-8)
    nf_va = np.full(len(yva), 1e-8)
    b1 = XM.tune_direct(xtr, ytr, xva, yva, nf_tr, nf_va, 1e-8)
    b2 = XM.tune_direct(xtr, ytr, xva, yva, nf_tr, nf_va, 1e-8)
    np.testing.assert_allclose(b1["model"].predict(xva), b2["model"].predict(xva), rtol=1e-12)
    assert b1["params"] in C.XGB_GRID


def test_residual_guardrail_in_grid_and_reproducible():
    xtr, ytr = _data(2)
    xva, yva = _data(3, m=150)
    base_tr = np.full(len(ytr), 0.02)
    base_va = np.full(len(yva), 0.02)
    z_tr = np.log((ytr + 1e-8) / (base_tr + 1e-8))
    z_va = np.log((yva + 1e-8) / (base_va + 1e-8))
    nf_va = np.full(len(yva), 1e-8)
    b1 = XM.tune_residual(xtr, z_tr, xva, z_va, base_va, yva, nf_va, 1e-8)
    b2 = XM.tune_residual(xtr, z_tr, xva, z_va, base_va, yva, nf_va, 1e-8)
    assert b1["alpha"] in C.ALPHA_GRID and b1["clip"] in C.CLIP_GRID
    assert b1["params"] in C.XGB_GRID
    assert abs(b1["val_qlike"] - b2["val_qlike"]) < 1e-12


def test_residual_alpha_zero_when_residual_is_pure_noise():
    # z target unrelated to features -> the correction cannot help -> val should prefer alpha=0
    rng = np.random.default_rng(7)
    xtr = rng.normal(size=(500, 6))
    xva = rng.normal(size=(200, 6))
    base_tr = np.full(500, 0.02)
    base_va = np.full(200, 0.02)
    ytr = np.maximum(base_tr * np.exp(0.3 * rng.normal(size=500)), 1e-6)   # noise residual, no signal in x
    yva = np.maximum(base_va * np.exp(0.3 * rng.normal(size=200)), 1e-6)
    z_tr = np.log((ytr + 1e-8) / (base_tr + 1e-8))
    z_va = np.log((yva + 1e-8) / (base_va + 1e-8))
    nf_va = np.full(200, 1e-8)
    b = XM.tune_residual(xtr, z_tr, xva, z_va, base_va, yva, nf_va, 1e-8)
    assert b["alpha"] == 0.0
