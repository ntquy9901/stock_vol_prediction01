"""Tests for the masked-panel GARCH add-on (scripts/garch_masked/compute_garch_masked.py).

Covers the pure metric aggregator and the two prediction builders on small synthetic panels,
so the throwaway analysis script ships with executable invariants (floor respected, finite,
correct key count, exact MSE/MAE).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "garch_masked"))
import compute_garch_masked as G  # noqa: E402


def test_metrics_exact():
    pred = {(0, "d1"): (1.0, 1.0), (0, "d2"): (2.0, 4.0)}  # y=[1,2], p=[1,4]
    m = G._metrics(pred, floor=1e-8)
    assert abs(m["mse"] - 2.0) < 1e-9        # (0^2 + 2^2)/2
    assert abs(m["mae"] - 1.0) < 1e-9        # (0 + 2)/2
    assert m["n"] == 2


def _fake_D(N=2, ntr=300, nte=5, seed=0):
    rng = np.random.default_rng(seed)
    y_tr = np.abs(rng.normal(2e-4, 5e-5, (ntr, N)))
    y_te = np.abs(rng.normal(2e-4, 5e-5, (nte, N)))
    har5_tr = np.abs(rng.normal(2e-4, 5e-5, (ntr, N, 5)))
    har5_te = np.abs(rng.normal(2e-4, 5e-5, (nte, N, 5)))
    return SimpleNamespace(
        N=N, y_tr=y_tr, y_te=y_te,
        tmask_tr=np.ones((ntr, N), bool), tmask_te=np.ones((nte, N), bool),
        har5_tr=har5_tr, har5_te=har5_te,
        t_mean=y_tr.mean(0), d_te=[f"d{i}" for i in range(nte)])


def test_garch_pred_floor_finite_and_count():
    D = _fake_D()
    cfg = SimpleNamespace(qlike_floor=1e-8)
    gd = G._garch_pred(D, horizon=1, cfg=cfg)
    assert len(gd) == D.N * len(D.d_te)
    floor = 1e-2 * D.t_mean + 1e-12
    for (j, _), (yv, pv) in gd.items():
        assert np.isfinite(pv) and pv >= floor[j] - 1e-18


def test_harx_pred_shape_and_floor():
    D = _fake_D()
    cfg = SimpleNamespace(qlike_floor=1e-8)
    hd = G._harx_pred(D, cfg)
    assert len(hd) == D.N * len(D.d_te)
    floor = 1e-2 * D.t_mean + 1e-12
    for (j, _), (yv, pv) in hd.items():
        assert np.isfinite(pv) and pv >= floor[j] - 1e-18
