"""Unit tests for the regime-HAR OLS experiment's pure helpers."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
# the driver imports delivered modules at import time; test only the pure helpers via direct import guard
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("_rrh_helpers", HERE.parent / "code" / "run_regime_har.py")


def _load():
    # importing run_regime_har pulls in delivered baselines; skip if unavailable, but they are on this repo
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ols_recovers_linear_and_floors():
    m = _load()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 2))
    y = 3.0 + 2.0 * x[:, 0] - 1.0 * x[:, 1]
    pred = m.ols_fit_predict(x, y, x, floor=-1e9)
    assert np.allclose(pred, y, atol=1e-6)
    # floor clamps
    pred2 = m.ols_fit_predict(x, y * 0 - 5.0, x, floor=0.0)
    assert (pred2 >= 0.0).all()


def test_design_shapes_with_and_without_regime():
    m = _load()
    har5 = np.zeros((4, 3, 5)); reg = np.zeros((4, 3))
    assert m._design(har5, reg, False).shape == (12, 5)
    assert m._design(har5, reg, True).shape == (12, 8)   # 5 HAR-X + 3 regime


def test_pooled_maps_masked_entries():
    m = _load()
    from types import SimpleNamespace
    D = SimpleNamespace(
        y_te=np.array([[1.0, 2.0], [3.0, 4.0]]),
        tmask_te=np.array([[True, False], [True, True]]),
        d_te=np.array(["2026-01-01", "2026-01-02"]),
        N=2,
    )
    pred_flat = np.array([10.0, 20.0, 30.0, 40.0])   # reshaped to y_te.shape
    pooled = m._pooled(pred_flat, D, "te")
    # only masked (True) cells kept: (0,d0),(0,d1),(1,d1) -> not (1,d0)
    assert set(pooled) == {(0, "2026-01-01"), (0, "2026-01-02"), (1, "2026-01-02")}
    assert pooled[(0, "2026-01-01")] == (1.0, 10.0)
    assert pooled[(1, "2026-01-02")] == (4.0, 40.0)
