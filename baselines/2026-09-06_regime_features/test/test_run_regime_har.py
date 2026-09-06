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
