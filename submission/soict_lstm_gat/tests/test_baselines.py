"""Tests for the self-contained classical-baselines module (written first, TDD).

Bare imports work because the submission-folder conftest.py puts the folder on sys.path.
"""
import numpy as np
import pytest

import baselines


def test_har_fit_recovers_known_coefficients():
    """OLS with intercept must recover the generating linear coefficients."""
    rng = np.random.default_rng(0)
    n = 400
    X = rng.uniform(0.1, 1.0, size=(n, 3))
    true = np.array([0.5, 2.0, -1.0, 0.7])  # [intercept, b1, b2, b3]
    y = true[0] + X @ true[1:]
    coef = baselines.har_fit(X, y)
    assert coef.shape == (4,)
    assert np.allclose(coef, true, atol=1e-6)


def test_har_predict_nonnegative_and_matches_linear():
    """Prediction equals intercept + X@coef[1:] where positive, and is floored at >= 0."""
    X = np.array([[0.2, 0.3, 0.4], [0.5, 0.5, 0.5]])
    coef = np.array([0.1, 1.0, 1.0, 1.0])  # positive region
    pred = baselines.har_predict(X, coef)
    expected = coef[0] + X @ coef[1:]
    assert np.allclose(pred, expected)
    assert np.all(pred >= 0.0)

    # A coefficient set that yields negative raw predictions must be clamped to the floor.
    neg_coef = np.array([-5.0, 0.0, 0.0, 0.0])
    floored = baselines.har_predict(X, neg_coef, floor=1e-8)
    assert np.all(floored >= 1e-8)
    assert np.all(floored > 0.0)


def test_garch_forecast_smoke():
    """Fit GARCH on a synthetic positive variance series -> n_test positive, finite forecasts."""
    rng = np.random.default_rng(7)
    n = 500
    # Build a persistent positive variance-like series (ARCH-ish magnitude).
    shocks = rng.standard_normal(n)
    var = np.empty(n)
    var[0] = 1e-4
    for t in range(1, n):
        var[t] = 5e-5 + 0.85 * var[t - 1] + 1e-5 * shocks[t] ** 2
    var = np.abs(var) + 1e-8  # strictly positive Parkinson-variance proxy

    n_test = 20
    out = baselines.garch_forecast(var, n_test)
    assert out.shape == (n_test,)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)


def test_garch_forecast_fallback_on_degenerate_series():
    """A degenerate (near-constant) series must still return positive finite forecasts."""
    series = np.full(300, 1e-4)
    out = baselines.garch_forecast(series, n_test=5)
    assert out.shape == (5,)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)
