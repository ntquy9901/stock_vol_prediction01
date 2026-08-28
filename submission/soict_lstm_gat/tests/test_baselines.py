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


def test_garch_forecast_finite_on_nonfinite_series():
    """External review M-01: a NaN/inf-contaminated (and short) series routes to the fallback,
    which must still honour the finite/positive/>=floor guarantee (was returning NaN)."""
    series = np.array([np.nan, np.inf, 1e-4, np.nan])
    out = baselines.garch_forecast(series, n_test=6, floor=1e-8)
    assert out.shape == (6,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 1e-8)


def test_garch_forecast_fallback_all_nonfinite_returns_floor():
    """External review M-01: if nothing finite remains, fall back to the floor, not NaN."""
    series = np.array([np.nan, np.inf, -np.inf])
    out = baselines.garch_forecast(series, n_test=3, floor=1e-6)
    assert np.allclose(out, 1e-6)


def test_garch_forecast_rejects_nonpositive_n_test():
    with pytest.raises(ValueError):
        baselines.garch_forecast(np.array([1e-4, 2e-4]), n_test=0)


def test_garch_fallback_rejects_invalid_floor():
    # short series -> fallback path; an invalid (non-positive) floor must raise, not emit a bad forecast
    with pytest.raises(ValueError):
        baselines.garch_forecast(np.array([1e-4]), n_test=3, floor=0.0)


def test_garch_forecast_fallback_overflow_safe():
    """External review R-02: extreme finite values whose mean overflows to +inf must fall back to the
    floor, keeping the finite/positive/>=floor guarantee."""
    # nan forces the fallback; the 1e308 entries make the mean overflow to +inf -> must clamp to floor
    out = baselines.garch_forecast(np.array([1e308, 1e308, 1e308, np.nan]), n_test=4, floor=1e-8)
    assert out.shape == (4,) and np.all(np.isfinite(out)) and np.all(out >= 1e-8)
    out2, st = baselines.garch_forecast(np.array([1e308, 1e308, 1e308, np.inf]), n_test=2,
                                        floor=1e-6, return_status=True)
    assert np.all(np.isfinite(out2)) and np.all(out2 >= 1e-6) and st["fallback"] is True


def test_garch_forecast_guards_nonfinite_forecast_path(monkeypatch):
    """Coverage/defensive: if the fitted path yields a non-finite/non-positive forecast, fall back (finite)."""
    rng = np.random.default_rng(0)
    series = np.abs(rng.standard_normal(400)) * 1e-3 + 1e-5
    monkeypatch.setattr(baselines, "_capped_forecast_path",
                        lambda *a, **k: np.full(a[6] if len(a) > 6 else k["total_steps"], np.inf))
    out, st = baselines.garch_forecast(series, n_test=5, return_status=True)
    assert np.all(np.isfinite(out)) and np.all(out > 0) and st["fallback"] is True


def test_garch_status_records_nonpositive_count():
    """External review F-04: zero/negative finite variance entries are floored (legit H~L sanitization),
    fallback stays False, but the count is recorded in the status for input-quality audit."""
    series = np.array([1e-4, 0.0, 2e-4, -1e-6, 3e-4] * 80, dtype=float)   # 2 nonpositive per block
    _, st = baselines.garch_forecast(series, n_test=5, return_status=True)
    assert "nonpositive_count" in st and st["nonpositive_count"] == 160     # 2 * 80 blocks
    _, st_clean = baselines.garch_forecast(np.abs(np.random.default_rng(1).standard_normal(300)) * 1e-3 + 1e-5,
                                           n_test=5, return_status=True)
    assert st_clean["nonpositive_count"] == 0


def test_garch_forecast_return_status_flags_fallback():
    """External review M-08: return_status surfaces GARCH degradation. A degenerate short series
    falls back (fallback=True + reason); a normal series fits (fallback=False)."""
    fc_bad, st_bad = baselines.garch_forecast(np.array([np.nan, 1e-4]), n_test=3, return_status=True)
    assert st_bad["fallback"] is True and st_bad["reason"]
    assert fc_bad.shape == (3,) and np.all(np.isfinite(fc_bad))
    rng = np.random.default_rng(0)
    series = np.abs(rng.standard_normal(400)) * 1e-3 + 1e-5
    fc_ok, st_ok = baselines.garch_forecast(series, n_test=10, return_status=True)
    assert "fallback" in st_ok and "arch_available" in st_ok
    assert fc_ok.shape == (10,)
    # default (no status) still returns a bare array (back-compat)
    assert isinstance(baselines.garch_forecast(series, n_test=5), np.ndarray)


def test_cap_params_reduces_persistence_via_variance_targeting():
    """When alpha+beta exceeds the cap, persistence is scaled to the cap (ratio preserved) and
    omega is re-set by variance targeting so the long-run variance equals the sample target."""
    cap = 0.999
    omega_c, alpha_c, beta_c = baselines._cap_params(
        omega=0.01, alpha=0.3, beta=0.9, uncond_target=5.0, cap=cap)
    assert abs((alpha_c + beta_c) - cap) < 1e-12          # capped exactly to the ceiling
    assert abs((alpha_c / beta_c) - (0.3 / 0.9)) < 1e-12  # alpha:beta ratio preserved
    assert abs(omega_c - 5.0 * (1.0 - cap)) < 1e-12       # variance targeting: uncond == target


def test_cap_params_noop_when_stationary():
    """A stationary fit (alpha+beta < cap) is returned unchanged."""
    omega_c, alpha_c, beta_c = baselines._cap_params(
        omega=0.02, alpha=0.1, beta=0.7, uncond_target=5.0, cap=0.999)
    assert (omega_c, alpha_c, beta_c) == (0.02, 0.1, 0.7)


def test_capped_forecast_path_matches_recursion_when_stationary():
    """For a stationary GARCH(1,1) the analytic path equals the direct one-step recursion."""
    omega, alpha, beta = 0.02, 0.1, 0.7
    last_h, last_eps2 = 5.0, 4.0
    path = baselines._capped_forecast_path(
        omega, alpha, beta, last_h, last_eps2, uncond_target=5.0, total_steps=6, cap=0.999)
    # direct recursion: sig2_{k} = omega + (alpha+beta) * sig2_{k-1}, sig2_1 from last state
    sig = omega + alpha * last_eps2 + beta * last_h
    manual = []
    for _ in range(6):
        manual.append(sig)
        sig = omega + (alpha + beta) * sig
    assert np.allclose(path, manual)


def test_capped_forecast_path_bounded_for_igarch():
    """An IGARCH fit (alpha+beta >= 1) must NOT diverge: the capped path stays finite, bounded,
    and converges toward the finite capped unconditional variance instead of growing linearly."""
    cap = 0.999
    omega, alpha, beta = 0.5, 0.2, 0.85   # persistence 1.05 -> IGARCH, would diverge uncapped
    uncond_target = 3.0
    total_steps = 2000
    path = baselines._capped_forecast_path(
        omega, alpha, beta, last_h=2.0, last_eps2=1.5,
        uncond_target=uncond_target, total_steps=total_steps, cap=cap)
    assert path.shape == (total_steps,)
    assert np.all(np.isfinite(path))
    uncond_capped = uncond_target                     # variance targeting sets long-run == target
    # converging toward the finite capped unconditional variance, not diverging away from it
    assert abs(path[-1] - uncond_capped) < abs(path[0] - uncond_capped)
    assert path.max() <= 5.0 * uncond_capped          # bounded (uncapped would reach ~100x+)
