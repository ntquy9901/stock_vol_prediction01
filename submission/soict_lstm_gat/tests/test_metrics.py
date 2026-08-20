"""Tests for the self-contained metrics module (written first, TDD).

Bare imports work because the submission-folder conftest.py puts the folder on sys.path.
"""
import numpy as np
import pytest

import metrics


def test_rmse_is_sqrt_of_mse():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.5, 1.0, 3.5, 3.0])
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt(metrics.mse(y, p)))


def test_mse_value():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 4.0, 3.0])
    # errors: 0, -2, 0 -> squared 0,4,0 -> mean 4/3
    assert metrics.mse(y, p) == pytest.approx(4.0 / 3.0)


def test_mae_value():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.5, 1.0, 3.5, 2.0])
    # abs errors: 0.5, 1.0, 0.5, 2.0 -> mean 1.0
    assert metrics.mae(y, p) == pytest.approx(1.0)


def test_r2_of_perfect_is_one():
    y = np.array([0.1, 0.2, 0.3, 0.5, 0.9])
    assert metrics.r2(y, y.copy()) == pytest.approx(1.0)


def test_r2_of_mean_predictor_is_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.full_like(y, y.mean())
    assert metrics.r2(y, p) == pytest.approx(0.0)


def test_per_obs_se():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 4.0, 6.0])
    np.testing.assert_allclose(metrics.per_obs_se(y, p), np.array([0.0, 4.0, 9.0]))


def test_qlike_perfect_is_zero():
    y = np.array([0.01, 0.02, 0.03, 0.5])
    per = metrics.per_obs_qlike(y, y.copy())
    np.testing.assert_allclose(per, np.zeros_like(y), atol=1e-12)
    assert metrics.qlike(y, y.copy()) == pytest.approx(0.0, abs=1e-12)


def test_qlike_mean_matches_per_obs_mean():
    y = np.array([0.02, 0.05, 0.1, 0.2])
    p = np.array([0.03, 0.04, 0.15, 0.1])
    assert metrics.qlike(y, p) == pytest.approx(metrics.per_obs_qlike(y, p).mean())


def test_qlike_shared_floor_keeps_finite_on_tiny_target():
    # Target contains a zero and negatives; prediction near zero. Shared floor (1e-8)
    # clamps both -> must stay finite (no log(0)/divide-by-zero).
    y = np.array([0.0, -1e-20, 1e-15, 0.05])
    p = np.array([1e-20, 0.0, 1e-12, 0.05])
    per = metrics.per_obs_qlike(y, p)
    assert per.shape == y.shape
    assert np.all(np.isfinite(per))
    assert np.isfinite(metrics.qlike(y, p))


def test_qlike_matches_reference_formula():
    # r - log(r) - 1 with shared floor, computed independently.
    floor = 1e-8
    y = np.array([0.0, 0.02, 0.1])
    p = np.array([0.01, 0.0, 0.05])
    yc = np.maximum(y, floor)
    pc = np.maximum(p, floor)
    r = yc / pc
    expected = r - np.log(r) - 1.0
    np.testing.assert_allclose(metrics.per_obs_qlike(y, p), expected)


def test_dm_sign_a_lower_loss_is_negative():
    rng = np.random.default_rng(0)
    n = 200
    base = rng.random(n)
    loss_a = base
    loss_b = base + 0.5 + rng.random(n) * 0.1  # b clearly larger -> A more accurate
    res = metrics.diebold_mariano(loss_a, loss_b, h=1)
    assert res.n == n
    assert res.mean_diff < 0  # d = a - b < 0
    assert res.dm_hln < 0  # negative favors A
    assert res.p_value < 0.05


def test_dm_h5_uses_hac_and_is_finite():
    rng = np.random.default_rng(1)
    n = 150
    loss_a = rng.random(n)
    loss_b = loss_a + 0.3 + rng.random(n) * 0.2
    res = metrics.diebold_mariano(loss_a, loss_b, h=5)
    assert res.n == n
    assert np.isfinite(res.dm_hln)
    assert np.isfinite(res.p_value)
    assert 0.0 <= res.p_value <= 1.0
    # HAC lag h-1 changes the long-run variance vs h=1 -> different statistic.
    res1 = metrics.diebold_mariano(loss_a, loss_b, h=1)
    assert res.dm_hln != pytest.approx(res1.dm_hln)


def test_dm_raises_on_too_few_observations():
    with pytest.raises(ValueError):
        metrics.diebold_mariano(np.array([1.0]), np.array([2.0]), h=5)


def test_dm_raises_on_non_finite_losses():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([0.5, 1.5, 2.5, 3.5])
    with pytest.raises(ValueError):
        metrics.diebold_mariano(a, b, h=1)
