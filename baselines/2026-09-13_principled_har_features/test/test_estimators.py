"""Formula-exact + causality tests for realized semivariance."""
import numpy as np

import estimators as E


def test_semivariance_splits_by_sign_matches_formula():
    ret = np.array([0.0, -0.02, 0.01, -0.03, 0.0])
    rm, rp = E.semivariance(ret, window=3, min_periods=1)
    # at t=3, window {t-2,t-1,t} = idx 1,2,3 -> returns -0.02, 0.01, -0.03
    assert np.isclose(rm[3], np.mean([0.02 ** 2, 0.0, 0.03 ** 2]))       # negatives only
    assert np.isclose(rp[3], np.mean([0.0, 0.01 ** 2, 0.0]))            # positives only
    assert np.isclose(rm[3] + rp[3], np.mean([0.02 ** 2, 0.01 ** 2, 0.03 ** 2]))  # sum = mean squared


def test_semivariance_min_periods_nan_then_defined():
    ret = np.array([-0.01, 0.02, -0.03, 0.04])
    rm, rp = E.semivariance(ret, window=3, min_periods=3)
    assert np.isnan(rm[0]) and np.isnan(rm[1])      # < min_periods -> NaN
    assert np.isfinite(rm[2]) and np.isfinite(rp[2])


def test_semivariance_nan_return_excluded_not_counted_as_zero():
    # a leading NaN return (e.g. first day) must be EXCLUDED from the rolling mean, not treated as a 0.
    ret = np.array([np.nan, -0.02, -0.04])
    rm, rp = E.semivariance(ret, window=3, min_periods=2)
    # at t=2, window {nan,-0.02,-0.04}: mean over the 2 valid negatives, NaN excluded
    assert np.isclose(rm[2], np.mean([0.02 ** 2, 0.04 ** 2]))
    assert np.isclose(rp[2], 0.0)


def test_semivariance_is_causal_trailing():
    ret = np.array([-0.01, 0.02, -0.03, 0.04, -0.05, 0.06], float)
    rm0, rp0 = E.semivariance(ret, window=3, min_periods=1)
    perturbed = ret.copy(); perturbed[4:] *= 5      # change the FUTURE (idx >= 4)
    rm1, rp1 = E.semivariance(perturbed, window=3, min_periods=1)
    assert np.allclose(rm0[:4], rm1[:4]) and np.allclose(rp0[:4], rp1[:4])   # t<4 unchanged
