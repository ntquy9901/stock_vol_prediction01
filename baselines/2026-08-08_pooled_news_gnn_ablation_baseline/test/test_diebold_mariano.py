"""Unit tests for the Diebold-Mariano equal-predictive-accuracy test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from diebold_mariano import diebold_mariano  # noqa: E402


def test_hand_computed_h1_case() -> None:
    """DM stat + HLN correction match a hand-computed 4-observation, h=1 case.

    loss_a=[1,2,3,4], loss_b=[1,1,1,1] -> d=[0,1,2,3], dbar=1.5, gamma0=1.25.
    Var(dbar)=gamma0/n=0.3125; DM=1.5/sqrt(0.3125)=2.6832815729997477.
    HLN factor (h=1)=sqrt((n-1)/n)=sqrt(0.75); DM_hln=2.32379000772445.
    """
    loss_a = np.array([1.0, 2.0, 3.0, 4.0])
    loss_b = np.array([1.0, 1.0, 1.0, 1.0])
    result = diebold_mariano(loss_a, loss_b, h=1)
    assert result.n == 4
    assert result.mean_diff == pytest.approx(1.5)
    assert result.dm_stat == pytest.approx(2.6832815729997477, rel=1e-12)
    assert result.dm_hln == pytest.approx(2.32379000772445, rel=1e-12)
    assert result.p_value == pytest.approx(0.10272807885839899, rel=1e-9)


def test_negative_dm_when_model_a_more_accurate() -> None:
    """d_t = loss_a - loss_b; a smaller loss for A => negative DM statistic."""
    rng = np.random.default_rng(0)
    loss_b = rng.uniform(1.0, 2.0, size=200)
    loss_a = loss_b - 0.1  # A uniformly better
    result = diebold_mariano(loss_a, loss_b, h=1)
    assert result.dm_stat < 0
    assert result.dm_hln < 0
    assert result.p_value < 0.05


def test_matches_statsmodels_hac_bartlett_h3() -> None:
    """DM stat (pre-HLN) equals dbar / HAC-Newey-West bse from statsmodels for h=3.

    statsmodels OLS(d ~ const) with cov_type='HAC', Bartlett kernel, maxlags=h-1,
    use_correction=False gives the Newey-West long-run variance of the mean; the
    DM statistic is the corresponding t-statistic.
    """
    import statsmodels.api as sm

    rng = np.random.default_rng(42)
    d = rng.normal(0.3, 1.0, size=500)
    # Inject serial correlation so the HAC lags actually matter.
    d = d + 0.5 * np.roll(d, 1)
    d[0] = d[1]
    loss_a = d
    loss_b = np.zeros_like(d)

    h = 3
    result = diebold_mariano(loss_a, loss_b, h=h, kernel="bartlett")

    ones = np.ones((len(d), 1))
    model = sm.OLS(d, ones).fit(
        cov_type="HAC", cov_kwds={"maxlags": h - 1, "use_correction": False}
    )
    ref_dm = float(model.params[0] / model.bse[0])
    assert result.dm_stat == pytest.approx(ref_dm, rel=1e-9)


def test_rectangular_kernel_matches_classic_truncated_sum() -> None:
    """Classic DM(1995) uses an un-weighted (rectangular) truncated autocovariance sum."""
    rng = np.random.default_rng(7)
    d = rng.normal(0.0, 1.0, size=300)
    loss_a = d
    loss_b = np.zeros_like(d)
    h = 4
    n = len(d)
    dbar = d.mean()
    dev = d - dbar
    gamma = [float(np.sum(dev[k:] * dev[:n - k]) / n) for k in range(h)]
    lrv = gamma[0] + 2.0 * sum(gamma[1:])
    ref = dbar / np.sqrt(lrv / n)

    result = diebold_mariano(loss_a, loss_b, h=h, kernel="rectangular")
    assert result.dm_stat == pytest.approx(ref, rel=1e-12)


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        diebold_mariano(np.array([1.0, 2.0]), np.array([1.0]), h=1)


def test_rejects_horizon_below_one() -> None:
    with pytest.raises(ValueError):
        diebold_mariano(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0]), h=0)
