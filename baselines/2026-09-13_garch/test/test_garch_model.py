"""Formula-exact + units + causality unit tests for the GARCH conditional-variance core.

The multi-step and reversion tests recompute the expected value by an INDEPENDENT iteration of the
expectation recursion (never reusing the implementation), per the named-estimator rule in CLAUDE.md.
"""
import numpy as np
import pytest

import config
import garch_model as G


def _params(omega, alpha, beta, gamma=0.0, mu=0.0, fallback_var=0.0):
    return G.Params(mu=mu, omega=omega, alpha=alpha, gamma=gamma, beta=beta, ok=True,
                    fallback_var=fallback_var)


def _iterate(omega, phi, s_next, h):
    """Independent reference: v_1 = s_next; v_{k+1} = omega + phi*v_k; return v_h."""
    v = s_next
    for _ in range(h - 1):
        v = omega + phi * v
    return v


@pytest.mark.parametrize("h", [1, 2, 5, 10, 22])
def test_multistep_matches_iterated_recursion_garch(h):
    p = _params(omega=0.02, alpha=0.08, beta=0.90)          # phi = 0.98
    s_next = 1.7
    got = float(G.multistep(np.array([s_next]), p, "garch", h)[0])
    assert got == pytest.approx(_iterate(0.02, 0.98, s_next, h), rel=1e-10)


@pytest.mark.parametrize("h", [1, 3, 10, 22])
def test_multistep_matches_iterated_recursion_gjr(h):
    p = _params(omega=0.02, alpha=0.05, beta=0.88, gamma=0.06)   # phi = 0.05+0.88+0.03 = 0.96
    phi = 0.05 + 0.88 + 0.06 / 2.0
    s_next = 2.1
    got = float(G.multistep(np.array([s_next]), p, "gjr", h)[0])
    assert got == pytest.approx(_iterate(0.02, phi, s_next, h), rel=1e-10)


def test_multistep_reverts_to_unconditional_variance():
    p = _params(omega=0.02, alpha=0.08, beta=0.90)          # phi = 0.98
    uncond = 0.02 / (1.0 - 0.98)
    far = float(G.multistep(np.array([5.0]), p, "garch", 2000)[0])
    assert far == pytest.approx(uncond, rel=1e-6)


def test_reversion_persistence():
    p = _params(omega=0.01, alpha=0.1, beta=0.8, gamma=0.04)
    assert G.reversion_persistence(p, "garch") == pytest.approx(0.9)
    assert G.reversion_persistence(p, "gjr") == pytest.approx(0.1 + 0.8 + 0.02)


def test_fallback_returns_sample_variance_in_original_scale():
    # alternating +/- a -> sample variance (ddof=1) ~ a**2; must NOT be x SCALE**2 or / SCALE**2.
    a = 0.013
    r = np.array([a, -a] * 40, dtype=float)                 # 80 obs < MIN_TRAIN_OBS -> forced fallback
    v = float(np.var(r, ddof=1))
    f = G.ticker_forecast(r, n_train=len(r), test_idx=np.array([10, 20]), h=5, variant="garch")
    assert np.allclose(f, v)
    assert v == pytest.approx(a * a, rel=5e-2)               # sanity: variance is O(a**2), original scale


def test_fitted_iid_series_recovers_variance_scale():
    rng = np.random.default_rng(7)
    sigma = 0.02
    r = rng.standard_normal(1200) * sigma                   # iid -> alpha,beta ~ 0, uncond ~ sigma**2
    v = sigma * sigma
    f = G.ticker_forecast(r, n_train=1000, test_idx=np.array([1050]), h=1, variant="garch")
    # generous band around the true variance: proves the /SCALE**2 rescale is applied (not off by 1e4)
    assert 0.3 * v < float(f[0]) < 3.0 * v


def test_forecast_is_causal_future_returns_do_not_leak():
    rng = np.random.default_rng(3)
    r = rng.standard_normal(400) * 0.02
    f1 = G.ticker_forecast(r, n_train=300, test_idx=np.array([310]), h=1, variant="gjr")
    r2 = r.copy(); r2[350] += 0.5                           # perturb a FUTURE return (index 350 > 310)
    f2 = G.ticker_forecast(r2, n_train=300, test_idx=np.array([310]), h=1, variant="gjr")
    assert float(f1[0]) == float(f2[0])


def test_fit_params_degenerate_short_series_flags_fallback():
    p = G.fit_params(np.array([0.01, -0.01, 0.005]), "garch")   # < MIN_TRAIN_OBS
    assert p.ok is False
    assert p.fallback_var >= 0.0


def test_fit_params_falls_back_when_arch_raises(monkeypatch):
    import arch
    monkeypatch.setattr(arch, "arch_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    p = G.fit_params(np.linspace(-0.02, 0.02, 300), "garch")
    assert p.ok is False


def test_fit_params_falls_back_when_params_degenerate(monkeypatch):
    import arch
    import pandas as pd

    class _Res:
        params = pd.Series({"mu": 0.0, "omega": -1.0, "alpha[1]": 0.1, "beta[1]": 0.5})  # omega<=0 -> degenerate

    class _Model:
        def fit(self, **k):
            return _Res()

    monkeypatch.setattr(arch, "arch_model", lambda *a, **k: _Model())
    p = G.fit_params(np.linspace(-0.02, 0.02, 300), "garch")
    assert p.ok is False


def test_config_thresholds_are_sane():
    assert config.SCALE == 100.0
    assert 0.0 <= config.PERSIST_LO < config.PERSIST_HI <= 1.0
