"""Plan sections 8/12/15/16/17/41/65: correlation, lead-lag, FDR, null, market adj."""

import numpy as np
import pandas as pd

from graph_eda import relationships as rel
from graph_eda.leakage import assert_corr_matrix_valid


def _synthetic_leadlag(n=400, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    # b follows a with a 1-day lag: b(t) = a(t-1) + noise  => a leads b
    b = np.concatenate([[0.0], a[:-1]]) + 0.1 * rng.standard_normal(n)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_pairwise_corr_symmetric_unit_diag():
    w = _synthetic_leadlag()
    c = rel.pairwise_corr(w)
    assert_corr_matrix_valid(c)


def test_benjamini_hochberg_monotone_and_bounds():
    p = np.array([0.001, 0.01, 0.2, 0.5, np.nan])
    q = rel.benjamini_hochberg(p)
    assert np.nanmax(q) <= 1.0
    assert np.isnan(q[-1])
    # q >= p elementwise for BH
    assert np.all(q[:4] >= p[:4] - 1e-12)


def test_lead_lag_captures_direction_and_asymmetry():
    w = _synthetic_leadlag()
    lead1 = rel.lead_lag_matrix(w, k=1)
    # A(t) predicts B(t+1) strongly; reverse is weak -> positive asymmetry A->B
    assert lead1.loc["A", "B"] > 0.7
    assert abs(lead1.loc["B", "A"]) < 0.3
    asym = rel.directional_asymmetry(lead1)
    assert asym.loc["A", "B"] > 0.4
    assert asym.loc["A", "B"] == -asym.loc["B", "A"]


def test_permutation_null_detects_real_dependence():
    rng = np.random.default_rng(0)
    common = rng.standard_normal(300)
    idx = pd.date_range("2021-01-01", periods=300, freq="B")
    w = pd.DataFrame(
        {f"S{i}": common + 0.5 * rng.standard_normal(300) for i in range(5)}, index=idx
    )
    res = rel.permutation_null_mean_abs_corr(w, n_iter=200, seed=0)
    assert res["observed_mean_abs_corr"] > res["null_p95"]
    assert res["p_value_empirical"] < 0.05


def test_market_adjust_reduces_common_factor_corr():
    rng = np.random.default_rng(3)
    n = 300
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    # a latent common factor drives a whole cross-section (10 stocks) plus idiosyncratic
    # noise; X and Y share only that common factor -> raw corr high, residual corr ~0.
    factor = np.abs(rng.standard_normal(n)) + 0.5
    data = {f"S{i}": 1.0 * factor + 0.3 * rng.standard_normal(n) for i in range(10)}
    data["X"] = 1.0 * factor + 0.3 * rng.standard_normal(n)
    data["Y"] = 1.0 * factor + 0.3 * rng.standard_normal(n)
    pk = pd.DataFrame(data, index=idx)
    market = pk.median(axis=1)  # broad factor, not a function of the X,Y pair alone
    train_mask = np.arange(n) < 200
    raw = pk.corr().loc["X", "Y"]
    resid = rel.market_adjust_residuals(pk, market, train_mask)
    adj = resid.corr().loc["X", "Y"]
    assert raw > 0.6
    assert abs(adj) < raw - 0.3  # market removal materially shrinks pairwise correlation
