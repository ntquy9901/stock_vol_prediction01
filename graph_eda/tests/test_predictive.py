"""Plan section 33 / Gates 6-7: OOS neighbour-predictive test behaviour."""

import numpy as np
import pandas as pd

from graph_eda import predictive


def _panel(n=400, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    # driver whose PAST informs a dependent stock's future volatility
    drv = np.abs(rng.standard_normal(n)) + 0.5
    dep = np.empty(n)
    dep[0] = 0.5
    for t in range(1, n):
        dep[t] = 0.3 * dep[t - 1] + 0.6 * drv[t - 1] + 0.05 * rng.standard_normal()
    dep = np.abs(dep)
    pk = pd.DataFrame({"DRV": drv, "DEP": dep}, index=idx)
    return pk


def test_neighbor_features_help_when_dependence_is_real():
    pk = _panel()
    n = len(pk)
    train_mask = np.arange(n) < 280
    test_mask = np.arange(n) >= 340
    train_corr = pk[pd.Series(train_mask, index=pk.index)].corr().abs()
    res = predictive.run_baselines(
        pk,
        market=pk.median(axis=1),
        train_mask=train_mask,
        test_mask=test_mask,
        train_corr=train_corr,
        horizon=1,
        k=1,
    )
    assert {"A_own", "B_own+market", "C_own+neighbors"} <= set(res.index)
    # with genuine cross-stock lead structure, neighbours reduce OOS RMSE vs own-only
    assert res.loc["C_own+neighbors", "rmse"] <= res.loc["A_own", "rmse"] + 1e-9


def test_run_baselines_shared_mask_returns_per_stock():
    pk = _panel()
    n = len(pk)
    train_mask = np.arange(n) < 280
    test_mask = np.arange(n) >= 340
    train_corr = pk[pd.Series(train_mask, index=pk.index)].corr().abs()
    res, ps = predictive.run_baselines(
        pk, market=pk.median(axis=1), train_mask=train_mask, test_mask=test_mask,
        train_corr=train_corr, horizon=1, k=1, return_per_stock=True,
    )
    # all baselines evaluated on identical pooled sample size (shared valid mask, M1)
    assert res.loc["A_own", "n"] == res.loc["B_own+market", "n"] == res.loc["C_own+neighbors", "n"]
    assert {"ticker", "rmse_A", "rmse_B", "rmse_C"} <= set(ps.columns)


def test_metrics_keys_present_and_qlike_ignores_nonpositive():
    m = predictive._metrics(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.2]))
    assert set(m) == {"rmse", "r2", "qlike", "dir_acc", "n"}
    # a zero-variance target row must not blow up QLIKE (it is excluded)
    m2 = predictive._metrics(np.array([0.0, 2.0, 3.0]), np.array([1e-9, 1.9, 3.2]))
    assert np.isfinite(m2["qlike"])


def test_metrics_empty_per_stock_diracc_is_nan():
    m = predictive._metrics(
        np.array([1.0, 2.0]), np.array([1.1, 1.9]), per_stock_diracc=[]
    )
    assert np.isnan(m["dir_acc"])


def test_diracc_single_point_nan():
    assert np.isnan(predictive._diracc(np.array([1.0]), np.array([1.0])))


def test_metrics_all_nonpositive_targets_qlike_nan():
    m = predictive._metrics(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    assert np.isnan(m["qlike"])
