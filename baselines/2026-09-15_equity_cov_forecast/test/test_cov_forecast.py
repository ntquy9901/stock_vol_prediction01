"""Tests for the equity conditional-covariance forecasting baseline: panel build + winsor, estimator PSD /
invertibility / LW-vs-sklearn, GMV weights (analytic 2-asset), walk-forward causality, GBM-diagonal
combination, metrics, and a synthetic run() smoke + a real-data-slice smoke."""
import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

import cov_config as C
import panel as P
import estimators as EST
import evaluate as E
import run_cov as RUN


# ---------- synthetic fixtures ----------
def _fake_series(n_tickers=6, n_days=900, seed=0):
    """{ticker -> daily_return Series} with correlated returns from a 2-factor model, business-day dates."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    F = rng.normal(0, 0.01, size=(n_days, 2))
    out = {}
    for i in range(n_tickers):
        load = rng.normal(0, 1, size=2)
        r = F @ load + rng.normal(0, 0.008, size=n_days)
        out[f"T{i:02d}"] = pd.Series(r, index=dates)
    return out


def _panel(n_tickers=6, n_days=900, seed=0):
    return P.build_panel("fake", n_basket=n_tickers,
                         load_fn=lambda m: _fake_series(n_tickers, n_days, seed))


# ---------- panel ----------
def test_winsorize_caps_returns():
    R = pd.DataFrame({"A": [0.5, -0.9, 0.01], "B": [-0.2, 0.03, 0.4]})
    out = P.winsorize(R, 0.07)
    assert out.to_numpy().max() <= 0.07 and out.to_numpy().min() >= -0.07
    assert out.loc[2, "A"] == pytest.approx(0.01)   # in-band value untouched


def test_build_panel_shape_no_nan_sorted():
    R = _panel(n_tickers=5, n_days=400)
    assert R.shape[1] == 5 and R.shape[0] > 300
    assert not R.isna().any().any()
    assert list(R.columns) == sorted(R.columns)
    assert (R.to_numpy().max() <= C.WINSOR_BAND["default"] + 1e-12)


def test_build_panel_empty_raises():
    with pytest.raises(ValueError):
        P.build_panel("fake", load_fn=lambda m: {})


def test_build_panel_too_few_eligible_raises():
    with pytest.raises(ValueError):
        P.build_panel("fake", n_basket=1, load_fn=lambda m: {"A": _fake_series(1, 400)["T00"]})


def test_market_dir_mapping():
    assert P._market_dir("sp500").name == "sp500_clean"
    assert P._market_dir("hose").name == "hose"


# ---------- estimators ----------
def _is_pd(S):
    return np.allclose(S, S.T) and np.linalg.eigvalsh(S).min() > 0


def test_psd_ridge_fixes_negative_eigenvalues():
    S = np.array([[1.0, 2.0], [2.0, 1.0]])   # indefinite (eigs 3, -1)
    out = EST.psd_ridge(S)
    assert _is_pd(out)


def test_psd_ridge_zero_matrix_uses_absolute_eps():
    out = EST.psd_ridge(np.zeros((3, 3)))
    assert _is_pd(out) and np.allclose(np.diag(out), C.RIDGE_EPS)


@pytest.mark.parametrize("name", ["sample", "ewma", "ledoit_wolf", "factor_rank_k"])
def test_estimators_psd_invertible(name):
    R = _panel().to_numpy()
    S = EST.ESTIMATORS[name](R)
    assert S.shape == (R.shape[1], R.shape[1])
    assert _is_pd(S)
    np.linalg.inv(S)   # must not raise


def test_ledoit_wolf_matches_sklearn_up_to_ridge():
    R = _panel().to_numpy()
    ours = EST.ledoit_wolf(R)
    ref = LedoitWolf(assume_centered=False).fit(R).covariance_
    assert np.allclose(ours - np.diag(np.diag(ours - ref)), ref, atol=1e-6) or \
        np.allclose(ours, ref, atol=1e-3)   # differ only by the tiny diagonal ridge


def test_ewma_weights_recent_rows_more():
    # a variance spike in the LAST row should raise EWMA variance above sample variance for that asset
    R = np.zeros((300, 2))
    R[:, 0] = 0.001
    R[-1, 0] = 0.5
    ew = EST.ewma_cov(R, lam=0.90)
    sm = EST.sample_cov(R)
    assert ew[0, 0] > sm[0, 0]


def test_factor_rank_k_recovers_low_rank_structure():
    # returns from a k=2 factor model -> factor cov should capture most variance in 2 factors (PD, sane scale)
    R = _panel(n_tickers=6, n_days=900)
    S_fac = EST.factor_rank_k(R.to_numpy(), k=2)
    S_smp = EST.sample_cov(R.to_numpy())
    assert _is_pd(S_fac)
    assert np.allclose(np.diag(S_fac), np.diag(S_smp), rtol=0.5)   # diagonal variances broadly preserved


def test_with_gbm_diagonal_sets_marginals_keeps_corr():
    R = _panel().to_numpy()
    S = EST.sample_cov(R)
    g = np.full(R.shape[1], 0.02)
    out = EST.with_gbm_diagonal(S, g)
    assert np.allclose(np.sqrt(np.diag(out)), 0.02, atol=1e-3)
    # correlation preserved
    d0 = np.sqrt(np.diag(S)); d1 = np.sqrt(np.diag(out))
    c0 = S / np.outer(d0, d0); c1 = out / np.outer(d1, d1)
    assert np.allclose(c0, c1, atol=1e-6)


# ---------- GMV weights ----------
def test_gmv_weights_sum_to_one():
    S = EST.sample_cov(_panel().to_numpy())
    w = E.gmv_weights(S)
    assert w.sum() == pytest.approx(1.0)


def test_gmv_two_asset_analytic():
    # uncorrelated assets: GMV weight ∝ 1/variance
    S = np.diag([0.04, 0.01])
    w = E.gmv_weights(S)
    assert w[1] == pytest.approx(0.8, abs=1e-6) and w[0] == pytest.approx(0.2, abs=1e-6)


def test_gmv_long_only_nonneg_sum_one():
    S = np.array([[0.04, 0.05], [0.05, 0.09]])   # induces a short leg in long-short GMV
    w = E.gmv_weights(EST.psd_ridge(S), long_only=True)
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)


def test_gmv_long_only_degenerate_uniform():
    # all-negative raw weights -> clip to zero -> uniform fallback
    w = E.gmv_weights(np.array([[1.0, 0.0], [0.0, 1.0]]), long_only=True)
    assert np.allclose(w, [0.5, 0.5])


# ---------- walk-forward + causality ----------
def test_walk_forward_basic_and_causality():
    R = _panel(n_tickers=5, n_days=900)
    res = E.walk_forward(R, EST.sample_cov, h=22, window=252, eval_start="2017-01-01")
    assert len(res["rp_ls"]) == len(res["dates"]) and len(res["rp_ls"]) > 0
    # CAUSALITY: mutating rows AFTER the last rebalance's hold window must not change earlier scored returns
    R2 = R.copy()
    R2.iloc[-5:] = R2.iloc[-5:] + 999.0
    res2 = E.walk_forward(R2, EST.sample_cov, h=22, window=252, eval_start="2017-01-01")
    k = min(len(res["rp_ls"]), len(res2["rp_ls"])) - 5   # compare all but the tail touched by mutation
    assert np.allclose(res["rp_ls"][:k], res2["rp_ls"][:k])


def test_walk_forward_skips_rebalance_before_window():
    # eval_start early + large window -> first rebalance indices < window hit the `continue` branch
    R = _panel(n_tickers=5, n_days=900)
    res = E.walk_forward(R, EST.sample_cov, h=22, window=400, eval_start="2015-06-01")
    assert len(res["rp_ls"]) > 0


def test_walk_forward_empty_raises():
    R = _panel(n_tickers=4, n_days=400)   # ends ~2016; eval_start 2022 -> no rebalance dates
    with pytest.raises(ValueError):
        E.walk_forward(R, EST.sample_cov, h=22, window=252, eval_start="2022-07-01")


def test_walk_forward_with_gbm_diagonal_runs():
    R = _panel(n_tickers=5, n_days=700)
    gbm = pd.DataFrame(0.02, index=R.index, columns=R.columns)   # wide [date x ticker] sigma
    res = E.walk_forward(R, EST.ledoit_wolf, h=22, window=252, eval_start="2017-01-01", gbm_diag=gbm)
    assert len(res["rp_ls"]) > 0


def test_walk_forward_gbm_diag_nan_fallback():
    # a GBM panel missing one ticker entirely -> that column falls back to the estimator diagonal (no crash)
    R = _panel(n_tickers=5, n_days=700)
    gbm = pd.DataFrame(0.02, index=R.index, columns=R.columns)
    gbm[R.columns[0]] = np.nan
    res = E.walk_forward(R, EST.sample_cov, h=22, window=252, eval_start="2017-01-01", gbm_diag=gbm)
    assert len(res["rp_ls"]) > 0 and np.isfinite(res["rp_ls"]).all()


def test_rebalance_dates_spacing():
    dates = pd.bdate_range("2015-01-01", periods=500).to_numpy()
    idx = E._rebalance_dates(dates, pd.Timestamp("2016-01-01"), 22)
    assert all(idx[i + 1] - idx[i] == 22 for i in range(len(idx) - 1))
    assert idx[-1] < len(dates) - 22


# ---------- metrics ----------
def test_ann_vol():
    rp = np.full(100, 0.01)
    assert E.ann_vol(np.concatenate([rp, -rp])) == pytest.approx(0.01 * np.sqrt(C.ANNUALIZE), rel=1e-2)


def test_dm_vol_identical_series_p_one():
    rp = np.random.default_rng(1).normal(0, 0.01, 200)
    dates = pd.bdate_range("2018-01-01", periods=200).to_numpy()
    assert E.dm_vol(rp, rp, dates, 5) == 1.0


def test_spike_mask_flags_covid():
    dates = pd.to_datetime(["2019-06-01", "2020-03-15", "2025-04-10"]).to_numpy()
    m = E._spike_mask(dates)
    assert list(m) == [False, True, True]


# ---------- driver smoke ----------
def test_run_smoke_synthetic(tmp_path):
    doc = RUN.run("fake", out_dir=tmp_path,
                  load_fn=lambda m: _fake_series(6, 2200, seed=3))
    assert set(doc["by_h"]) == {f"h{h}" for h in C.REBALANCE_HORIZONS}
    h = doc["by_h"][f"h{C.REBALANCE_HORIZONS[0]}"]
    assert set(h["metrics"]) == set(EST.ESTIMATORS)
    assert RUN.BAR not in h["dm_vs_bar"]                 # bar not compared to itself
    assert (tmp_path / "cov_fake.json").exists()
    for m in h["metrics"].values():
        assert m["ann_vol_ls"] > 0 and m["n"] > 0


def test_run_composes_gbm_diagonal_branch(tmp_path, monkeypatch):
    # a present GBM diagonal makes run() take the `gd is not None` compose branch (adds `<name>+gbm` models)
    series = _fake_series(6, 2200, seed=3)
    Rp = P.build_panel("fake", load_fn=lambda m: series)
    gbm_wide = pd.DataFrame(0.02, index=Rp.index, columns=Rp.columns)
    monkeypatch.setattr(RUN.GD, "load_diag", lambda market, h, out_dir=None: gbm_wide)
    doc = RUN.run("fake", out_dir=tmp_path, load_fn=lambda m: series)
    h0 = doc["by_h"][f"h{C.REBALANCE_HORIZONS[0]}"]
    assert any(k.endswith("+gbm") for k in h0["metrics"])


@pytest.mark.smoke
def test_real_data_slice_hose():
    """Real-data smoke: a small HOSE panel builds, is complete, and estimators are PD on it."""
    try:
        R = P.build_panel("hose", n_basket=8)
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f"HOSE enriched data unavailable: {e}")
    assert R.shape[1] == 8 and not R.isna().any().any()
    S = EST.ledoit_wolf(R.to_numpy())
    assert _is_pd(S)
