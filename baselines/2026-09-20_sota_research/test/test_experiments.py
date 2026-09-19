"""Unit tests for the overnight SOTA experiments' pure helpers (XAI TreeSHAP + conformal intervals).

Drivers (run_market/main) are data-driven (pragma-no-cover); these cover the testable logic:
conformal band construction + coverage property, spike masking, SHAP aggregation, feature grouping.
"""
import numpy as np
import pytest
import xgboost as xgb

import run_conformal as CF
import run_xai as XAI


def test_cqr_band_finite_sample_coverage():
    rng = np.random.default_rng(0)
    # POSITIVE target (like Parkinson variance) so the FL floor never distorts coverage; the fixed lo/hi
    # quantile "predictions" are deliberately too narrow -> CQR must widen them to reach ~1-alpha on test.
    y_cal = rng.gamma(2.0, 1.0, 5000); lo_cal = np.full(5000, 1.0); hi_cal = np.full(5000, 3.0)
    y_te = rng.gamma(2.0, 1.0, 5000); lo_te = np.full(5000, 1.0); hi_te = np.full(5000, 3.0)
    lo, hi, Q = CF.cqr_band(y_cal, lo_cal, hi_cal, lo_te, hi_te, alpha=0.10)
    assert Q >= 0.0 and np.all(lo >= CF.FL)            # floored (variance non-negative)
    cov = float(((y_te >= lo) & (y_te <= hi)).mean())
    assert 0.87 <= cov <= 0.93                         # ~90% target, sampling tolerance


def test_split_band_coverage_and_floor():
    rng = np.random.default_rng(1)
    y_cal = rng.gamma(3.0, 0.5, 4000); mu_cal = np.full(4000, 1.5)
    y_te = rng.gamma(3.0, 0.5, 4000); mu_te = np.full(4000, 1.5)
    lo, hi, Q = CF.split_band(y_cal, mu_cal, mu_te, alpha=0.10)
    assert Q > 0.0 and np.all(lo >= CF.FL)
    cov = float(((y_te >= lo) & (y_te <= hi)).mean())
    assert 0.87 <= cov <= 0.93


def test_tighter_alpha_widens_band():
    y = np.linspace(-2, 2, 500); mu = np.zeros(500)
    _, _, q90 = CF.split_band(y, mu, mu, alpha=0.10)
    _, _, q99 = CF.split_band(y, mu, mu, alpha=0.01)
    assert q99 >= q90                                   # higher confidence -> wider band


def test_spike_mask_flags_configured_windows():
    dates = np.array(["2020-03-15", "2021-06-01", "2022-05-01", "2025-04-15"], dtype="datetime64[D]")
    m = CF._spike_mask(dates)
    assert m.tolist() == [True, False, True, True]      # COVID, calm, 2022, Apr-2025 tariff


def test_feature_group_buckets():
    assert XAI._feature_group("har_weekly") == "HAR"
    assert XAI._feature_group("earn_prox") == "earnings"
    assert XAI._feature_group("mr_z22") == "momentum"


def test_cov_width():
    y = np.array([1.0, 2.0, 3.0]); lo = np.array([0.5, 0.5, 0.5]); hi = np.array([2.5, 2.5, 2.5])
    cov, mw, mdw = CF._cov_width(y, lo, hi)
    assert cov == pytest.approx(2 / 3)                  # 1.0 and 2.0 inside, 3.0 outside
    assert mw == pytest.approx(2.0) and mdw == pytest.approx(2.0)


def test_fit_quantile_and_pred():
    import pandas as pd
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"f0": rng.normal(size=300), "f1": rng.normal(size=300),
                       "y": np.abs(rng.normal(1.0, 0.3, 300)) + 0.1})
    bst = CF._fit_quantile(df, ["f0", "f1"], 0.5)
    p = CF._pred(bst, df[["f0", "f1"]].to_numpy(float))
    assert p.shape == (300,) and np.all(np.isfinite(p))


def test_build_report_writes_html():
    import build_sota_report as BR
    out = BR.build()                                    # reads whatever result JSONs exist; renders HTML
    assert out.exists()
    txt = out.read_text(encoding="utf-8")
    assert "Overnight SOTA research" in txt and "TreeSHAP" in txt


def test_report_sections_both_branches():
    import build_sota_report as BR
    assert "not available" in BR._shap_section("sp500", None)     # missing-data branch
    assert "not available" in BR._conf_section("hose", None)
    shap_d = {"horizons": {"1": {"n_test_rows": 100,
                                 "group_share": {"HAR": 0.8, "momentum": 0.15, "earnings": 0.05},
                                 "ranking": ["har_weekly", "mr_z22", "earn_prox"],
                                 "importance": {"har_weekly": {"share": 0.5}, "mr_z22": {"share": 0.2},
                                                "earn_prox": {"share": 0.1}}}}}
    assert "HAR" in BR._shap_section("sp500", shap_d)             # data branch
    conf_d = {"target_coverage": 0.9, "horizons": {"1": {
        "split": {"coverage": 0.9, "mean_width": 1e-4, "median_width": 1e-4,
                  "coverage_spike": 0.9, "coverage_calm": 0.9},
        "cqr": {"coverage": 0.9, "mean_width": 1e-4, "median_width": 1e-4,
                "coverage_spike": None, "coverage_calm": None}}}}   # exercises the None-marker branch
    assert "coverage" in BR._conf_section("hose", conf_d)


def test_shap_global_shares_sum_to_one():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 3)); cols = ["har_daily", "mr_change", "earn_prox"]
    y = np.abs(X[:, 0] * 0.5 + 0.1 * X[:, 1]) + 0.3    # positive target for reg:gamma
    bst = xgb.train({"objective": "reg:gamma", "eta": 0.1, "max_leaves": 8, "seed": 0,
                     "tree_method": "hist", "verbosity": 0},
                    xgb.DMatrix(X, label=y), num_boost_round=20)
    g = XAI.shap_global(bst, X, cols)
    assert set(g["per_feature"]) == set(cols)
    assert g["per_feature"]["har_daily"]["mean_abs"] >= 0.0
    assert sum(v["share"] for v in g["per_feature"].values()) == pytest.approx(1.0, abs=1e-6)
    assert set(g["group_share"]) <= {"HAR", "momentum", "earnings"}
    assert sum(g["group_share"].values()) == pytest.approx(1.0, abs=1e-6)
