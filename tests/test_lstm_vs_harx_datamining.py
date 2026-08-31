"""Unit + smoke tests for the LSTM-vs-HAR-X VN100 data-mining primitives.

Unique basename (``test_lstm_vs_harx_datamining``) avoids the pytest duplicate-basename collision.
The pure numeric primitives are tested with small synthetic fixtures; a real-data-sample smoke test
skips cleanly when the VN100 processed panel is absent so coverage holds regardless.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MOD_DIR = _ROOT / "scripts" / "analysis"
_SUB_DIR = _ROOT / "submission" / "soict_lstm_gat"   # delivered `metrics` module used by _split_metrics
for _p in (_MOD_DIR, _SUB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lstm_vs_harx_datamining as dm  # noqa: E402


# --- error_by_magnitude ---------------------------------------------------------------------------

def test_error_by_magnitude_flat_predictor_has_larger_top_bin_error():
    # skewed (spike-tailed) target; a central "flat" predictor (median) under-predicts the tail and
    # incurs its largest squared error in the top decile -- the over-smoothing signature.
    y = np.exp(np.linspace(0.0, 4.0, 100))
    flat = np.full(100, float(np.median(y)))
    res = dm.error_by_magnitude(y, flat, n_bins=10)
    assert set(res) == {"count", "mean_target", "mean_pred", "mse", "mae", "qlike", "mean_signed_error"}
    assert res["count"].sum() == 100
    # top decile MSE >> bottom decile MSE for the flat predictor
    assert res["mse"][-1] > res["mse"][0]
    # flat predictor under-predicts the high decile (negative signed error) and over-predicts the low
    assert res["mean_signed_error"][-1] < 0 < res["mean_signed_error"][0]
    # constant prediction over ordered bins => monotone increasing mean target
    assert res["mean_target"][-1] > res["mean_target"][0]
    assert np.all(res["qlike"] >= 0.0)      # QLIKE is non-negative (0 iff exact)


def test_error_by_magnitude_perfect_predictor_zero_error():
    y = np.linspace(1, 5, 50)
    res = dm.error_by_magnitude(y, y.copy(), n_bins=5)
    assert np.allclose(res["mse"], 0.0)
    assert np.allclose(res["mean_signed_error"], 0.0)


def test_error_by_magnitude_invalid_bins_and_inputs_raise():
    y = np.arange(10.0)
    with pytest.raises(ValueError):
        dm.error_by_magnitude(y, y, n_bins=0)
    with pytest.raises(ValueError):
        dm.error_by_magnitude(y, y, n_bins=11)
    with pytest.raises(ValueError):
        dm.error_by_magnitude(y, y[:5], n_bins=2)          # shape mismatch
    with pytest.raises(ValueError):
        dm.error_by_magnitude(np.array([]), np.array([]), n_bins=1)  # empty
    with pytest.raises(ValueError):
        dm.error_by_magnitude(np.array([1.0, np.nan]), np.array([1.0, 2.0]), n_bins=1)  # non-finite


# --- per_obs_qlike --------------------------------------------------------------------------------

def test_per_obs_qlike_matches_closed_form_and_zero_at_exact():
    rng = np.random.default_rng(3)
    y = rng.uniform(0.1, 2.0, size=50)
    p = rng.uniform(0.1, 2.0, size=50)
    r = y / p
    expected = r - np.log(r) - 1.0            # independent recompute of the QLIKE formula
    assert np.allclose(dm.per_obs_qlike(y, p, floor=1e-12), expected)
    assert np.allclose(dm.per_obs_qlike(y, y), 0.0)   # exact forecast -> 0


def test_per_obs_qlike_bad_floor_raises():
    with pytest.raises(ValueError):
        dm.per_obs_qlike(np.array([1.0]), np.array([1.0]), floor=0.0)


def test_per_obs_qlike_penalises_under_prediction_more_than_over():
    # QLIKE is asymmetric: under-predicting a spike (p << y) costs more than the symmetric over-predict.
    y = np.array([4.0])
    under = dm.per_obs_qlike(y, np.array([2.0]))[0]   # predict half
    over = dm.per_obs_qlike(y, np.array([8.0]))[0]    # predict double
    assert under > over


# --- variance_ratio -------------------------------------------------------------------------------

def test_variance_ratio_known_value_and_compression():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    compressed = y * 0.5
    assert dm.variance_ratio(compressed, y) == pytest.approx(0.25)
    assert dm.variance_ratio(y, y) == pytest.approx(1.0)


def test_variance_ratio_zero_target_variance_raises():
    with pytest.raises(ValueError):
        dm.variance_ratio(np.array([1.0, 2.0]), np.array([3.0, 3.0]))


# --- signed_bias_top_decile -----------------------------------------------------------------------

def test_signed_bias_top_decile_detects_spike_under_prediction():
    y = np.arange(100.0)
    under = y - 5.0              # uniformly 5 below -> tail signed bias -5
    assert dm.signed_bias_top_decile(y, under, q=0.9) == pytest.approx(-5.0)


def test_signed_bias_top_decile_bad_quantile_raises():
    y = np.arange(10.0)
    with pytest.raises(ValueError):
        dm.signed_bias_top_decile(y, y, q=1.5)


# --- generalization_gap ---------------------------------------------------------------------------

def test_generalization_gap_values_and_ratio():
    g = dm.generalization_gap(0.5, 0.6)
    assert g["diff"] == pytest.approx(0.1)
    assert g["ratio"] == pytest.approx(1.2)


def test_generalization_gap_zero_train_is_inf_and_nonfinite_raises():
    assert dm.generalization_gap(0.0, 0.3)["ratio"] == float("inf")
    with pytest.raises(ValueError):
        dm.generalization_gap(np.nan, 0.3)


# --- OLS R^2 helpers ------------------------------------------------------------------------------

def test_har_ols_r2_perfect_linear_is_one():
    x = np.linspace(0, 10, 40)
    X = x.reshape(-1, 1)
    y = 2.0 + 3.0 * x
    assert dm.har_ols_r2(X, y) == pytest.approx(1.0)


def test_ols_oos_r2_generalises_on_clean_linear_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    beta = np.array([1.5, -2.0])
    y = 0.3 + x @ beta
    r2 = dm.ols_oos_r2(x[:150], y[:150], x[150:], y[150:])
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_ols_fit_validates_shapes():
    with pytest.raises(ValueError):
        dm._ols_fit(np.arange(10.0), np.arange(10.0))      # X not 2-D
    with pytest.raises(ValueError):
        dm._ols_fit(np.ones((5, 2)), np.ones(4))           # row mismatch


def test_r2_constant_target_branches():
    # ss_tot == 0: exact -> 1.0, inexact -> 0.0
    assert dm._r2(np.full(5, 3.0), np.full(5, 3.0)) == 1.0
    assert dm._r2(np.full(5, 3.0), np.full(5, 4.0)) == 0.0


# --- signal_to_noise_har --------------------------------------------------------------------------

def test_signal_to_noise_matches_r2_over_one_minus_r2():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 1))
    y = (2.0 * x[:, 0] + rng.normal(scale=1.0, size=300))
    # independent recompute of the in-sample OLS R^2 (do not reuse dm._r2 / dm.har_ols_r2)
    design = np.column_stack([np.ones(300), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    assert dm.signal_to_noise_har(x, y) == pytest.approx(r2 / (1.0 - r2))
    assert 0.0 < r2 < 1.0


def test_signal_to_noise_perfect_fit_raises():
    x = np.linspace(0, 1, 20).reshape(-1, 1)
    y = 1.0 + 2.0 * x[:, 0]
    with pytest.raises(ValueError):
        dm.signal_to_noise_har(x, y)


# --- autocorr -------------------------------------------------------------------------------------

def test_autocorr_persistent_series_decays_from_high_positive():
    rng = np.random.default_rng(2)
    n = 500
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.9 * s[t - 1] + rng.normal(scale=0.1)
    ac = dm.autocorr(s, max_lag=5)
    assert ac.shape == (5,)
    assert ac[0] > 0.5                      # strong lag-1 persistence
    assert ac[0] > ac[-1]                   # decays with lag


def test_autocorr_validation_branches():
    s = np.arange(10.0)
    with pytest.raises(ValueError):
        dm.autocorr(s, max_lag=0)           # lag < 1
    with pytest.raises(ValueError):
        dm.autocorr(s, max_lag=10)          # lag >= n
    with pytest.raises(ValueError):
        dm.autocorr(np.array([1.0, np.nan, 2.0]), max_lag=1)   # non-finite
    with pytest.raises(ValueError):
        dm.autocorr(np.full(10, 2.0), max_lag=1)               # zero variance


# --- driver integration (GPU-free, synthetic bundle) — verifies the mirrored floor/arg/leakage logic

class _FakeHAR:
    """Stand-in for the delivered ``baselines`` module: OLS predict floored at the given floor only
    (the extra per-node ``1e-2*t_mean`` floor is applied by ``_har_predict_split`` itself)."""

    @staticmethod
    def har_predict(X, coef, floor):
        return np.maximum(coef[0] + X @ coef[1:], floor)


def test_har_predict_split_applies_shared_per_node_floor():
    # node0 floor = 1e-2*1 = 0.01, node1 floor = 1e-2*100 = 1.0; raw prediction is a constant 0.5.
    har3 = np.zeros((2, 2, 3))
    coef = np.array([0.5, 0.0, 0.0, 0.0])
    t_mean = np.array([1.0, 100.0])
    out = dm._har_predict_split(har3, coef, t_mean, _FakeHAR)
    assert out.shape == (2, 2)
    assert np.allclose(out[:, 0], 0.5)      # above node0 floor -> unchanged
    assert np.allclose(out[:, 1], 1.0)      # below node1 floor -> clamped up (identical floor to HAR-X)


def test_harx_predict_split_applies_same_floor():
    har5 = np.zeros((2, 2, 5))
    cx = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    t_mean = np.array([1.0, 100.0])
    out = dm._harx_predict_split(har5, cx, t_mean)
    assert np.allclose(out[:, 0], 0.5)
    assert np.allclose(out[:, 1], 1.0)      # same per-node floor as _har_predict_split (no floor mismatch)


def test_flat_returns_masked_target_then_pred():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[10.0, 20.0], [30.0, 40.0]])
    tmask = np.array([[1, 0], [0, 1]])
    yv, pv = dm._flat(pred, y, tmask)
    assert list(yv) == [10.0, 40.0]
    assert list(pv) == [1.0, 4.0]


def test_analyse_horizon_arg_order_and_structure():
    from types import SimpleNamespace
    rng = np.random.default_rng(7)
    n_tr, n_va, n_te, N = 40, 15, 40, 3

    def pos(shape):
        return rng.uniform(0.5, 2.0, size=shape)

    y2d = {"train": pos((n_tr, N)), "val": pos((n_va, N)), "test": pos((n_te, N))}
    tmask = {"train": np.ones((n_tr, N)), "val": np.ones((n_va, N)), "test": np.ones((n_te, N))}
    # HAR-X pred = 0.5 * actual so var(pred)/var(actual) == 0.25 IFF variance_ratio got (pred, y) order.
    harx = {s: 0.5 * y2d[s] for s in ("train", "val", "test")}
    lstm = {s: 0.8 * y2d[s] for s in ("train", "val", "test")}
    D = SimpleNamespace(tmask_tr=np.ones((n_tr, N)), tmask_te=np.ones((n_te, N)),
                        y_tr=y2d["train"], y_te=y2d["test"],
                        har_tr=pos((n_tr, N, 3)), har_te=pos((n_te, N, 3)), N=N)
    a = dm.analyse_horizon({"y2d": y2d, "tmask": tmask, "harx": harx, "lstm": lstm, "D": D})
    assert set(a) == {"metrics", "flat", "ebm", "vratio", "tail_bias", "gap",
                      "har_is_r2", "har_oos_r2", "snr"}
    # pins the (pred, y) argument order into variance_ratio (wrong order would give 4.0, not 0.25)
    assert a["vratio"]["HAR-X"] == pytest.approx(0.25)
    assert a["vratio"]["actual"] == 1.0
    assert a["ebm"]["HAR-X"]["qlike"].shape == (10,)
    # HAR-X predicts 0.5*actual -> under-predicts everywhere, so the top-decile signed bias is negative
    assert a["tail_bias"]["HAR-X"] < 0.0
    assert a["metrics"]["HAR-X"]["test"]["n"] == n_te * N
    assert np.isfinite(a["snr"])


# --- real-data-sample smoke (skips cleanly if the VN100 panel is absent) --------------------------

@pytest.mark.smoke
def test_real_data_sample_persistence_smoke():
    files = glob.glob(str(dm._SUB / "data" / "vn100" / "*_processed.csv"))
    if not files:
        pytest.skip("VN100 processed panel absent; smoke skipped")  # pragma: no cover
    import pandas as pd
    s = pd.read_csv(files[0], parse_dates=["date"]).sort_values("date")["parkinson_variance"]
    s = s.dropna().to_numpy(dtype=float)
    if s.size < 40 or np.var(s) == 0.0:
        pytest.skip("first VN100 series too short/degenerate for smoke")  # pragma: no cover
    ac = dm.autocorr(s, max_lag=5)
    assert np.isfinite(ac).all()
    assert -1.0 <= ac[0] <= 1.0
