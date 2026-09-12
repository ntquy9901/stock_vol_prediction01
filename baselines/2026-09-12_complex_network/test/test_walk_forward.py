"""Tests for run_index.walk_forward (causal expanding walk-forward with the target_end embargo)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import config
import run_index


def test_walk_forward_excludes_leaked_future_target(monkeypatch):
    monkeypatch.setattr(config, "MIN_TRAIN_WINDOWS", 2)
    d0_test = pd.Timestamp("2020-06-01")
    # 4 clean training rows y = 2x, target fully observed before d0_test
    rows = [
        {"d0": pd.Timestamp("2020-01-15"), "target_end": pd.Timestamp("2020-01-20"), "x": 1.0, "y": 2.0},
        {"d0": pd.Timestamp("2020-02-15"), "target_end": pd.Timestamp("2020-02-20"), "x": 2.0, "y": 4.0},
        {"d0": pd.Timestamp("2020-03-15"), "target_end": pd.Timestamp("2020-03-20"), "x": 3.0, "y": 6.0},
        {"d0": pd.Timestamp("2020-04-15"), "target_end": pd.Timestamp("2020-04-20"), "x": 4.0, "y": 8.0},
        # poison: target_end AFTER d0_test -> must be excluded from the test row's training set
        {"d0": pd.Timestamp("2020-05-15"), "target_end": pd.Timestamp("2020-06-15"), "x": 100.0, "y": 9999.0},
        # the test row: y = 2x = 10 if (and only if) the poison row is excluded
        {"d0": d0_test, "target_end": pd.Timestamp("2020-07-01"), "x": 5.0, "y": 10.0},
    ]
    S = pd.DataFrame(rows)
    ys, yhats, tr2, n_tr, imp = run_index.walk_forward(S, ["x"], "y", LinearRegression)
    # the last scored point is the test row; poison excluded -> prediction ~= 10
    assert ys[-1] == 10.0
    assert abs(yhats[-1] - 10.0) < 1e-6
    assert n_tr == 4                               # exactly the 4 clean rows trained the final fit
    assert imp.shape == (1,)                       # standardized LR coef, one per feature


def test_walk_forward_skips_when_train_too_small(monkeypatch):
    monkeypatch.setattr(config, "MIN_TRAIN_WINDOWS", 3)
    # every row's target_end is in the future of every d0 -> no eligible training rows, ever
    rows = [
        {"d0": pd.Timestamp("2020-01-01"), "target_end": pd.Timestamp("2030-01-01"), "x": 1.0, "y": 2.0},
        {"d0": pd.Timestamp("2020-02-01"), "target_end": pd.Timestamp("2030-02-01"), "x": 2.0, "y": 4.0},
    ]
    S = pd.DataFrame(rows)
    ys, yhats, tr2, n_tr, imp = run_index.walk_forward(S, ["x"], "y", LinearRegression)
    assert len(ys) == 0 and len(yhats) == 0 and len(tr2) == 0
    assert n_tr == 0
    assert imp.shape == (1,) and np.isnan(imp).all()   # no fold scored -> all-NaN importance


def test_walk_forward_returns_aligned_arrays(monkeypatch):
    monkeypatch.setattr(config, "MIN_TRAIN_WINDOWS", 2)
    n = 10
    d0 = pd.bdate_range("2020-01-01", periods=n)
    S = pd.DataFrame({
        "d0": d0,
        "target_end": d0,                          # target_end == d0 -> every earlier row is eligible
        "x": np.arange(n, dtype=float),
        "y": 2.0 * np.arange(n, dtype=float),
    })
    ys, yhats, tr2, n_tr, imp = run_index.walk_forward(S, ["x"], "y", LinearRegression)
    assert len(ys) == len(yhats) == len(tr2)
    assert len(ys) > 0
    assert np.isfinite(yhats).all()


def test_walk_forward_rf_importance_shape_six(monkeypatch):
    # RandomForest branch of _feature_importance: importance vector length == len(config.TOPO) metrics
    monkeypatch.setattr(config, "MIN_TRAIN_WINDOWS", 5)
    rng = np.random.default_rng(3)
    n = 40
    p = len(config.TOPO)
    d0 = pd.bdate_range("2018-01-01", periods=n)
    X = rng.standard_normal((n, p))
    S = pd.DataFrame({"d0": d0, "target_end": d0})
    for k, c in enumerate(config.TOPO):
        S[c] = X[:, k]
    S["y"] = X @ np.arange(1, p + 1) + 0.01 * rng.standard_normal(n)
    ys, yhats, tr2, n_tr, imp = run_index.walk_forward(
        S, config.TOPO, "y", lambda: RandomForestRegressor(**config.RF_KW))
    assert imp.shape == (p,)
    assert np.isfinite(imp).all() and (imp >= 0).all()   # RF importances are non-negative
