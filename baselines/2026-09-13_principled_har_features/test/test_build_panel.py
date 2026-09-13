"""Tests for the principled feature panel (semivariance add + causality)."""
import numpy as np
import pandas as pd

import build_panel as B


def _frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n),
                      "daily_return": rng.standard_normal(n) * 0.01,
                      "parkinson_variance": np.abs(rng.standard_normal(n)) * 1e-4 + 1e-6})
    for c in ["har_daily", "har_weekly", "har_monthly",
              "garman_klass_variance", "rogers_satchell_variance", "yang_zhang_n20"]:
        d[c] = np.abs(rng.standard_normal(n)) * 1e-4
    return d


def test_features_list_and_added():
    f = B.add_features(_frame())
    assert B.FEATURES[:6] == ["har_daily", "har_weekly", "har_monthly",
                              "garman_klass_variance", "rogers_satchell_variance", "yang_zhang_n20"]
    assert "semi_neg" in f.columns and "semi_pos" in f.columns
    for col in B.FEATURES:
        assert col in f.columns


def test_semivariance_features_causal():
    base = _frame()
    f0 = B.add_features(base.copy())
    perturbed = base.copy()
    perturbed.loc[30:, "daily_return"] *= 5          # change the FUTURE
    f1 = B.add_features(perturbed)
    assert np.allclose(f0.loc[20, ["semi_neg", "semi_pos"]].to_numpy(float),
                       f1.loc[20, ["semi_neg", "semi_pos"]].to_numpy(float), equal_nan=True)


def test_feature_frames_maps_all():
    frames = {"A": _frame(seed=1), "B": _frame(seed=2)}
    out = B.feature_frames(frames)
    assert set(out) == {"A", "B"}
    assert "semi_neg" in out["A"].columns
