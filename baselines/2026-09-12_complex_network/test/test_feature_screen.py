"""Tests for the causal per-fold feature screen (Variance -> Pearson/Spearman -> MI -> VIF)."""
import numpy as np
import pandas as pd
import pytest

import config
import feature_screen as fs


def _synth_fold(n=3000, seed=0):
    """Panel with a known-informative feature, a pure-noise feature, and a near-duplicate (collinear) one.

    ``y`` depends non-linearly on ``har_daily``; ``dens`` is independent noise; ``avg_deg`` ~= ``har_daily``.
    All 16 real FEATURES columns are present (others filled with independent noise).
    """
    rng = np.random.default_rng(seed)
    har = rng.normal(size=n)
    df = pd.DataFrame({f: rng.normal(size=n) for f in fs.FEATURES})
    df["har_daily"] = har
    df["avg_deg"] = har + rng.normal(scale=1e-3, size=n)     # duplicate of har_daily -> high VIF
    df["dens"] = rng.normal(size=n)                          # pure noise -> drop
    df["y"] = np.abs(har) + 0.01 * rng.normal(size=n)        # non-linear dependence on har_daily
    df["date"] = pd.date_range("2015-01-01", periods=n, freq="D")
    df["ticker"] = "T"
    return df


def test_vif_flags_collinear_pair():
    """A feature that is a near-copy of another gets a large VIF; an independent one stays ~1."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=500)
    X = np.column_stack([base, base + rng.normal(scale=1e-3, size=500), rng.normal(size=500)])
    vif = fs._vif(X)
    assert vif[0] > config.SCREEN_VIF_HI and vif[1] > config.SCREEN_VIF_HI
    assert vif[2] < 5.0


def test_vif_zero_variance_is_inf():
    X = np.column_stack([np.ones(100), np.random.default_rng(0).normal(size=100)])
    assert np.isinf(fs._vif(X)[0])


def test_fold_stats_detects_signal_and_noise():
    pear, spear, mi, vif, sub = fs._fold_stats(_synth_fold())
    assert sub is False                                     # small fold, no subsample
    i_har = fs.FEATURES.index("har_daily")
    i_dens = fs.FEATURES.index("dens")
    # |y| vs har: strong MI, weak linear Pearson (symmetric), noise feature ~0 on all
    assert mi[i_har] > mi[i_dens]
    assert mi[i_dens] < config.SCREEN_MI_LO * 5
    assert abs(pear[i_dens]) < 0.1


@pytest.mark.parametrize("pear,spear,mi,vif,expect", [
    (0.0, 0.0, 0.0, 1.0, "drop (no signal)"),
    (0.5, 0.5, 0.3, 2.0, "keep"),
    (0.5, 0.5, 0.3, 50.0, "review (redundant)"),
    (0.0, 0.0, 0.2, 1.0, "keep"),                            # non-linear-only signal is kept
])
def test_verdict_rules(pear, spear, mi, vif, expect):
    assert fs._verdict(pear, spear, mi, vif) == expect


def test_screen_horizon_runs_causal_folds():
    """screen_horizon aggregates across folds and returns a verdict per feature; noise feature is dropped."""
    a = _synth_fold(n=6000)
    res = fs.screen_horizon(a, horizon=1, min_rows=10)
    assert res["n_folds"] >= 1
    assert set(res["features"]) == set(fs.FEATURES)
    assert res["features"]["dens"]["verdict"] == "drop (no signal)"
    assert res["features"]["har_daily"]["group"] == "har"
    assert res["features"]["dens"]["group"] == "graph"


def test_fold_stats_mi_subsample(monkeypatch):
    """When train rows exceed the MI cap, MI is estimated on a fixed-seed subsample (flagged, still ranks)."""
    monkeypatch.setattr(config, "SCREEN_MI_CAP", 1000)
    pear, spear, mi, vif, sub = fs._fold_stats(_synth_fold(n=2500))
    assert sub is True
    assert mi[fs.FEATURES.index("har_daily")] > mi[fs.FEATURES.index("dens")]


def test_screen_horizon_skips_thin_folds():
    """Folds with fewer than min_rows causal train rows are skipped (n_folds == 0)."""
    with np.errstate(invalid="ignore"):
        res = fs.screen_horizon(_synth_fold(n=3000), horizon=1, min_rows=10 ** 9)
    assert res["n_folds"] == 0
    assert res["mi_subsampled_folds"] == 0
    assert np.isnan(res["features"]["dens"]["mi"])


@pytest.mark.smoke
def test_run_screen_smoke(monkeypatch):
    """Smoke: run_screen wires panel->per-fold screen for all horizons with a stub loader (no real data)."""
    def fake_load(market):
        return {"T": None}, {}, {}
    monkeypatch.setattr(fs.FM, "load", fake_load)
    monkeypatch.setattr(fs.run_gbm, "_merge_topo", lambda frames, market: frames)
    monkeypatch.setattr(fs.FM, "panel", lambda frames, ed, h: _synth_fold(n=5000, seed=h))
    out = fs.run_screen("hose")
    assert set(out["horizons"]) == {f"h{h}" for h in config.HORIZONS}
    assert out["thresholds"]["vif_hi"] == config.SCREEN_VIF_HI
