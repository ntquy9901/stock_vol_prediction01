"""Tests for the range-compression (squeeze) runner: causal range_comp/range_exp features, feature merge,
feature sets, and a run() smoke on synthetic data."""
import json

import numpy as np
import pandas as pd

import config
import range_compression_test as R


def _frames(nt=25, n=195, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for t in range(nt):
        d = pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n),
                          "daily_return": rng.standard_normal(n) * 0.01,
                          "log_range": np.abs(rng.standard_normal(n)) * 0.02 + 1e-4,
                          "parkinson_variance": np.abs(rng.standard_normal(n)) * 1e-4 + 1e-6})
        for c in ["har_daily", "har_weekly", "har_monthly", "garman_klass_variance",
                  "rogers_satchell_variance", "yang_zhang_n20", "rq", "mr_change", "mr_slope5",
                  "mr_slope10", "mr_dev5", "mr_z22"]:
            d[c] = np.abs(rng.standard_normal(n)) * 1e-4
        d["ticker"] = f"T{t}"; d["sector"] = t % 3
        out[f"T{t}"] = d
    return out


def test_add_range_features_causal_and_nonneg():
    d = R.add_range_features(next(iter(_frames(nt=1, n=80).values())))
    assert {"range_comp", "range_exp"} <= set(d.columns)
    # long_mean uses min_periods=SEMI_MIN_PERIODS -> first SEMI_MIN_PERIODS-1 rows undefined
    assert d["range_exp"].head(config.SEMI_MIN_PERIODS - 1).isna().all()
    # ratios of non-negative rolling means are non-negative where defined
    assert (d["range_comp"].dropna() >= 0).all() and (d["range_exp"].dropna() >= 0).all()


def test_feature_frames_merges_range():
    ff = R.feature_frames(_frames(nt=5, n=80))
    d = next(iter(ff.values()))
    assert {"range_comp", "range_exp", "semi_neg"} <= set(d.columns)
    assert len(d) == 80                                   # per-ticker transform, no row change


def test_sets_earn_toggle():
    assert set(R._sets(False)) == {"base", "base+comp", "base+exp", "base+both"}
    assert "earn_prox" in R._sets(True)["base"] and "earn_prox" not in R._sets(False)["base"]


def test_run_smoke(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 500})
    monkeypatch.setattr(R.AL.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    out = R.run("hose", load_fn=lambda m: (_frames(), {}, {}))
    r = out["h1"]
    assert r["has_earn"] is False
    assert set(r["qlike"]) == {"base", "base+comp", "base+exp", "base+both"}
    assert set(r["vs_base"]) == {"base+comp", "base+exp", "base+both"}
    for m in r["vs_base"]:
        assert set(r["vs_base"][m]) == {"gain_vs_base_pct", "dm_p"}
        assert set(r["spike_robustness"][m]) == {"gain_vs_own_pct_ex_spike", "dm_p_ex_spike", "n_ex_spike"}
    json.dumps(out)


def test_run_skips_horizon_when_pooled_insufficient(monkeypatch):
    # min_rows above any fold's train size -> _pooled returns None for all sets -> horizon skipped
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"default": 10 ** 9})
    monkeypatch.setattr(R.AL.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    assert R.run("hose", load_fn=lambda m: (_frames(nt=5, n=80), {}, {})) == {}
