"""Tests for the cross-sectional dispersion (CSV) runner: market_csv scalars, causal placebo shift, feature
merge, feature sets, and a run() smoke on synthetic data."""
import json

import numpy as np
import pandas as pd

import config
import csv_dispersion_test as C


def _frames(nt=25, n=195, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for t in range(nt):
        d = pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n),
                          "daily_return": rng.standard_normal(n) * 0.01,
                          "parkinson_variance": np.abs(rng.standard_normal(n)) * 1e-4 + 1e-6})
        for c in ["har_daily", "har_weekly", "har_monthly", "garman_klass_variance",
                  "rogers_satchell_variance", "yang_zhang_n20", "rq", "mr_change", "mr_slope5",
                  "mr_slope10", "mr_dev5", "mr_z22"]:
            d[c] = np.abs(rng.standard_normal(n)) * 1e-4
        d["ticker"] = f"T{t}"; d["sector"] = t % 3
        out[f"T{t}"] = d
    return out


def test_market_csv_nonneg_and_shift():
    csv = C.market_csv(_frames(nt=6, n=80))
    assert list(csv.columns) == ["csv_neg", "csv_pos", "csv_neg_shift"]
    assert (csv["csv_neg"].dropna() >= 0).all() and (csv["csv_pos"].dropna() >= 0).all()
    # the placebo is the csv_neg series shifted forward by CSV_SHIFT -> first CSV_SHIFT rows are NaN
    assert csv["csv_neg_shift"].head(C.CSV_SHIFT).isna().all()
    pd.testing.assert_series_equal(csv["csv_neg_shift"].iloc[C.CSV_SHIFT:].reset_index(drop=True),
                                   csv["csv_neg"].iloc[:-C.CSV_SHIFT].reset_index(drop=True),
                                   check_names=False)


def test_feature_frames_merges_csv():
    ff = C.feature_frames(_frames(nt=5, n=80))
    d = next(iter(ff.values()))
    assert {"csv_neg", "csv_pos", "csv_neg_shift", "semi_neg"} <= set(d.columns)
    assert len(d) == 80                                   # merge is left-join on date, no row change


def test_sets_earn_toggle():
    assert set(C._sets(False)) == {"base", "base+csv_neg", "base+csv_pos", "base+csv_shift"}
    assert "earn_prox" in C._sets(True)["base"] and "earn_prox" not in C._sets(False)["base"]


def test_run_smoke(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 500})
    monkeypatch.setattr(C.AL.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    out = C.run("hose", load_fn=lambda m: (_frames(), {}, {}))
    r = out["h1"]
    assert r["has_earn"] is False
    assert set(r["qlike"]) == {"base", "base+csv_neg", "base+csv_pos", "base+csv_shift"}
    assert set(r["vs_base"]) == {"base+csv_neg", "base+csv_pos", "base+csv_shift"}
    for m in r["vs_base"]:
        assert set(r["vs_base"][m]) == {"gain_vs_base_pct", "dm_p"}
        assert set(r["spike_robustness"][m]) == {"gain_vs_own_pct_ex_spike", "dm_p_ex_spike", "n_ex_spike"}
    json.dumps(out)


def test_run_skips_horizon_when_pooled_insufficient(monkeypatch):
    # min_rows above any fold's train size -> _pooled returns None for all sets -> horizon skipped
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"default": 10 ** 9})
    monkeypatch.setattr(C.AL.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    assert C.run("hose", load_fn=lambda m: (_frames(nt=5, n=80), {}, {})) == {}
