"""Tests for the reduced-vs-full own-history DM verification."""
import json

import numpy as np
import pandas as pd

import config
import verify_reduced_features as R


def _frames(n_tickers=25, n_rows=195, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_rows)
    frames = {}
    for t in range(n_tickers):
        d = pd.DataFrame({"date": dates, "daily_return": rng.standard_normal(n_rows),
                          "parkinson_variance": np.abs(rng.standard_normal(n_rows)) * 1e-4 + 1e-6})
        for c in ["har_daily", "har_weekly", "har_monthly", "rq", "mr_change", "mr_slope5",
                  "mr_slope10", "mr_dev5", "mr_z22"]:
            d[c] = rng.standard_normal(n_rows)
        d["ticker"] = f"T{t}"; d["sector"] = t % 3
        frames[f"T{t}"] = d
    return frames


def test_feature_sets_are_own_subsets():
    assert set(R.FEATURE_SETS["own"]) == set(R.FM.OWN)
    assert "rq" not in R.FEATURE_SETS["own_minus_rq"]
    assert set(R.FEATURE_SETS["own_minus_rq"]) == set(R.FM.OWN) - {"rq"}
    assert R.FEATURE_SETS["har3"] == R.FM.HAR


def test_run_reduced_smoke(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(R.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    out = R.run_reduced("hose", load_fn=lambda m: (_frames(), {}, {}))
    assert "h1" in out
    r = out["h1"]
    assert set(r["qlike"]) == set(R.FEATURE_SETS)
    assert set(r["vs_own"]) == {"own_minus_rq", "har3"}               # baseline excluded from comparisons
    for m in ("own_minus_rq", "har3"):
        assert set(r["vs_own"][m]) == {"gain_vs_own_pct", "dm_p"}
    assert set(r["fit_diagnostics"]["own"]) == {"verdict", "train_qlike", "test_qlike"}
    json.dumps(out)


def test_run_reduced_empty_when_no_fold(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(R.S1, "FOLDS", ["2015-01-15", "2015-02-01", "2100-01-01"])
    assert R.run_reduced("hose", load_fn=lambda m: (_frames(), {}, {})) == {}
