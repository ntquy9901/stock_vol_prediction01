"""Smoke + structure tests for the principled-vs-own DM runner."""
import json

import numpy as np
import pandas as pd

import config
import run_har as R


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


def test_run_smoke_structure(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 500})
    monkeypatch.setattr(R.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    out = R.run("hose", load_fn=lambda m: (_frames(), {}, {}))
    assert "h1" in out
    r = out["h1"]
    assert set(r) >= {"n", "qlike", "vs_own", "leave_one_out", "train_metrics", "fit_diagnostics"}
    assert set(r["qlike"]) == {"principled", "own"}
    assert set(r["vs_own"]) == {"gain_vs_own_pct", "dm_p"}
    assert set(r["leave_one_out"]) == set(R.FEATURES)          # one LOO entry per feature
    assert set(r["fit_diagnostics"]["principled"]) == {"verdict", "train_qlike", "test_qlike"}
    json.dumps(out)


def test_run_empty_when_no_fold(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 10 ** 9})   # nothing passes
    monkeypatch.setattr(R.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2100-01-01"])
    assert R.run("hose", load_fn=lambda m: (_frames(), {}, {})) == {}
