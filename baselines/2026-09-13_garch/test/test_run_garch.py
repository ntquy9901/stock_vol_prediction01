"""Smoke + structure tests for the GARCH walk-forward runner (serial n_jobs=1, stub loader)."""
import json

import numpy as np
import pandas as pd
import pytest

import config
import run_garch as R


def _frames(nt=8, n=180, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for t in range(nt):
        d = pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n),
                          "daily_return": rng.standard_normal(n) * 0.015,
                          "parkinson_variance": np.abs(rng.standard_normal(n)) * 1e-4 + 1e-6})
        # OWN feature columns consumed by FM.panel's dropna + FM._har_ols
        for c in ["har_daily", "har_weekly", "har_monthly", "rq", "mr_change", "mr_slope5",
                  "mr_slope10", "mr_dev5", "mr_z22"]:
            d[c] = np.abs(rng.standard_normal(n)) * 1e-4
        d["ticker"] = f"T{t}"; d["sector"] = t % 3
        out[f"T{t}"] = d
    return out


def test_run_smoke_structure(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 200})
    monkeypatch.setattr(config, "MIN_TRAIN_OBS", 60)        # low -> exercise BOTH fit and fallback branches
    monkeypatch.setattr(R.S1, "FOLDS", ["2015-05-01", "2015-08-01", "2015-09-15", "2100-01-01"])
    out = R.run_garch("hose", load_fn=lambda m: (_frames(), {}, {}), n_jobs=1)
    assert "h1" in out
    r = out["h1"]
    assert set(r) >= {"n", "n_excluded", "qlike", "gain_vs_HAR_pct", "dm_vs_HAR", "n_fallback",
                      "train_metrics", "fit_diagnostics"}
    assert set(r["qlike"]) == {"GARCH", "GJR-GARCH", "HAR"}
    assert set(r["dm_vs_HAR"]) == {"GARCH", "GJR-GARCH"}
    assert set(r["dm_vs_HAR"]["GARCH"]) == {"p_value", "mean_diff"}
    assert set(r["n_fallback"]) == {"GARCH", "GJR-GARCH"}
    assert r["n"] > 0
    assert all(np.isfinite(v) for v in r["qlike"].values())
    for m in ("GARCH", "GJR-GARCH", "HAR"):
        assert set(r["fit_diagnostics"][m]) == {"verdict", "train_qlike", "test_qlike"}
    json.dumps(out)                                          # JSON-serialisable


class _SerialPool:
    """Stand-in for ThreadPoolExecutor that maps serially -> exercises the n_jobs>1 dispatch branch."""
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, it):
        return [fn(x) for x in it]


def test_run_dispatch_parallel_branch(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 200})
    monkeypatch.setattr(config, "MIN_TRAIN_OBS", 60)
    monkeypatch.setattr(R.S1, "FOLDS", ["2015-05-01", "2015-08-01", "2100-01-01"])
    monkeypatch.setattr(R, "ThreadPoolExecutor", _SerialPool)
    out = R.run_garch("hose", load_fn=lambda m: (_frames(), {}, {}), n_jobs=2)   # n_jobs>1 -> pool branch
    assert out["h1"]["n"] > 0


def test_run_handles_ticker_in_train_but_not_last_test(monkeypatch):
    # T0 is delisted before the last fold's test window: present in train, ABSENT from the last test
    # fold -> its train rows must still be forecast (else the train-metric array holds NaN and fails loud).
    frames = _frames(nt=6, n=180)
    frames["T0"] = frames["T0"].iloc[:120].copy()          # T0 stops trading mid-series
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 150})
    monkeypatch.setattr(config, "MIN_TRAIN_OBS", 50)
    monkeypatch.setattr(R.S1, "FOLDS", ["2015-05-01", "2015-07-01", "2015-08-15", "2100-01-01"])
    out = R.run_garch("hose", load_fn=lambda m: (frames, {}, {}), n_jobs=1)
    assert out["h1"]["n"] > 0
    assert all(np.isfinite(v) for v in out["h1"]["train_metrics"].values())


def test_run_raises_on_unfilled_rows(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 200})
    monkeypatch.setattr(config, "MIN_TRAIN_OBS", 60)       # so tickers are estimable and get forecast
    monkeypatch.setattr(R.S1, "FOLDS", ["2015-05-01", "2015-08-01", "2100-01-01"])
    # force a NaN forecast -> assembly must fail loud rather than score misaligned rows
    monkeypatch.setattr(R.GM, "forecast",
                        lambda returns, p, v, idx, h: np.full(np.asarray(idx).shape[0], np.nan))
    with pytest.raises(ValueError, match="unfilled forecast rows"):
        R.run_garch("hose", load_fn=lambda m: (_frames(), {}, {}), n_jobs=1)


def test_run_excludes_short_history_ticker(monkeypatch):
    # MIN_TRAIN_OBS=100: at the first fold every ticker has < 100 pre-fold returns so GARCH is not
    # estimable -> the whole fold's test rows are EXCLUDED (n_excluded>0, fold skipped) rather than
    # forecast as ~0 variance; the later fold (>=100 history) is still scored.
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 200})
    monkeypatch.setattr(config, "MIN_TRAIN_OBS", 100)
    monkeypatch.setattr(R.S1, "FOLDS", ["2015-05-01", "2015-08-01", "2100-01-01"])
    out = R.run_garch("hose", load_fn=lambda m: (_frames(nt=6, n=180), {}, {}), n_jobs=1)
    assert out["h1"]["n_excluded"] > 0
    assert out["h1"]["n"] > 0


def test_run_empty_when_no_fold(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 10 ** 9})   # nothing passes
    monkeypatch.setattr(R.S1, "FOLDS", ["2015-05-01", "2015-08-01", "2100-01-01"])
    assert R.run_garch("hose", load_fn=lambda m: (_frames(), {}, {}), n_jobs=1) == {}
