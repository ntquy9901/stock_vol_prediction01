"""End-to-end smoke of run_index / run_gbm cores on tiny monkeypatched synthetic data (FM.load + load_index +
load_log_volume all faked), plus unit tests for the small pure helpers (fit_verdict, _oos, _n_stats, empty)."""
import json

import numpy as np
import pandas as pd
import pytest

import config
import run_index
import run_gbm
import volume_io

pytestmark = pytest.mark.smoke


def _shrink_index_config(monkeypatch):
    monkeypatch.setattr(config, "MIN_COMMON_TICKERS", 3)
    monkeypatch.setattr(config, "MIN_TRAIN_WINDOWS", 2)
    monkeypatch.setattr(config, "WIN", 10)
    monkeypatch.setattr(config, "WIN_ROBUST", 12)
    monkeypatch.setattr(config, "STEP", 5)
    monkeypatch.setattr(config, "L", 3)
    monkeypatch.setattr(config, "ALPHA_GRID", (0.0, 0.7))


def _fake_stock_frames(n_tickers, n_rows, seed=0, with_own=False):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_rows)
    frames = {}
    for t in range(n_tickers):
        d = pd.DataFrame({
            "date": dates,
            "daily_return": rng.standard_normal(n_rows),
            "parkinson_variance": np.abs(rng.standard_normal(n_rows)) * 1e-4 + 1e-6,
        })
        if with_own:
            for c in ["har_daily", "har_weekly", "har_monthly", "rq", "mr_change", "mr_slope5",
                      "mr_slope10", "mr_dev5", "mr_z22", "market_pk"]:
                d[c] = rng.standard_normal(n_rows)
            d["ticker"] = f"T{t}"; d["sector"] = t % 3
        frames[f"T{t}"] = d
    return frames


def _patch_lnvol(monkeypatch, seed=7):
    """Fake load_log_volume: synthetic ln(volume) for whatever tickers/dates are requested."""
    def fake(market, tickers):
        rng = np.random.default_rng(seed)
        # align to a generous business-day span covering all synthetic frames
        idx = pd.bdate_range("2015-01-01", periods=400)
        return pd.DataFrame({tk: rng.standard_normal(len(idx)) for tk in tickers}, index=idx)
    monkeypatch.setattr(volume_io, "load_log_volume", fake)


def _fake_index(n_rows=130, seed=9):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_rows)
    close = 100 + np.cumsum(rng.standard_normal(n_rows))
    df = pd.DataFrame({"date": dates, "close": close, "volume": rng.integers(1_000, 5_000, n_rows)})
    df["lr"] = np.log(df["close"] / df["close"].shift(1))
    return df


def test_run_index_smoke(monkeypatch):
    _shrink_index_config(monkeypatch)
    _patch_lnvol(monkeypatch)
    monkeypatch.setattr(run_index.FM, "load", lambda m: (_fake_stock_frames(6, 120), None, None))
    monkeypatch.setattr(run_index.market_index, "load_index", lambda m: _fake_index())
    result = run_index.run_index("hose")
    assert set(result) == {"market", "n_windows", "n_tickers_per_window", "headline",
                           "fit_diagnostics", "robustness"}
    assert result["headline"]["win"] == config.WIN
    npw = result["n_tickers_per_window"]
    assert set(npw) == {"min", "median", "max"} and npw["min"] is not None
    for mname in run_index.MODELS:
        assert set(result["headline"][mname]) == set(run_index.TARGETS)
        # every (model,target) reports a per-topology-metric feature-importance block
        assert set(result["headline"][mname]["idx_vol"]["feature_importance"]) == set(config.TOPO)
    assert "alpha_grid" in result["robustness"] and "win132" in result["robustness"]
    json.dumps(result)                              # must be JSON-serialisable


def test_run_gbm_smoke(monkeypatch):
    monkeypatch.setattr(config, "MIN_COMMON_TICKERS", 3)
    monkeypatch.setattr(config, "WIN", 10)
    monkeypatch.setattr(config, "STEP", 5)
    monkeypatch.setattr(config, "HORIZONS", (1,))
    _patch_lnvol(monkeypatch)
    # fold 0 has a tiny train set (< min_rows) and is skipped; folds 1-2 are valid
    monkeypatch.setattr(run_gbm.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    monkeypatch.setattr(run_gbm.FM, "load", lambda m: (_fake_stock_frames(25, 195, with_own=True), {}, {}))
    out = run_gbm.run_gbm("hose")
    assert "h1" in out
    r = out["h1"]
    assert set(r) >= {"n", "GBM", "GBM+topo", "gain_pct", "dm_p",
                      "train_metrics", "test_metrics", "fit_diagnostics"}
    for m in ("GBM", "GBM+topo"):
        assert set(r["fit_diagnostics"][m]) == {"verdict", "train_qlike", "test_qlike"}
    json.dumps(out)


def test_run_gbm_all_folds_skip(monkeypatch):
    # every fold's train set is smaller than min_rows -> all skipped -> empty result (covers the no-data path)
    monkeypatch.setattr(config, "MIN_COMMON_TICKERS", 3)
    monkeypatch.setattr(config, "WIN", 10)
    monkeypatch.setattr(config, "STEP", 5)
    monkeypatch.setattr(config, "HORIZONS", (1,))
    _patch_lnvol(monkeypatch)
    monkeypatch.setattr(run_gbm.S1, "FOLDS", ["2015-01-15", "2015-02-01", "2100-01-01"])
    monkeypatch.setattr(run_gbm.FM, "load", lambda m: (_fake_stock_frames(25, 195, with_own=True), {}, {}))
    out = run_gbm.run_gbm("hose")
    assert out == {}


def test_fit_verdict_branches():
    assert run_index.fit_verdict(0.9, 0.2)["verdict"] == "overfit"      # train-test gap > 0.30
    assert run_index.fit_verdict(0.05, 0.02)["verdict"] == "underfit"   # both below floor
    assert run_index.fit_verdict(0.8, 0.7)["verdict"] == "ok"           # healthy
    assert run_index.fit_verdict(float("nan"), 0.5)["verdict"] == "underfit"  # not enough windows


def test_oos_empty_returns_nan():
    r2, rmse = run_index._oos(np.array([]), np.array([]))
    assert np.isnan(r2) and np.isnan(rmse)


def test_n_stats_empty_and_nonempty():
    assert run_index._n_stats(pd.Series([], dtype=float)) == {"min": None, "median": None, "max": None}
    s = run_index._n_stats(pd.Series([20.0, 30.0, 40.0]))
    assert s == {"min": 20, "median": 30.0, "max": 40}


def test_empty_sample_paths(monkeypatch):
    # MIN_COMMON high -> no windows -> empty S; evaluate_sample must return NaN metrics, not crash
    monkeypatch.setattr(config, "MIN_COMMON_TICKERS", 999)
    _patch_lnvol(monkeypatch)
    frames = _fake_stock_frames(5, 120)
    S, n_by_window = run_index.build_sample(frames, "hose", _fake_index(), config.ALPHA, config.WIN)
    assert S.empty and n_by_window.empty
    headline, diags = run_index.evaluate_sample(S)
    r = headline["LinearRegression"]["idx_vol"]
    assert r["n_test"] == 0 and np.isnan(r["r2_oos"])
