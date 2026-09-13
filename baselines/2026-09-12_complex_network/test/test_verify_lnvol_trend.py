"""Tests for the idx_lnvol trend-artifact verification (time baseline vs topology)."""
import numpy as np
import pandas as pd
import pytest

import config
import verify_lnvol_trend as V


@pytest.fixture(autouse=True)
def _fast_rf(monkeypatch):
    """Shrink the forest + train-window floor so the per-window walk-forward fits run fast in tests."""
    monkeypatch.setattr(config, "RF_KW",
                        dict(n_estimators=20, max_depth=4, min_samples_leaf=5, random_state=0, n_jobs=1))
    monkeypatch.setattr(config, "MIN_TRAIN_WINDOWS", 10)


def _sample(n=60, kind="trend", seed=0):
    """Assembled-style sample S: d0/target_end sorted, the 6 TOPO cols, and idx_lnvol/idx_vol targets.

    kind='trend'  -> idx_lnvol is a pure linear function of time (+noise); idx_vol is white noise.
    kind='topo'   -> idx_lnvol depends on a topology column, not time.
    """
    rng = np.random.default_rng(seed)
    d0 = pd.bdate_range("2018-01-01", periods=n)
    S = pd.DataFrame({"d0": d0, "target_end": d0})
    X = rng.standard_normal((n, len(config.TOPO)))
    for k, c in enumerate(config.TOPO):
        S[c] = X[:, k]
    t = np.arange(n, dtype=float)
    if kind == "trend":
        S["idx_lnvol"] = 0.01 * t + 0.05 * rng.standard_normal(n)      # trend-driven
    else:
        S["idx_lnvol"] = 3.0 * X[:, 0] + 0.05 * rng.standard_normal(n)  # topology-driven
    S["idx_vol"] = 0.05 * rng.standard_normal(n)                        # trend-free noise
    return S


def test_add_time_is_causal_ordinal():
    S = _sample(n=10)
    out = V.add_time(S)
    assert list(out["time"]) == list(range(10))            # ordinal position, sorted order preserved


def test_time_baseline_wins_on_trend_target():
    """On a pure-time-trend idx_lnvol, the linear time baseline explains it; topology (noise features) does not."""
    res = V.verify_sample(V.add_time(_sample(kind="trend")))
    assert res["idx_lnvol"]["time_lr"] > 0.5               # trend captured by time
    assert res["idx_lnvol"]["time_lr"] > res["idx_lnvol"]["topo_rf"]   # topology noise < time


def test_topology_wins_when_signal_is_topological():
    """When idx_lnvol is driven by a topology feature, topology beats the time baseline."""
    res = V.verify_sample(V.add_time(_sample(kind="topo")))
    assert res["idx_lnvol"]["topo_rf"] > res["idx_lnvol"]["time_lr"]


def test_run_verify_wiring(monkeypatch):
    """run_verify wires build_sample -> add_time -> verify_sample with stub loaders (no real data)."""
    S = _sample()
    monkeypatch.setattr(V.run_index, "build_sample", lambda *a, **k: (S, None))
    monkeypatch.setattr(V.FM, "load", lambda m: ({}, {}, {}))
    monkeypatch.setattr(V.market_index, "load_index", lambda m: None)
    out = V.run_verify("hose")
    assert out["market"] == "hose" and out["n_windows"] == len(S)
    assert set(out["targets"]) == {"idx_lnvol", "idx_vol"}
    assert set(out["targets"]["idx_vol"]) == {"topo_rf", "time_lr", "topo_time_rf"}
