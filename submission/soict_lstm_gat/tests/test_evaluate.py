"""Tests for the self-contained evaluate helpers (submission/soict_lstm_gat/evaluate.py).

Bare import works via the submission-folder conftest.py sys.path insert.
"""
from types import SimpleNamespace

import numpy as np

import evaluate as E


def _snap(n_nodes=2, n_tr=300, n_te=6, horizon=5, seed=0):
    rng = np.random.default_rng(seed)
    def _rows(n, start):
        rows = []
        for i in range(n):
            v = np.abs(rng.normal(2e-4, 5e-5, n_nodes))
            rows.append({"date": f"d{start+i}", "y_raw": v})
        return rows
    tr = _rows(int(n_tr * 0.9), 0)
    va = _rows(n_tr - len(tr), len(tr))
    te = _rows(n_te, n_tr)
    return SimpleNamespace(num_nodes=n_nodes, train=tr, val=va, test=te, horizon=horizon)


def test_garch_baseline_shape_finite_and_positive():
    snap = _snap()
    out = E.garch_baseline(snap, floor=1e-8)
    # one entry per (ticker, test date); every prediction finite, positive, >= floor
    assert len(out) == snap.num_nodes * len(snap.test)
    for (j, date), (y_raw, pred) in out.items():
        assert 0 <= j < snap.num_nodes
        assert np.isfinite(pred) and pred >= 1e-8
        assert np.isfinite(y_raw)


def test_garch_baseline_uses_horizon_one(monkeypatch):
    """HIGH-03: garch_baseline must call garch_forecast with horizon=1 (the pre-test series already ends at the
    test boundary; the experiment horizon is not re-applied)."""
    seen = {}
    import baselines as B
    real = B.garch_forecast
    def _spy(series, n_test, horizon=1, floor=1e-8, **kw):
        seen["horizon"] = horizon
        return real(series, n_test=n_test, horizon=horizon, floor=floor)
    monkeypatch.setattr(B, "garch_forecast", _spy)
    E.garch_baseline(_snap(horizon=22), floor=1e-8)
    assert seen["horizon"] == 1


def test_garch_baseline_fails_loud_when_most_tickers_fall_back(monkeypatch):
    """HIGH-01: if GARCH fitting fails for more than the allowlist fraction of tickers, the constant-variance
    fallback must NOT be silently reported as GARCH -- garch_baseline raises."""
    import baselines as B

    def _always_fail(*a, **k):
        raise RuntimeError("convergence failure")

    monkeypatch.setattr(B, "garch_forecast", _always_fail)
    import pytest
    with pytest.raises(ValueError, match="degraded"):
        E.garch_baseline(_snap(n_nodes=2), floor=1e-8)


def test_garch_baseline_tolerates_fallback_within_allowlist(monkeypatch, capsys):
    """A single failing ticker out of many (<= 2%) is allowed but surfaced with a loud warning, not silent."""
    import baselines as B
    real = B.garch_forecast
    state = {"n": 0}

    def _fail_first(series, n_test, horizon=1, floor=1e-8, **kw):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("convergence failure")
        return real(series, n_test=n_test, horizon=horizon, floor=floor)

    monkeypatch.setattr(B, "garch_forecast", _fail_first)
    snap = _snap(n_nodes=100, n_tr=120, n_te=4)
    out = E.garch_baseline(snap, floor=1e-8)  # 1/100 = 1% <= 2% -> no raise
    assert len(out) == snap.num_nodes * len(snap.test)
    assert "constant-variance fallback" in capsys.readouterr().out
