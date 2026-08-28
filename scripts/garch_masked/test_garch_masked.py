"""Tests for the masked-panel GARCH add-on (scripts/garch_masked/compute_garch_masked.py).

Covers the pure metric aggregator and the two prediction builders on small synthetic panels,
so the throwaway analysis script ships with executable invariants (floor respected, finite,
correct key count, exact MSE/MAE).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "garch_masked"))
import compute_garch_masked as G  # noqa: E402


def test_metrics_exact():
    pred = {(0, "d1"): (1.0, 1.0), (0, "d2"): (2.0, 4.0)}  # y=[1,2], p=[1,4]
    m = G._metrics(pred, floor=1e-8)
    assert abs(m["mse"] - 2.0) < 1e-9        # (0^2 + 2^2)/2
    assert abs(m["mae"] - 1.0) < 1e-9        # (0 + 2)/2
    assert m["n"] == 2


def _fake_D(N=2, ntr=300, nte=5, seed=0):
    rng = np.random.default_rng(seed)
    y_tr = np.abs(rng.normal(2e-4, 5e-5, (ntr, N)))
    y_te = np.abs(rng.normal(2e-4, 5e-5, (nte, N)))
    har5_tr = np.abs(rng.normal(2e-4, 5e-5, (ntr, N, 5)))
    har5_te = np.abs(rng.normal(2e-4, 5e-5, (nte, N, 5)))
    return SimpleNamespace(
        N=N, y_tr=y_tr, y_te=y_te,
        tmask_tr=np.ones((ntr, N), bool), tmask_va=np.ones((30, N), bool),
        tmask_te=np.ones((nte, N), bool),
        har5_tr=har5_tr, har5_te=har5_te,
        t_mean=y_tr.mean(0), d_te=[f"d{i}" for i in range(nte)])


def test_garch_pred_skips_validation_interval(monkeypatch):
    """_garch_pred forecasts n_va+n_te steps and keeps the LAST n_te (skips the validation block) so each
    test target aligns to its distance from train-end (Codex finding 2)."""
    ntr, nva, nte = 40, 3, 4
    D = SimpleNamespace(
        N=1, y_tr=np.ones((ntr, 1)), y_te=np.ones((nte, 1)),
        tmask_tr=np.ones((ntr, 1), bool), tmask_va=np.ones((nva, 1), bool),
        tmask_te=np.ones((nte, 1), bool), t_mean=np.array([1.0]),
        d_te=[f"d{i}" for i in range(nte)])
    # identifiable ramp: fc_full = [1, 2, ..., n_test]; expect the test window = fc_full[n_va:] = [4,5,6,7]
    def _fake_fc(series, n_test, horizon, floor, return_status=False):
        arr = np.arange(1, n_test + 1, dtype=float)
        st = {"fallback": False, "reason": "", "arch_available": True}
        return (arr, st) if return_status else arr
    monkeypatch.setattr(G.B, "garch_forecast", _fake_fc)
    gd = G._garch_pred(D, horizon=1, cfg=SimpleNamespace(qlike_floor=1e-8))
    preds = [gd[(0, f"d{i}")][1] for i in range(nte)]
    assert preds == [4.0, 5.0, 6.0, 7.0]   # skipped the first n_va=3 (validation) steps


def test_garch_pred_collects_status_and_garch_meta_aggregates(monkeypatch):
    """External review M-08/L-03: _garch_pred(status_out=...) collects per-node fallback status; garch_meta
    aggregates the fallback rate + reasons + provenance. One node fits, one falls back -> rate 0.5."""
    ntr, nva, nte = 40, 3, 4
    D = SimpleNamespace(
        N=2, y_tr=np.ones((ntr, 2)), y_te=np.ones((nte, 2)),
        tmask_tr=np.ones((ntr, 2), bool), tmask_va=np.ones((nva, 2), bool),
        tmask_te=np.ones((nte, 2), bool), t_mean=np.array([1.0, 1.0]),
        d_te=[f"d{i}" for i in range(nte)])
    calls = {"i": 0}
    def _fake_fc(series, n_test, horizon, floor, return_status=False):
        arr = np.arange(1, n_test + 1, dtype=float)
        fb = calls["i"] == 1                       # second node falls back
        calls["i"] += 1
        st = {"fallback": fb, "reason": "arch fit/forecast error: x" if fb else "",
              "arch_available": True}
        return (arr, st) if return_status else arr
    monkeypatch.setattr(G.B, "garch_forecast", _fake_fc)
    cfg = SimpleNamespace(qlike_floor=1e-8, seed=7)
    st_list = []
    G._garch_pred(D, horizon=5, cfg=cfg, status_out=st_list)
    assert len(st_list) == 2
    meta = G.garch_meta(st_list, horizon=5, cfg=cfg)
    assert meta["schema"] == 1 and meta["n_nodes"] == 2 and meta["n_fallback"] == 1
    assert abs(meta["fallback_rate"] - 0.5) < 1e-12
    assert meta["degraded"] is False and meta["horizon"] == 5 and meta["seed"] == 7
    assert "arch fit/forecast error: x" in meta["fallback_reasons"]


def test_garch_pred_floor_finite_and_count():
    D = _fake_D()
    cfg = SimpleNamespace(qlike_floor=1e-8)
    gd = G._garch_pred(D, horizon=1, cfg=cfg)
    assert len(gd) == D.N * len(D.d_te)
    floor = 1e-2 * D.t_mean + 1e-12
    for (j, _), (yv, pv) in gd.items():
        assert np.isfinite(pv) and pv >= floor[j] - 1e-18


def test_harx_pred_shape_and_floor():
    D = _fake_D()
    cfg = SimpleNamespace(qlike_floor=1e-8)
    hd = G._harx_pred(D, cfg)
    assert len(hd) == D.N * len(D.d_te)
    floor = 1e-2 * D.t_mean + 1e-12
    for (j, _), (yv, pv) in hd.items():
        assert np.isfinite(pv) and pv >= floor[j] - 1e-18
