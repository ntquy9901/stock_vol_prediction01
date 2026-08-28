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
    def _fake_fc(series, n_test, horizon, floor, seed=42, return_status=False):
        arr = np.arange(1, n_test + 1, dtype=float)
        st = {"fallback": False, "reason": "", "arch_available": True}
        return (arr, st) if return_status else arr
    monkeypatch.setattr(G.B, "garch_forecast", _fake_fc)
    gd = G._garch_pred(D, horizon=1, cfg=SimpleNamespace(qlike_floor=1e-8))
    preds = [gd[(0, f"d{i}")][1] for i in range(nte)]
    assert preds == [4.0, 5.0, 6.0, 7.0]   # skipped the first n_va=3 (validation) steps


import pytest


def _synth_panel_files(tmp_path, n_tickers=12, n_days=500, seed=0):
    """Write synthetic processed Parkinson CSVs + matching raw OHLCV (with volume) and return
    (files, price_dir) suitable for MR.build_masked_rich -- a REAL panel with a real purge."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_days)
    proc = tmp_path / "proc"; raw = tmp_path / "raw"; proc.mkdir(); raw.mkdir()
    for k in range(n_tickers):
        tk = f"T{k:02d}"
        v = np.empty(n_days); v[0] = 1e-4 * (k + 1)
        for t in range(1, n_days):
            v[t] = 5e-5 * (k + 1) + 0.85 * v[t - 1] + 1e-5 * abs(rng.standard_normal())
        pd.DataFrame({"date": dates, "parkinson_volatility": v}).to_csv(proc / f"{tk}_processed.csv", index=False)
        close = 20.0 + np.cumsum(rng.normal(0, 0.2, n_days))
        span = np.sqrt(v) * close
        pd.DataFrame({"date": dates, "open": close, "high": close + span, "low": close - span,
                      "close": close, "volume": rng.integers(1e5, 1e6, n_days)}).to_csv(
            raw / f"{tk}_ohlcv.csv", index=False)
    return sorted(str(p) for p in proc.glob("*_processed.csv")), str(raw)


@pytest.mark.parametrize("horizon", [1, 5, 10])
def test_garch_integration_alignment_on_real_purged_panel(tmp_path, monkeypatch, horizon):
    """External review F-01: prove the GARCH offset on a REAL MR.build_masked_rich panel (with the real
    train/val/test purge of `horizon` anchors), not a hand-built SimpleNamespace. The delivered design is
    OBSERVATION-CONTIGUOUS: the purge-dropped anchors are targets that exist in NO series, so for node j the
    k-th valid TEST observation is the (n_va_j + k)-th step of the frozen path (n_va_j = the node's valid VAL
    observation count). With garch_forecast mocked as the ramp [1..n_test], each node's chronological test
    predictions must therefore be exactly [n_va_j+1, n_va_j+2, ...]. This holds at every horizon (a purge-
    induced off-by-h in observation space would break it), settling the "drift grows with horizon" concern:
    there is no drift in observation space; the purge changes counts, not the per-node step mapping."""
    files, price_dir = _synth_panel_files(tmp_path)
    D = G.MR.build_masked_rich(files, price_dir, lookback=10, horizon=horizon, min_valid=2, min_train_rows=60)
    def _ramp(series, n_test, horizon, floor, seed=42, return_status=False):
        arr = np.arange(1.0, n_test + 1)
        return (arr, {"fallback": False, "reason": "", "arch_available": True}) if return_status else arr
    monkeypatch.setattr(G.B, "garch_forecast", _ramp)
    gd = G._garch_pred(D, horizon, cfg=SimpleNamespace(qlike_floor=1e-12))
    checked = 0
    for j in range(D.N):
        n_va = int(D.tmask_va[:, j].sum())
        te_rows = [i for i in range(D.y_te.shape[0]) if D.tmask_te[i, j]]
        if not te_rows:
            continue
        got = [gd[(j, D.d_te[i])][1] for i in te_rows]                 # chronological (rows are date-ordered)
        expected = [float(n_va + k + 1) for k in range(len(te_rows))]  # observation-contiguous mapping
        assert got == expected, f"node {j} h{horizon}: {got[:6]} != {expected[:6]}"
        checked += 1
    assert checked >= 2                                                # actually exercised multiple nodes


@pytest.mark.parametrize("horizon", [1, 5, 10, 22])
def test_garch_alignment_with_missing_dates_and_purge(monkeypatch, horizon):
    """External review R-01: prove the observation-space alignment holds when the node has MISSING val/test
    observations (sparse masks) and across horizons -- not just a fully-contiguous ramp. The invariant: the
    k-th valid TEST observation (chronological) receives forecast step (n_va + k), where n_va is the node's
    count of valid VALIDATION observations, so no test target is paired with the wrong observation step. The
    purge/horizon shift is absorbed inside garch_forecast (mocked here as a ramp), and the offset is a pure
    observation count -- independent of the calendar gaps between the retained anchors."""
    tmask_va = np.array([[True], [False], [True], [False], [True]])        # 3 valid of 5 val rows
    tmask_te = np.array([[False], [True], [True], [False], [True], [True]])  # 4 valid of 6 test rows
    n_va, n_te = int(tmask_va.sum()), int(tmask_te.sum())
    D = SimpleNamespace(
        N=1, y_tr=np.ones((40, 1)), y_te=np.ones((6, 1)),
        tmask_tr=np.ones((40, 1), bool), tmask_va=tmask_va, tmask_te=tmask_te,
        t_mean=np.array([1e-9]), d_te=[f"d{i}" for i in range(6)])
    captured = {}
    def _ramp(series, n_test, horizon, floor, seed=42, return_status=False):
        captured["n_test"], captured["horizon"] = n_test, horizon
        arr = np.arange(1.0, n_test + 1)                                    # identifiable ramp
        return (arr, {"fallback": False, "reason": "", "arch_available": True}) if return_status else arr
    monkeypatch.setattr(G.B, "garch_forecast", _ramp)
    gd = G._garch_pred(D, horizon=horizon, cfg=SimpleNamespace(qlike_floor=1e-12))
    # exactly the n_te valid test rows are populated, in chronological order, with fc_full[n_va:]
    valid_rows = [i for i in range(6) if tmask_te[i, 0]]
    got = [gd[(0, f"d{i}")][1] for i in valid_rows]
    assert captured["n_test"] == n_va + n_te and captured["horizon"] == horizon
    assert got == [float(n_va + k + 1) for k in range(n_te)]                # [4,5,6,7]; skipped the 3 val steps


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
    def _fake_fc(series, n_test, horizon, floor, seed=42, return_status=False):
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
