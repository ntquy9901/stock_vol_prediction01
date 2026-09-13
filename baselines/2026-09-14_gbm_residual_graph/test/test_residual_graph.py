"""Unit + smoke tests for the residual-graph refiner: reconstruction positivity, causal expanding stack
(no future-fold leakage), signal recovery, noise neutrality, verdict logic, and the run() structure +
per-horizon checkpoint. refine() is tested directly on synthetic streams (fast, no GBM/data)."""
import json

import numpy as np
import pandas as pd

import config
import residual_graph as R


def _stream(nfolds, npf, resid_fn, seed=0):
    """Synthetic fold stream. y = gbm * exp(r) so the log-variance residual is EXACTLY r (controllable)."""
    rng = np.random.default_rng(seed)
    s = []
    for _k in range(nfolds):
        gbm = np.abs(rng.standard_normal(npf)) * 1e-4 + 1e-5
        feats = rng.standard_normal((npf, 3))
        r = resid_fn(feats, rng)
        y = np.maximum(gbm, R.FL) * np.exp(r)
        s.append({"y": y, "gbm": gbm, "feats": feats,
                  "dates": pd.bdate_range("2020-01-01", periods=npf).to_numpy()})
    return s


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


def test_refine_positive_and_identity_first_fold(monkeypatch):
    monkeypatch.setattr(R.config, "MIN_STACK_ROWS", 50)
    s = _stream(3, 100, lambda f, rng: 0.3 * f[:, 0])
    y, gbm, raw, clip, dates, train_mse, test_mse = R.refine(s)
    assert np.all(raw > 0) and np.all(clip > 0)                 # both variants positive by construction
    assert np.allclose(raw[:100], gbm[:100]) and np.allclose(clip[:100], gbm[:100])   # fold 0 identity
    assert len(test_mse) >= 1                                    # later folds activated the ridge


def test_refine_causal_prefix_invariance(monkeypatch):
    """Adding a later fold must not change any earlier fold's output (expanding stack uses only folds < k)."""
    monkeypatch.setattr(R.config, "MIN_STACK_ROWS", 50)
    s = _stream(3, 100, lambda f, rng: 0.3 * f[:, 0])
    _, _, f2, *_ = R.refine(s[:2])
    _, _, f3, *_ = R.refine(s[:3])
    assert np.allclose(f2, f3[:len(f2)])


def test_refine_recovers_linear_signal(monkeypatch):
    monkeypatch.setattr(R.config, "MIN_STACK_ROWS", 50)
    s = _stream(5, 200, lambda f, rng: 0.4 * f[:, 0], seed=1)
    y, gbm, raw, clip, *_ = R.refine(s)
    assert _mse(raw, y) < _mse(gbm, y)             # unbounded ridge recovers the linear residual
    assert _mse(clip, y) < _mse(gbm, y)            # bounded ridge also helps when the signal is in-range


def test_refine_neutral_on_noise(monkeypatch):
    monkeypatch.setattr(R.config, "MIN_STACK_ROWS", 50)
    s = _stream(5, 200, lambda f, rng: rng.standard_normal(len(f)) * 0.3, seed=2)   # resid ⟂ feats
    y, gbm, raw, clip, *_ = R.refine(s)
    assert _mse(clip, y) <= _mse(gbm, y) * 1.5     # no signal -> bounded refiner ~ identity, not worse


def test_verdict():
    assert R._verdict(0.5, 0.01) is True            # gain>0 AND DM-significant
    assert R._verdict(-0.5, 0.01) is False          # negative gain
    assert R._verdict(0.5, 0.5) is False            # not significant


def _frames(nt=25, n=300, seed=0):
    """Frames carrying every column FM.panel / FM.gbm / S1.build_graph / S1.graph_feats read."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    out = {}
    for t in range(nt):
        pv = np.abs(rng.standard_normal(n)) * 1e-4 + 1e-6
        d = pd.DataFrame({"date": dates, "parkinson_variance": pv, "logpk": np.log(pv),
                          "daily_return": rng.standard_normal(n) * 0.01,
                          "volume_zscore_22": rng.standard_normal(n)})
        for c in ["har_daily", "har_weekly", "har_monthly", "rq", "mr_change", "mr_slope5",
                  "mr_slope10", "mr_dev5", "mr_z22"]:
            d[c] = np.abs(rng.standard_normal(n)) * 1e-4
        d["ticker"] = f"T{t}"; d["sector"] = t % 3
        out[f"T{t}"] = d
    return out


def _shrink(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 20})
    monkeypatch.setattr(config, "MIN_STACK_ROWS", 10)          # activate the ridge on the later fold
    monkeypatch.setattr(R.S1, "FOLDS", ["2015-06-01", "2015-09-01", "2015-10-01", "2100-01-01"])


def test_run_structure_and_success(monkeypatch):
    _shrink(monkeypatch)
    out = R.run("hose", load_fn=lambda m: (_frames(), {}, {}))
    r = out["h1"]
    assert set(r) >= {"n", "qlike_gbm", "raw", "clip", "refiner_active_folds", "fit_diagnostics", "verdict"}
    for name in ("raw", "clip"):
        assert set(r[name]) == {"qlike", "gain_vs_gbm_pct", "dm_vs_gbm"}
        assert set(r[name]["dm_vs_gbm"]) == {"p_value", "mean_diff"}
    assert r["verdict"] in {"GO", "NO-GO"}
    assert set(r["fit_diagnostics"]) == {"resid_train_mse", "resid_test_mse", "verdict"}
    assert isinstance(out["success"], bool)
    json.dumps(out)                                            # JSON-serialisable


def test_run_out_path_checkpoints(monkeypatch, tmp_path):
    _shrink(monkeypatch)
    outp = tmp_path / "residual_graph_hose.json"
    out = R.run("hose", load_fn=lambda m: (_frames(), {}, {}), out_path=outp)
    assert outp.exists()                                       # written mid-run, not only at the end
    assert "h1" in json.loads(outp.read_text()) and "h1" in out


def test_run_empty_when_no_fold(monkeypatch):
    _shrink(monkeypatch)
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 10 ** 9})
    out = R.run("hose", load_fn=lambda m: (_frames(), {}, {}))
    assert [k for k in out if k.startswith("h")] == [] and out["success"] is False
