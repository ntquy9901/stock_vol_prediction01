"""End-to-end smoke of run_dy on tiny monkeypatched synthetic frames (no real data), plus the empty-fold
path, the degenerate-feature guard, the default-loader branch and the _success verdict helper."""
import json

import numpy as np
import pandas as pd
import pytest

import config
import run_dy


def _fake_frames(n_tickers=25, n_rows=300, seed=0):
    """n_tickers across 3 sectors with the own-history block FM.OWN present, so FM.panel/FM.gbm run."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_rows)
    base = {s: np.cumsum(rng.standard_normal(n_rows)) * 0.0 + rng.standard_normal(n_rows) for s in range(3)}
    frames = {}
    for i in range(n_tickers):
        sec = i % 3
        pv = np.abs(base[sec] + 0.3 * rng.standard_normal(n_rows)) * 1e-4 + 1e-6
        d = pd.DataFrame({"date": dates, "daily_return": rng.standard_normal(n_rows),
                          "parkinson_variance": pv})
        for c in ["har_daily", "har_weekly", "har_monthly", "rq", "mr_change", "mr_slope5",
                  "mr_slope10", "mr_dev5", "mr_z22"]:
            d[c] = rng.standard_normal(n_rows)
        d["ticker"] = f"T{i}"; d["sector"] = sec
        frames[f"T{i}"] = d
    return frames


def _shrink(monkeypatch):
    monkeypatch.setattr(config, "DY_SECTOR_MIN_STOCKS", 3)
    monkeypatch.setattr(config, "DY_WINDOW", 60)
    monkeypatch.setattr(config, "DY_STEP", 5)
    monkeypatch.setattr(config, "DY_MIN_SECTORS", 2)
    monkeypatch.setattr(config, "HORIZONS", (1,))


def test_run_dy_smoke(monkeypatch):
    """GBM(own) vs GBM(own+spillover) wires end-to-end on synthetic data with a valid scored fold."""
    _shrink(monkeypatch)
    monkeypatch.setattr(run_dy.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    out = run_dy.run_dy("hose", load_fn=lambda m: (_fake_frames(), {}, {}))
    assert "h1" in out
    r = out["h1"]
    assert set(r) >= {"n", "GBM", "GBM+spillover", "gain_pct", "dm_p",
                      "train_metrics", "test_metrics", "fit_diagnostics"}
    for m in ("GBM", "GBM+spillover"):
        assert set(r["fit_diagnostics"][m]) == {"verdict", "train_qlike", "test_qlike"}
    assert set(out["diag"]) == {"n_sectors", "n_anchor_windows", "n_failed_windows"}
    assert isinstance(out["success"], bool)
    json.dumps(out)                                            # JSON-serialisable


def test_run_dy_all_folds_skip_default_loader(monkeypatch):
    """Every fold's train set below min_rows -> all skipped -> no horizon keys; also exercises the
    ``load_fn or FM.load`` default-branch by patching FM.load and passing load_fn=None."""
    _shrink(monkeypatch)
    monkeypatch.setattr(run_dy.S1, "FOLDS", ["2015-04-01", "2015-04-15", "2100-01-01"])
    monkeypatch.setattr(run_dy.FM, "load", lambda m: (_fake_frames(), {}, {}))
    out = run_dy.run_dy("hose", load_fn=None)
    assert not [k for k in out if k.startswith("h")]
    assert out["success"] is False and "diag" in out


def test_run_dy_degenerate_feature_raises(monkeypatch):
    """No sector reaches the constituent threshold -> spillover all-NaN -> fail loud (no silent degradation)."""
    _shrink(monkeypatch)
    monkeypatch.setattr(config, "DY_SECTOR_MIN_STOCKS", 999)
    with pytest.raises(ValueError, match="degenerate"):
        run_dy.run_dy("hose", load_fn=lambda m: (_fake_frames(), {}, {}))


def test_success_verdict_branches():
    assert run_dy._success({"diag": {}}) is False                # no horizon keys
    good = {"h1": {"gain_pct": 0.5, "dm_p": 0.01, "fit_diagnostics": {"GBM+spillover": {"verdict": "ok"}}}}
    assert run_dy._success(good) is True
    weak = {"h1": {"gain_pct": 0.05, "dm_p": 0.01, "fit_diagnostics": {"GBM+spillover": {"verdict": "ok"}}}}
    assert run_dy._success(weak) is False                        # gain below the economic floor
    insig = {"h1": {"gain_pct": 0.5, "dm_p": 0.5, "fit_diagnostics": {"GBM+spillover": {"verdict": "ok"}}}}
    assert run_dy._success(insig) is False                       # DM not significant
    overfit = {"h1": {"gain_pct": 0.5, "dm_p": 0.01, "fit_diagnostics": {"GBM+spillover": {"verdict": "overfit"}}}}
    assert run_dy._success(overfit) is False                     # overfit verdict
