"""Tests for diag_gbm (Experiment-B 'why topo hurts' diagnostic) and build_topo_diag_html (HTML render)."""
import numpy as np
import pandas as pd
import pytest

import config
import diag_gbm
import build_topo_diag_html as H
import volume_io

_OWN = ["har_daily", "har_weekly", "har_monthly", "rq", "mr_change", "mr_slope5",
        "mr_slope10", "mr_dev5", "mr_z22"]


def _frames(n_tickers=30, n_rows=260, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_rows)
    frames = {}
    for t in range(n_tickers):
        d = pd.DataFrame({"date": dates,
                          "daily_return": rng.standard_normal(n_rows),
                          "parkinson_variance": np.abs(rng.standard_normal(n_rows)) * 1e-4 + 1e-6})
        for c in _OWN + ["market_pk"]:
            d[c] = rng.standard_normal(n_rows)
        d["ticker"] = f"T{t}"
        frames[f"T{t}"] = d
    return frames


def _patch_lnvol(monkeypatch, seed=7):
    def fake(market, tickers):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2015-01-01", periods=400)
        return pd.DataFrame({tk: rng.standard_normal(len(idx)) for tk in tickers}, index=idx)
    monkeypatch.setattr(volume_io, "load_log_volume", fake)


def _shrink(monkeypatch):
    monkeypatch.setattr(config, "MIN_COMMON_TICKERS", 3)
    monkeypatch.setattr(config, "WIN", 10)
    monkeypatch.setattr(config, "STEP", 5)
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(diag_gbm, "PERM_REPEATS", 1)


@pytest.mark.smoke
def test_run_diag_smoke(monkeypatch):
    _shrink(monkeypatch)
    _patch_lnvol(monkeypatch)
    monkeypatch.setattr(diag_gbm.S1, "FOLDS", ["2015-08-01", "2015-10-01", "2100-01-01"])
    monkeypatch.setattr(diag_gbm.FM, "load", lambda m: (_frames(), {}, {}))
    out = diag_gbm.run_diag("hose")
    assert "h1" in out
    r = out["h1"]
    assert set(r) >= {"perm_importance", "own_importance_sum", "topo_importance_sum",
                      "topo_importance_frac", "pred_corr", "qlike_err_corr", "train_qlike", "test_qlike"}
    # importance reported for all 15 features (own 9 + topo 6)
    assert set(r["perm_importance"]) == set(diag_gbm.FM.OWN + config.TOPO)
    assert -1.0 <= r["pred_corr"] <= 1.0


def test_run_diag_skips_when_no_valid_fold(monkeypatch):
    _shrink(monkeypatch)
    _patch_lnvol(monkeypatch)
    # test window lies entirely before any data -> every fold has empty te or tr -> fold is None -> skip
    monkeypatch.setattr(diag_gbm.S1, "FOLDS", ["2000-01-01", "2000-02-01", "2000-03-01"])
    monkeypatch.setattr(diag_gbm.FM, "load", lambda m: (_frames(n_tickers=5, n_rows=60), {}, {}))
    out = diag_gbm.run_diag("hose")
    assert out == {}


def test_qlike_scorer_and_train_qlike():
    rng = np.random.default_rng(1)
    tr = pd.DataFrame({"x": rng.standard_normal(200), "y": np.abs(rng.standard_normal(200)) * 1e-4 + 1e-6})
    m = diag_gbm._fit(tr.assign(**{c: tr["x"] for c in ["x"]}), ["x"])
    X = tr[["x"]].to_numpy(float)
    y = np.maximum(tr["y"].to_numpy(float), diag_gbm.FL)
    assert diag_gbm._qlike_scorer(m, X, y) < 0                # negative mean QLIKE (higher is better)
    assert diag_gbm._train_qlike(m, tr, ["x"]) > 0


def test_perm_importance_subsamples_when_large(monkeypatch):
    monkeypatch.setattr(diag_gbm, "PERM_N", 20)               # force the te.sample branch
    monkeypatch.setattr(diag_gbm, "PERM_REPEATS", 1)
    rng = np.random.default_rng(2)
    n = 80
    te = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n),
                       "y": np.abs(rng.standard_normal(n)) * 1e-4 + 1e-6})
    m = diag_gbm._fit(te, ["a", "b"])
    imp = diag_gbm._perm_importance(m, te, ["a", "b"])
    assert set(imp) == {"a", "b"} and all(np.isfinite(v) for v in imp.values())


# ---------- build_topo_diag_html ----------

def _tiny_result():
    return {"h1": {"GBM": 1.57, "GBM+topo": 1.58, "gain_pct": -0.67, "dm_p": 0.0},
            "h10": {"GBM": 1.68, "GBM+topo": 1.68, "gain_pct": 0.17, "dm_p": 0.28}}


def _tiny_diag():
    return {"h1": {"n_test": 1000,
                   "perm_importance": {"har_daily": 0.9, "har_weekly": 0.3, "dens": 0.001, "betw": -0.002,
                                       "clus": 0.0, "avg_deg": 0.0, "avg_w": 0.0, "diam": 0.0},
                   "own_importance_sum": 1.2, "topo_importance_sum": -0.001,
                   "topo_importance_frac": 0.0, "pred_corr": 0.987, "qlike_err_corr": 0.98,
                   "train_qlike": {"GBM": 1.86, "GBM+topo": 1.85}, "test_qlike": {"GBM": 1.57, "GBM+topo": 1.58}}}


def test_build_html_contains_sections_and_bar_classes():
    html = H.build_html(_tiny_result(), _tiny_diag(), "hose")
    assert "Experiment B" in html and "why adding the topology metrics" in html
    assert "har_daily" in html and "betw" in html
    assert "bar own" in html and "bar topo" in html              # both feature classes rendered
    assert "-0.67%" in html                                       # headline gain formatted
    assert "class=bad" in html and "class=good" in html          # _sign both branches exercised


def test_importance_bars_empty_returns_blank():
    assert H._importance_bars({}) == ""                          # empty -> no bars, scale default path


def test_sign_helper():
    assert H._sign(0.5) == "good" and H._sign(-0.5) == "bad"
    assert H._sign(-0.5, inv=True) == "good" and H._sign(0.5, inv=True) == "bad"
