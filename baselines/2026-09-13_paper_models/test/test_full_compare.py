"""Tests for the full-comparison harness: DM matrix + stubbed run() loop (all branches)."""
import json

import numpy as np
import pandas as pd

import config
import full_compare as FC


def test_dm_matrix_all_pairs():
    rng = np.random.default_rng(0)
    err = {"A": rng.random(200), "B": rng.random(200), "C": rng.random(200)}
    dates = pd.bdate_range("2020-01-01", periods=200).to_numpy()
    dm = FC.dm_matrix(err, dates, h=1)
    assert set(dm) == {"A_vs_B", "A_vs_C", "B_vs_C"}          # every unordered pair, once
    assert all(0.0 <= p <= 1.0 for p in dm.values())


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


def _stub(monkeypatch):
    """Stub the heavy per-fold deps; each GBM feature-set gets a DISTINCT constant so DM is well defined."""
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(FC.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    monkeypatch.setattr(FC.S1, "build_graph", lambda tr, tk, rng: (np.zeros((len(tk), len(tk))), None))
    monkeypatch.setattr(FC.FM, "nb", lambda fold, tk, W: np.zeros(len(fold)))
    monkeypatch.setattr(FC.PM, "nb_col", lambda fold, tk, W, col: np.zeros(len(fold)))
    monkeypatch.setattr(FC.PM, "adj_sector", lambda tk, sect: np.zeros((len(tk), len(tk))))
    monkeypatch.setattr(FC.FM, "_har_ols", lambda trf, tef: np.full(len(tef), 2e-4))
    monkeypatch.setattr(FC.FM, "_harq_ols", lambda trf, tef: np.full(len(tef), 3e-4))
    monkeypatch.setattr(FC.FM, "gbm", lambda trf, tef, cols, seed:
                        np.full(len(tef), 1e-4 + 1e-9 * sum(ord(ch) for c in cols for ch in c)))


def test_run_smoke_hose_with_earn(monkeypatch):
    """market='hose' loads real crawled VN earnings (48/50 True) -> earn models present; full metrics + DM."""
    _stub(monkeypatch)
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 500})
    out = FC.run("hose", load_fn=lambda m: (_frames(), {}, {}))
    r = out["h1"]
    assert "GBM" in r["metrics"] and "GBM+earn" in r["metrics"]              # earn injected
    assert "principled" not in r["metrics"] and "HARQ" not in r["metrics"]   # dropped
    assert set(r["metrics"]["GBM"]) == {"mse", "rmse", "mae", "r2", "qlike"}
    ms = list(r["metrics"]); assert len(r["dm_qlike_matrix"]) == len(ms) * (len(ms) - 1) // 2
    json.dumps(out)


def test_run_sp500_path_no_earn(monkeypatch):
    """market='sp500' skips the earnings inject (48 False); stub edates empty -> no earn models."""
    _stub(monkeypatch)
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 500, "default": 500})
    out = FC.run("sp500", load_fn=lambda m: (_frames(), {}, {}))
    assert "GBM+earn" not in out["h1"]["metrics"]
    assert "GBM" in out["h1"]["metrics"]


def test_run_hose_earn_file_absent(monkeypatch, tmp_path):
    """market='hose' but earnings parquet absent (50 False) -> no earn models."""
    _stub(monkeypatch)
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 500})
    monkeypatch.setattr(FC, "REPO", tmp_path)                              # ep = tmp/.../parquet -> absent
    out = FC.run("hose", load_fn=lambda m: (_frames(), {}, {}))
    assert "GBM+earn" not in out["h1"]["metrics"]


def test_run_empty_when_no_fold(monkeypatch):
    _stub(monkeypatch)
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 10 ** 9})
    assert FC.run("hose", load_fn=lambda m: (_frames(), {}, {})) == {}
