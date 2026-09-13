"""Tests for the case-level idx-vol diagnostic (train/val/test per-(ticker,day) cases)."""
import json

import numpy as np
import pandas as pd

import config
import diag_idxvol_cases as DC


def _frames(n_tickers=25, n_rows=195, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_rows)
    frames = {}
    for t in range(n_tickers):
        d = pd.DataFrame({"date": dates, "daily_return": rng.standard_normal(n_rows),
                          "parkinson_variance": np.abs(rng.standard_normal(n_rows)) * 1e-4 + 1e-6})
        for c in ["har_daily", "har_weekly", "har_monthly", "rq", "mr_change", "mr_slope5",
                  "mr_slope10", "mr_dev5", "mr_z22"]:
            d[c] = rng.standard_normal(n_rows)
        d["ticker"] = f"T{t}"; d["sector"] = t % 3
        frames[f"T{t}"] = d
    return frames


def _index(n=260, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(n))
    df = pd.DataFrame({"date": dates, "close": close, "volume": np.abs(rng.standard_normal(n)) * 1e6 + 1e5})
    df["lr"] = np.log(df["close"] / df["close"].shift(1))
    return df


def test_val_cut_holds_out_tail(monkeypatch):
    monkeypatch.setattr(config, "IDXVOL_VAL_FRAC", 0.2)
    dates = pd.bdate_range("2015-01-01", periods=100).to_numpy()
    cut = DC._val_cut(dates)
    u = np.sort(np.unique(dates))
    assert cut == u[80]                                    # last 20% become validation


def test_cases_ranking_sign():
    """_cases ranks by ΔQLIKE; hurt-mode surfaces positive ΔQLIKE, help-mode negative."""
    frame = pd.DataFrame({"y": np.abs(np.random.default_rng(0).standard_normal(50)) * 1e-4 + 1e-6,
                          "ticker": ["T"] * 50, "date": pd.bdate_range("2015-01-01", periods=50),
                          DC.IV.FEAT: np.linspace(0.01, 0.03, 50)})
    pr = {"GBM": np.full(50, 1e-4), "GBM+idxvol": np.full(50, 5e-4)}   # idxvol predicts worse everywhere
    hurt = DC._cases(frame, pr, best_for_idxvol=False)
    assert len(hurt) == config.IDXVOL_CASE_K
    assert all(c["dqlike"] >= hurt[-1]["dqlike"] for c in hurt)        # sorted desc by ΔQLIKE
    assert hurt[0]["dqlike"] > 0                                       # idxvol hurt (positive ΔQLIKE)


def test_diag_horizon_and_run_cases(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(DC.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    out = DC.run_cases("hose", load_fn=lambda m: (_frames(), {}, {}), index_fn=lambda m: _index())
    assert "h1" in out["horizons"]
    r = out["horizons"]["h1"]
    assert set(r) >= {"fold_start", "val_cut", "train", "val", "test"}
    for sp in ("train", "val", "test"):
        assert set(r[sp]) == {"n", "qlike_own", "qlike_idxvol", "gain_pct", "cases"}
        assert len(r[sp]["cases"]) <= config.IDXVOL_CASE_K
    json.dumps(out)


def test_run_cases_empty_when_no_fold(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(DC.S1, "FOLDS", ["2015-01-15", "2015-02-01", "2100-01-01"])   # all train < min_rows
    out = DC.run_cases("hose", load_fn=lambda m: (_frames(), {}, {}), index_fn=lambda m: _index())
    assert out["horizons"] == {}
