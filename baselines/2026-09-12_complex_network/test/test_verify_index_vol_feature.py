"""Tests for the causal index-volatility-as-GBM-feature check."""
import json

import numpy as np
import pandas as pd

import config
import verify_index_vol_feature as IV


def _fake_frames(n_tickers=25, n_rows=195, seed=0):
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


def _fake_index(n=260, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(n))
    df = pd.DataFrame({"date": dates, "close": close,
                       "volume": np.abs(rng.standard_normal(n)) * 1e6 + 1e5})
    df["lr"] = np.log(df["close"] / df["close"].shift(1))
    return df


def test_idx_rv_series_is_causal_trailing():
    idx = _fake_index()
    s = IV.idx_rv_series(idx)
    assert s.isna().iloc[:config.IDX_RV_MIN_PERIODS - 1].all()      # undefined before min_periods
    k = config.IDX_RV_WINDOW + 3                                    # a fully-populated window
    manual = idx["lr"].to_numpy()[k - config.IDX_RV_WINDOW + 1:k + 1].std(ddof=1)
    assert np.isclose(s.iloc[k], manual)                            # trailing std over past WINDOW days only


def test_merge_idx_rv_broadcasts_by_date():
    frames = _fake_frames(n_tickers=2, n_rows=40)
    idx = _fake_index()
    merged = IV.merge_idx_rv(frames, idx)
    rv = IV.idx_rv_series(idx)
    d = merged["T0"]
    assert IV.FEAT in d.columns
    row = d.iloc[30]
    assert np.isclose(row[IV.FEAT], rv.loc[row["date"]])           # each stock-date gets the index vol at that date


def test_run_idxvol_smoke(monkeypatch):
    """GBM(own) vs GBM(own+idxvol) wires end-to-end on synthetic data with a valid scored fold."""
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(IV.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    out = IV.run_idxvol("hose", load_fn=lambda m: (_fake_frames(), {}, {}), index_fn=lambda m: _fake_index())
    assert "h1" in out
    r = out["h1"]
    assert set(r) >= {"n", "GBM", "GBM+idxvol", "gain_pct", "dm_p",
                      "train_metrics", "test_metrics", "fit_diagnostics"}
    for m in ("GBM", "GBM+idxvol"):
        assert set(r["fit_diagnostics"][m]) == {"verdict", "train_qlike", "test_qlike"}
    json.dumps(out)                                                # JSON-serialisable


def test_run_idxvol_all_folds_skip(monkeypatch):
    """Every fold's train set below min_rows -> all skipped -> empty result (covers the no-data path)."""
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(IV.S1, "FOLDS", ["2015-01-15", "2015-02-01", "2100-01-01"])
    out = IV.run_idxvol("hose", load_fn=lambda m: (_fake_frames(), {}, {}), index_fn=lambda m: _fake_index())
    assert out == {}
