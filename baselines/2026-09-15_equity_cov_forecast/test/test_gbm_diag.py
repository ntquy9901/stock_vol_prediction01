"""Tests for the GBM-diagonal panel builder (design section 12): causal walk-forward forecast assembly with
injected load/gbm/seeds, parquet build+load round-trip, and the no-fold error path."""
import numpy as np
import pandas as pd
import pytest

import gbm_diag as GD


def _panel_df(ntick=6, start="2015-01-01", ndays=1400, seed=0):
    """Synthetic long panel [date, ticker, y, <OWN cols>] standing in for FM.panel output."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=ndays)
    tickers = [f"T{i:02d}" for i in range(ntick)]
    df = pd.DataFrame([(d, tk) for tk in tickers for d in dates], columns=["date", "ticker"])
    df["y"] = np.abs(rng.standard_normal(len(df))) * 1e-4
    for c in GD.FM.OWN:
        df[c] = rng.standard_normal(len(df)) * 1e-3
    return df


def _gbm(trf, tef, cols, s):
    return np.abs(np.random.default_rng(s).standard_normal(len(tef))) * 1e-4


def test_forecast_build_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(GD.FM, "panel", lambda frames, edates, h: _panel_df())
    monkeypatch.setattr(GD.S1, "TRAIN_START", "2015-01-01")
    monkeypatch.setattr(GD.S1, "FOLDS", ["2018-01-01", "2018-06-01", "2018-07-01"])
    df = GD._forecast_panel("hose", 5, load_fn=lambda m: ({}, {}, {}), gbm_fn=_gbm, seeds=[0, 1])
    assert set(df.columns) == {"date", "ticker", "sigma"}
    assert (df["sigma"] >= 0).all() and len(df) > 0
    # build caches parquet
    p = GD.build("hose", 5, out_dir=tmp_path, load_fn=lambda m: ({}, {}, {}), gbm_fn=_gbm, seeds=[0, 1])
    assert p.exists()
    # load_diag reads + pivots to wide [date x ticker]
    wide = GD.load_diag("hose", 5, out_dir=tmp_path)
    assert wide is not None and wide.shape[1] == 6
    # missing horizon -> None
    assert GD.load_diag("hose", 99, out_dir=tmp_path) is None


def test_forecast_panel_no_folds_raises(monkeypatch):
    monkeypatch.setattr(GD.FM, "panel", lambda frames, edates, h: _panel_df(ndays=120))
    monkeypatch.setattr(GD.S1, "TRAIN_START", "2015-01-01")
    monkeypatch.setattr(GD.S1, "FOLDS", ["2015-01-05", "2015-01-06"])   # train window empty -> all folds skip
    with pytest.raises(ValueError):
        GD._forecast_panel("hose", 5, load_fn=lambda m: ({}, {}, {}),
                           gbm_fn=lambda *a: np.zeros(1), seeds=[0])
