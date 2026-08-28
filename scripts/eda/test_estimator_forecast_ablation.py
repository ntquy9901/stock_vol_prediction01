"""Smoke tests for the estimator forecast ablation (scripts/eda/estimator_forecast_ablation.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estimator_forecast_ablation as AB  # noqa: E402
import volatility_estimators as VE          # noqa: E402


def _synth(tmp_path, tickers=("AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "II", "JJ"), n=400):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", periods=n)
    proc = tmp_path / "proc"; raw = tmp_path / "raw"; proc.mkdir(); raw.mkdir()
    for tk in tickers:
        c = 20.0 + np.cumsum(rng.normal(0, 0.2, n))
        span = np.abs(rng.normal(0, 0.3, n))
        pd.DataFrame({"date": dates, "parkinson_volatility": (span / c) ** 2 + 1e-6}).to_csv(
            proc / f"{tk}_processed.csv", index=False)
        pd.DataFrame({"date": dates, "open": c, "high": c + span, "low": c - span,
                      "close": c, "volume": rng.integers(1e5, 1e6, n)}).to_csv(
            raw / f"{tk}_ohlcv.csv", index=False)
    return proc, raw


def test_write_estimator_processed_same_grid_floored(tmp_path, monkeypatch):
    proc, raw = _synth(tmp_path)
    monkeypatch.setitem(AB.PROC, "synthetic", proc)
    monkeypatch.setitem(VE.PRICE, "synthetic", raw)
    out = tmp_path / "out"; out.mkdir()
    files_p = AB._write_estimator_processed("synthetic", "parkinson", out)
    n_park = pd.read_csv(files_p[0]).shape[0]
    out2 = tmp_path / "out2"; out2.mkdir()
    files_r = AB._write_estimator_processed("synthetic", "rs_overnight", out2)
    n_rs = pd.read_csv(files_r[0]).shape[0]
    # FAIR grid: flooring (not dropping) keeps the SAME row count across estimators (bar the 1 overnight NaN)
    assert abs(n_park - n_rs) <= 1
    # all written targets are strictly positive (floored)
    assert (pd.read_csv(files_r[0])["parkinson_volatility"] > 0).all()


def test_write_estimator_processed_is_date_sorted_and_deduped(tmp_path, monkeypatch):
    # External review H-04: raw with shuffled + duplicate dates must not corrupt the order-dependent
    # rolling/ewm estimators nor the (date, value) pairing. The written processed CSV must be sorted,
    # unique-by-date, and its per-date values must match the estimator computed on the CLEANED raw.
    proc, raw = _synth(tmp_path, tickers=("AA",), n=300)
    rf = raw / "AA_ohlcv.csv"
    clean = pd.read_csv(rf)
    dup = pd.concat([clean, clean.iloc[[10, 20, 30]]], ignore_index=True)   # inject duplicate dates
    shuffled = dup.sample(frac=1.0, random_state=7).reset_index(drop=True)  # shuffle row order
    shuffled.to_csv(rf, index=False)
    monkeypatch.setitem(AB.PROC, "synthetic", proc)
    monkeypatch.setitem(VE.PRICE, "synthetic", raw)
    out = tmp_path / "out"; out.mkdir()
    files = AB._write_estimator_processed("synthetic", "yz_rma20", out)
    got = pd.read_csv(files[0], parse_dates=["date"])
    assert got["date"].is_monotonic_increasing                 # sorted
    assert not got["date"].duplicated().any()                  # deduped
    # values match estimators computed on the CLEANED (sorted, unique) raw
    ref_raw = clean.copy(); ref_raw["date"] = pd.to_datetime(ref_raw["date"])
    ref_raw = ref_raw.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    ref = VE.estimators_from_ohlcv(ref_raw)["yz_rma20"].to_numpy(float)
    ref = np.where(np.isfinite(ref), np.maximum(ref, 1e-10), np.nan)
    ref_df = pd.DataFrame({"date": ref_raw["date"], "v": ref}).dropna()
    merged = got.merge(ref_df, on="date")
    np.testing.assert_allclose(merged["parkinson_volatility"], merged["v"], rtol=1e-9)


def test_keep_tickers_filters_universe(tmp_path, monkeypatch):
    proc, raw = _synth(tmp_path, tickers=("AA", "BB", "CC"))
    monkeypatch.setitem(AB.PROC, "synthetic", proc)
    monkeypatch.setitem(VE.PRICE, "synthetic", raw)
    out = tmp_path / "o"; out.mkdir()
    files = AB._write_estimator_processed("synthetic", "parkinson", out, keep_tickers={"AA", "CC"})
    stems = sorted(Path(f).name.replace("_processed.csv", "") for f in files)
    assert stems == ["AA", "CC"]                     # BB excluded by the fixed universe


def test_screened_tickers_none_for_unscreened_panel():
    # a panel not in the liquidity-screen set keeps all tickers (None sentinel)
    assert AB.screened_tickers("vn30") is None
    assert "hnx" in AB.SCREEN                          # illiquid panels ARE screened


def test_harx_scores_smoke(tmp_path, monkeypatch):
    proc, raw = _synth(tmp_path)
    monkeypatch.setitem(AB.PROC, "synthetic", proc)
    monkeypatch.setitem(VE.PRICE, "synthetic", raw)
    out = tmp_path / "o"; out.mkdir()
    files = AB._write_estimator_processed("synthetic", "parkinson", out)
    from config import Config
    s = AB._harx_scores(files, raw, Config(), horizon=1)
    for k in ["n_nodes", "n_obs", "qlike", "mse", "r2", "true_floored_frac"]:
        assert k in s
    assert np.isfinite(s["qlike"]) and s["qlike"] > 0
    assert 0.0 <= s["true_floored_frac"] <= 1.0
