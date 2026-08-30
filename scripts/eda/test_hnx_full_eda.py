"""Unit + smoke tests for the HNX full-EDA generator (scripts/eda/hnx_full_eda.py).

Unique module name (test_hnx_full_eda) avoids the repo's duplicate-basename pytest collision AND is the
adjacent ``test_<module>.py`` the pre-push gate runs for coverage. Covers every detector branch with small
fixtures, one synthetic end-to-end run_eda, and one real-HNX-data-sample smoke.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hnx_full_eda as E  # noqa: E402


# --------------------------------------------------------------------------------------------------
# Pure detectors
# --------------------------------------------------------------------------------------------------
def test_ohlc_violation_mask_flags_each_kind():
    df = pd.DataFrame({
        "open":  [10, 10, 12, 10, -1],   # row2 open>high (oc_out), row4 nonpositive
        "high":  [11, 11, 11, 10, 5],     # row3 high==low (zero_range)
        "low":   [9, 12, 8, 10, 4],       # row1 high<low
        "close": [10, 10, 9, 10, 4],
    })
    m = E.ohlc_violation_mask(df)
    assert m["high_lt_low"].tolist() == [False, True, False, False, False]
    assert m["oc_out"][2] and not m["oc_out"][1]          # high<low counted only in its own bucket
    assert m["zero_range"][3]
    assert m["nonpositive"][4]
    assert m["any"].tolist() == [False, True, True, False, True]


def test_ohlc_tolerance_absorbs_float_noise():
    df = pd.DataFrame({"open": [10.0], "high": [10.0 * (1 - 1e-7)], "low": [9.0], "close": [10.0]})
    assert not E.ohlc_violation_mask(df)["any"][0]         # within OHLC_RTOL -> not flagged


def test_parkinson_variance_formula_and_nan_on_invalid():
    df = pd.DataFrame({"open": [10, 10], "high": [12, 5], "low": [8, 9], "close": [10, 10]})
    pk = E.parkinson_variance(df)
    assert pk[0] == pytest.approx(np.log(12 / 8) ** 2 / (4 * np.log(2)))
    assert np.isnan(pk[1])                                 # high<low -> NaN


def test_log_returns_nonpositive_to_nan():
    r = E.log_returns(np.array([10.0, 20.0, 0.0, 5.0]))
    assert np.isnan(r[0])
    assert r[1] == pytest.approx(np.log(2))
    assert np.isnan(r[2])                                  # close==0
    assert np.isnan(r[3])                                  # prev==0


def test_stale_close_runs_and_empty():
    d = E.stale_close_runs(np.array([5, 5, 5, 5, 5, 6, 7]), min_run=5)
    assert d["max_run"] == 5 and d["n_runs"] == 1 and d["n_days_in_runs"] == 5
    assert E.stale_close_runs(np.array([]))["max_run"] == 0


def test_split_jump_days():
    idx = E.split_jump_days(np.array([1.0, 1.0, 3.0, 3.0]), thr=0.5)   # ln(3)=1.1 > 0.5
    assert idx.tolist() == [2]


def test_robust_z_outliers_paths():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])          # median 3, MAD 1 -> 100 is an outlier
    assert E.robust_z_outliers(x, thr=5.0)[4]
    assert not E.robust_z_outliers(np.array([np.nan, np.nan])).any()   # no finite
    assert not E.robust_z_outliers(np.array([2.0, 2.0, 2.0])).any()    # mad==0


def test_leading_zero_volume():
    assert E.leading_zero_volume(np.array([0, 0, 3, 4])) == 2
    assert E.leading_zero_volume(np.array([0, 0, 0])) == 3             # never trades


def test_skew_kurt_paths():
    assert E._skew_kurt(np.array([1.0, 2.0])) == (0.0, 0.0)            # < 3 points
    assert E._skew_kurt(np.array([3.0, 3.0, 3.0])) == (0.0, 0.0)      # zero variance
    sk, ku = E._skew_kurt(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert sk > 0 and np.isfinite(ku)


def test_acf_paths():
    assert np.isnan(E.acf(np.array([1.0, 2.0, 3.0]), nlags=30)).all()  # too short
    assert (E.acf(np.array([5.0] * 40), nlags=5) == 0).all()          # zero variance
    a = E.acf(np.tile([1.0, -1.0], 40), nlags=2)                       # alternating -> lag1 ~ -1
    assert a[0] < 0


# --------------------------------------------------------------------------------------------------
# Charts (empty-input branches)
# --------------------------------------------------------------------------------------------------
def test_chart_hist_logx_empty_and_normal():
    assert isinstance(E.chart_hist(np.array([0.0, 0.0]), "t", "x", log_x=True), str)   # v.size==0 branch
    assert isinstance(E.chart_hist(np.array([1.0, 2.0, 3.0]), "t", "x"), str)


def test_chart_line():
    assert isinstance(E.chart_line(np.arange(3), np.array([0.1, 0.2, 0.3]), "t", "x", "y"), str)


# --------------------------------------------------------------------------------------------------
# _tbl branches + return_correlations
# --------------------------------------------------------------------------------------------------
def test_tbl_handles_float_nan_and_str():
    df = pd.DataFrame({"a": [1.5, np.nan], "b": ["x", "y"]})
    html = E._tbl(df)
    assert "1.500" in html and "<td>-</td>" in html and "<td>x</td>" in html


def test_return_correlations_min_overlap():
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    a = rng.normal(size=120)
    df = pd.DataFrame({"A": a, "B": a + rng.normal(scale=0.1, size=120),
                       "C": [np.nan] * 60 + list(rng.normal(size=60))}, index=idx)
    m, labels, off = E.return_correlations(df, min_overlap=100)
    assert labels == ["A", "B", "C"]
    assert off.size >= 1 and (off > 0.7).any()             # A,B strongly correlated
    assert np.isnan(m[0, 2])                                # A,C overlap < 100 -> NaN


# --------------------------------------------------------------------------------------------------
# per_ticker_stats manual-Panel branches (no-volume, empty-pk)
# --------------------------------------------------------------------------------------------------
def test_per_ticker_stats_missing_volume_and_empty_pk():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    raw = pd.DataFrame({"date": dates, "open": [10.0] * 6, "high": [11.0] * 6,
                        "low": [9.0] * 6, "close": [10.0] * 6})          # no 'volume' column
    pk_wide = pd.DataFrame({"T": [np.nan] * 6}, index=dates)             # all-NaN -> empty after dropna
    panel = E.Panel(tickers=["T"], raw={"T": raw}, pk_wide=pk_wide,
                    ret_wide=pd.DataFrame({"T": E.log_returns(raw["close"].to_numpy())}, index=dates))
    pt = E.per_ticker_stats(panel)
    assert pt.loc[0, "zero_vol_days"] == 6                  # else-branch zeros
    assert np.isnan(pt.loc[0, "zero_pk_frac"])             # empty pk -> NaN branch


# --------------------------------------------------------------------------------------------------
# End-to-end run_eda on synthetic panels
# --------------------------------------------------------------------------------------------------
def _write_ticker(raw_dir: Path, proc_dir: Path, tk: str, close, high, low, opn, vol, dates):
    raw = pd.DataFrame({"date": dates, "open": opn, "high": high, "low": low,
                        "close": close, "volume": vol})
    raw.to_csv(raw_dir / f"{tk}_ohlcv.csv", index=False)
    pk = E.parkinson_variance(raw)
    pk = np.where(np.isfinite(pk), pk, 0.0)
    pd.DataFrame({"date": dates, "parkinson_volatility": pk}).to_csv(
        proc_dir / f"{tk}_processed.csv", index=False)


def _dirty_panel(tmp_path: Path):
    raw_dir = tmp_path / "raw"; proc_dir = tmp_path / "proc"
    raw_dir.mkdir(); proc_dir.mkdir()
    n = 320
    dates = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(1)
    for tk in ["LIQA", "LIQB"]:                            # liquid -> pass screen, real intraday range
        base = 20 + np.cumsum(rng.normal(scale=0.2, size=n))
        base = np.clip(base, 5, None)
        high = base * (1 + np.abs(rng.normal(scale=0.02, size=n)) + 0.005)
        low = base * (1 - np.abs(rng.normal(scale=0.02, size=n)) - 0.005)
        opn = np.clip(base + rng.normal(scale=0.05, size=n), low, high)
        close = np.clip(base + rng.normal(scale=0.05, size=n), low, high)
        vol = rng.integers(1000, 5000, size=n).astype(float)
        vol[:3] = 0                                        # leading zero volume
        close[100] = close[99] * 2.0                       # >50% jump (split candidate)
        high[100] = close[100] * 1.01; low[100] = close[99] * 0.99
        if tk == "LIQA":
            opn[150] = high[150] * 1.5                      # corrupt: open above high (shared-date seam)
        else:
            opn[150] = high[150] * 1.5
        _write_ticker(raw_dir, proc_dir, tk, close, high, low, opn, vol, dates)
    # illiquid ticker: mostly zero-range (H==L) -> high zero-Parkinson frac -> dropped by screen
    flat = np.full(n, 12.0)
    _write_ticker(raw_dir, proc_dir, "ILLQ", flat, flat.copy(), flat.copy(), flat.copy(),
                  np.zeros(n), dates)
    return raw_dir, proc_dir


def test_run_eda_end_to_end_dirty(tmp_path):
    raw_dir, proc_dir = _dirty_panel(tmp_path)
    out_html = tmp_path / "r.html"; out_md = tmp_path / "r.md"
    stats = E.run_eda(raw_dir, proc_dir, out_html, out_md)
    assert stats["n_tickers"] == 3
    assert stats["n_screened"] == 2 and stats["n_dropped"] == 1
    assert stats["corrupt_total"] >= 2                     # both LIQ tickers corrupt on date[150]
    assert stats["jump_total"] >= 2                        # both have the >50% jump
    assert stats["zero_pk_frac"] > 0                       # ILLQ contributes zeros
    html = out_html.read_text(encoding="utf-8")
    assert "Executive summary" in html and "data:image/png;base64," in html
    md = out_md.read_text(encoding="utf-8")
    assert "Headline dirty-data figures" in md and "seam" in md.lower()


def test_run_eda_clean_single_ticker(tmp_path):
    """Clean, single liquid ticker -> exercises the empty-corr / no-corrupt / no-jump / no-seam branches."""
    raw_dir = tmp_path / "raw"; proc_dir = tmp_path / "proc"
    raw_dir.mkdir(); proc_dir.mkdir()
    n = 300
    dates = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(2)
    base = 30 + np.cumsum(rng.normal(scale=0.1, size=n)); base = np.clip(base, 5, None)
    high = base * 1.02; low = base * 0.98
    opn = np.clip(base + rng.normal(scale=0.03, size=n), low, high)
    close = np.clip(base + rng.normal(scale=0.03, size=n), low, high)
    vol = rng.integers(2000, 4000, size=n).astype(float)
    _write_ticker(raw_dir, proc_dir, "CLEAN", close, high, low, opn, vol, dates)
    stats = E.run_eda(raw_dir, proc_dir, tmp_path / "c.html", tmp_path / "c.md")
    assert stats["n_tickers"] == 1 and stats["corrupt_total"] == 0 and stats["jump_total"] == 0
    assert np.isnan(stats["med_abs_corr"])                 # single ticker -> no pairs
    html = (tmp_path / "c.html").read_text(encoding="utf-8")
    assert "No corrupt OHLC rows detected." in html and "None." in html


# --------------------------------------------------------------------------------------------------
# Real-data-sample smoke (a few real HNX tickers, not the whole panel)
# --------------------------------------------------------------------------------------------------
@pytest.mark.smoke
def test_real_hnx_sample_smoke(tmp_path):
    raw_files = sorted(glob.glob(str(E.HNX_RAW / "*_ohlcv.csv")))[:6]
    if not raw_files:  # pragma: no cover - data present in this repo; guard for a data-less checkout
        pytest.skip("HNX raw data not present")
    raw_dir = tmp_path / "raw"; proc_dir = tmp_path / "proc"
    raw_dir.mkdir(); proc_dir.mkdir()
    for rf in raw_files:
        tk = Path(rf).stem.replace("_ohlcv", "")
        pf = E.HNX_PROC / f"{tk}_processed.csv"
        if not pf.exists():  # pragma: no cover - sampled raw tickers all have a processed file
            continue
        pd.read_csv(rf).to_csv(raw_dir / f"{tk}_ohlcv.csv", index=False)
        pd.read_csv(pf).to_csv(proc_dir / f"{tk}_processed.csv", index=False)
    stats = E.run_eda(raw_dir, proc_dir, tmp_path / "s.html", tmp_path / "s.md")
    assert stats["n_tickers"] >= 1
    assert 0.0 <= stats["zero_pk_frac"] <= 1.0
    assert (tmp_path / "s.html").read_text(encoding="utf-8").startswith("<html>")
