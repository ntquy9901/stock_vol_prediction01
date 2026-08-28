"""Tests for the daily volatility-estimator comparison (scripts/eda/volatility_estimators.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volatility_estimators as VE  # noqa: E402


def test_yz_daily_is_proxy_yang_zhang_is_windowed(tmp_path):
    """Senior review HIGH-01: yz_daily is a per-day PROXY (not YZ); yang_zhang is the TRUE windowed estimator.
    Guard the module documents the distinction and both columns exist."""
    import inspect
    doc = inspect.getdoc(VE) or ""
    assert "standard Yang--Zhang (2000) estimator" in doc and "PER-DAY PROXY" in doc
    est = VE.estimators_from_ohlcv(pd.DataFrame({
        "date": range(3), "open": [10.0] * 3, "high": [11.0] * 3, "low": [9.0] * 3,
        "close": [10.5] * 3, "volume": [1] * 3}))
    assert {"yz_daily", "yz_rma20", "yang_zhang"} <= set(est.columns)


def test_windowed_yz_invalid_row_yields_nan_not_inf():
    """Code review LOW-1: an invalid mid-series bar (close=0 -> inf log-returns) must NOT poison the rolling
    Yang-Zhang with inf; windows spanning it are NaN, and yang_zhang is never a wrong finite/inf value."""
    rng = np.random.default_rng(1)
    n = 80
    c = 20.0 + np.cumsum(rng.normal(0, 0.2, n)); o = c.copy()
    hi = np.maximum(o, c) + 0.2; lo = np.minimum(o, c) - 0.2
    c[40] = 0.0                                                   # invalid bar mid-series
    df = pd.DataFrame({"date": range(n), "open": o, "high": hi, "low": lo, "close": c, "volume": [1] * n})
    yz = VE.estimators_from_ohlcv(df)["yang_zhang"].to_numpy()
    assert not np.isinf(yz).any()                                # never inf
    finite = yz[np.isfinite(yz)]
    assert (finite >= 0).all()                                   # variances non-negative where defined


def test_windowed_yang_zhang_matches_reference_formula():
    """The yang_zhang column must equal the standard Yang-Zhang (2000) windowed composite computed
    independently (mean-subtracted n-day sample variances of overnight & open-close returns + rolling RS mean,
    blended by k=0.34/(1.34+(n+1)/(n-1))). Verified against the TTR-R / strimpel definitions."""
    rng = np.random.default_rng(0)
    n = 60
    c = 20.0 + np.cumsum(rng.normal(0, 0.2, n))
    o = c * np.exp(rng.normal(0, 0.01, n))                       # open near prev close-ish, with a gap
    hi = np.maximum.reduce([o, c]) + np.abs(rng.normal(0, 0.15, n))
    lo = np.minimum.reduce([o, c]) - np.abs(rng.normal(0, 0.15, n))
    df = pd.DataFrame({"date": range(n), "open": o, "high": hi, "low": lo, "close": c, "volume": [1] * n})
    est = VE.estimators_from_ohlcv(df, overnight_cap=None)       # no winsor -> exact match to raw returns
    w = VE._YZ_N; k = VE._YZ_K
    prev_c = np.concatenate([[np.nan], c[:-1]])
    r_on = np.log(o / prev_c)
    r_co = np.log(c / o)
    rs = np.log(hi / c) * np.log(hi / o) + np.log(lo / c) * np.log(lo / o)
    ref = (pd.Series(r_on).rolling(w).var() + k * pd.Series(r_co).rolling(w).var()
           + (1 - k) * pd.Series(np.clip(rs, 0, None)).rolling(w).mean()).clip(lower=0.0)
    got = est["yang_zhang"].to_numpy()
    ok = est["ok"].to_numpy() & np.isfinite(ref.to_numpy())
    np.testing.assert_allclose(got[ok], ref.to_numpy()[ok], rtol=1e-9)
    assert np.isfinite(got[ok]).sum() > 20                       # the windowed estimator produced real values


def test_parkinson_formula_matches_definition():
    df = pd.DataFrame({"date": [1, 2], "open": [10.0, 10.0], "high": [11.0, 10.5],
                       "low": [9.0, 9.5], "close": [10.0, 10.0], "volume": [1, 1]})
    est = VE.estimators_from_ohlcv(df)
    exp = np.log(11.0 / 9.0) ** 2 / (4 * np.log(2))
    assert abs(est["parkinson"].iloc[0] - exp) < 1e-12


def test_zero_range_day_intraday_zero_overnight_rescues():
    # day 2 is a limit/gap day: O=H=L=C=12 (zero intraday range) but prior close was 10 (a +20% gap)
    df = pd.DataFrame({"date": [1, 2], "open": [10.0, 12.0], "high": [10.0, 12.0],
                       "low": [10.0, 12.0], "close": [10.0, 12.0], "volume": [1, 1]})
    est = VE.estimators_from_ohlcv(df)
    r = est.iloc[1]
    assert r["is_zero_range"]                       # H approx L
    # intraday-only estimators collapse to 0 on a zero-range bar
    assert r["parkinson"] == 0.0
    assert r["garman_klass"] == 0.0
    assert r["rogers_satchell"] == 0.0
    # overnight-bearing estimators stay strictly positive (the price gapped from the prior close)
    assert r["close2close"] > 0.0
    assert r["rs_overnight"] > 0.0
    assert abs(r["close2close"] - np.log(12.0 / 10.0) ** 2) < 1e-12


def test_zero_prior_close_gives_nan_overnight_not_inf():
    # day 2 follows a ZERO prior close (bad data) -> overnight ln(O/0) must NOT become +inf; NaN instead
    df = pd.DataFrame({"date": [1, 2], "open": [10.0, 5.0], "high": [10.0, 5.5],
                       "low": [10.0, 4.5], "close": [0.0, 5.0], "volume": [1, 1]})
    est = VE.estimators_from_ohlcv(df)
    r = est.iloc[1]
    assert np.isnan(r["close2close"]) and np.isnan(r["rs_overnight"])   # prior close 0 -> overnight undefined
    assert np.isfinite(r["parkinson"])                                  # intraday estimator unaffected


def test_overnight_winsorized_against_unadjusted_split():
    # a 3x overnight gap (unadjusted split / bad data) must be winsorized so it cannot dominate
    df = pd.DataFrame({"date": [1, 2], "open": [10.0, 30.0], "high": [10.0, 30.0],
                       "low": [10.0, 30.0], "close": [10.0, 30.0], "volume": [1, 1]})
    cap = 0.20
    est = VE.estimators_from_ohlcv(df, overnight_cap=cap)
    # overnight contribution is capped at cap^2 (here rs=0 on the flat bar) rather than ln(3)^2 ~ 1.207
    assert abs(est["rs_overnight"].iloc[1] - cap ** 2) < 1e-9
    assert abs(est["close2close"].iloc[1] - cap ** 2) < 1e-9
    # without winsorization the raw spike is ~ln(3)^2, far larger
    raw = VE.estimators_from_ohlcv(df, overnight_cap=None)
    assert raw["rs_overnight"].iloc[1] > 1.0


def test_yz_daily_matches_indicator_formula():
    df = pd.DataFrame({"date": [1, 2], "open": [10.0, 10.2], "high": [10.0, 10.6],
                       "low": [10.0, 9.9], "close": [10.0, 10.3], "volume": [1, 1]})
    est = VE.estimators_from_ohlcv(df, overnight_cap=None)   # exact, no winsor
    r = est.iloc[1]
    r_o = np.log(10.2 / 10.0)
    r_c = np.log(10.3 / 10.2)
    rs = np.log(10.6 / 10.3) * np.log(10.6 / 10.2) + np.log(9.9 / 10.3) * np.log(9.9 / 10.2)
    exp = r_o ** 2 + VE._YZ_K * r_c ** 2 + (1.0 - VE._YZ_K) * max(rs, 0.0)   # r_o^2 + k r_c^2 + (1-k) RS
    assert abs(r["yz_daily"] - exp) < 1e-9
    assert "yz_rma20" in est.columns and np.isfinite(r["yz_rma20"])


def test_yz_rma_is_smoother_than_yz_daily():
    # a series with alternating high/low daily variance -> RMA smoothing reduces variance of the series
    rng = np.random.default_rng(1)
    n = 300
    c = 20.0 + np.cumsum(rng.normal(0, 0.2, n))
    span = np.abs(rng.normal(0, 0.4, n))
    df = pd.DataFrame({"date": range(n), "open": c, "high": c + span, "low": c - span,
                       "close": c + rng.normal(0, 0.1, n), "volume": 1})
    est = VE.estimators_from_ohlcv(df)
    d = est["yz_daily"].dropna().to_numpy()
    s = est["yz_rma20"].dropna().to_numpy()
    assert np.nanstd(s) < np.nanstd(d)          # smoothed series is less volatile


def test_invalid_prices_are_nan():
    df = pd.DataFrame({"date": [1], "open": [0.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]})
    est = VE.estimators_from_ohlcv(df)
    assert est["parkinson"].isna().iloc[0]          # non-positive open -> invalid


def test_ohlc_geometry_violations_are_invalid():
    # R-09: high must be the day's max and low its min; a bar where high < close (or low > open) is corrupt.
    df = pd.DataFrame({
        "date": [1, 2, 3, 4],
        "open":  [10.0, 10.0, 10.0, 10.0],
        "high":  [12.0,  9.5, 12.0, 12.0],   # row1: high < close (9.5<11) invalid
        "low":   [ 9.0,  9.0, 10.5, 11.0],   # row2: low > open (10.5>10) invalid; row3: low > open (11>10) invalid
        "close": [11.0, 11.0, 11.0, 11.0],
        "volume": [1, 1, 1, 1],
    })
    ok = VE.estimators_from_ohlcv(df)["ok"].to_numpy()
    assert ok.tolist() == [True, False, False, False]


def test_ohlc_float32_noise_within_tolerance_stays_valid():
    # R-09 regression (reviewer-found): float32 price storage puts high ~1e-7 below max(open,close); the raw
    # quality gate certifies these CLEAN (OHLC_RTOL=1e-5), so the estimator mask must NOT drop them.
    close = 7265.588
    df = pd.DataFrame({
        "date": [1],
        "open":  [7265.0],
        "high":  [close * (1 - 5e-7)],       # 1e-7-ish below close -> gate-clean float32 noise
        "low":   [7260.0],
        "close": [close],
        "volume": [1],
    })
    ok = VE.estimators_from_ohlcv(df)["ok"].to_numpy()
    assert ok.tolist() == [True]             # tolerance keeps it valid (exact compare would drop it)
    assert np.isfinite(VE.estimators_from_ohlcv(df)["parkinson"].iloc[0])


def test_panel_summary_robust_to_unsorted_duplicate_dates(tmp_path):
    # R-08: panel_summary must sort + dedup raw before the order-dependent rolling/overnight estimators,
    # so a shuffled-with-duplicates file yields the SAME summary as its clean, sorted version.
    rng = np.random.default_rng(3)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    c = 20.0 + np.cumsum(rng.normal(0, 0.2, n))
    span = np.abs(rng.normal(0, 0.3, n))
    clean = pd.DataFrame({"date": dates, "open": c, "high": c + span, "low": c - span,
                          "close": c, "volume": rng.integers(1e5, 1e6, n)})
    d_clean = tmp_path / "clean"; d_mess = tmp_path / "mess"; d_clean.mkdir(); d_mess.mkdir()
    clean.to_csv(d_clean / "AAA_ohlcv.csv", index=False)
    messy = pd.concat([clean, clean.iloc[[10, 20]]], ignore_index=True).sample(frac=1.0, random_state=9)
    messy.to_csv(d_mess / "AAA_ohlcv.csv", index=False)
    s_clean = VE.panel_summary("synthetic", d_clean)["per_ticker"]
    s_mess = VE.panel_summary("synthetic", d_mess)["per_ticker"]
    for e in VE.EST:
        assert abs(float(s_clean[f"{e}_mean"].iloc[0]) - float(s_mess[f"{e}_mean"].iloc[0])) < 1e-9


def test_panel_summary_smoke(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    for tk in ("AAA", "BBB"):
        c = 20.0 + np.cumsum(rng.normal(0, 0.2, n))
        span = np.abs(rng.normal(0, 0.3, n))
        pd.DataFrame({"date": dates, "open": c, "high": c + span, "low": c - span,
                      "close": c, "volume": rng.integers(1e5, 1e6, n)}).to_csv(
            tmp_path / f"{tk}_ohlcv.csv", index=False)
    s = VE.panel_summary("synthetic", tmp_path)
    assert len(s["per_ticker"]) == 2
    for e in VE.EST:
        assert f"{e}_zero_frac" in s["per_ticker"].columns
    out = VE.render_html([s], tmp_path / "est.html")
    html = Path(out).read_text(encoding="utf-8")
    assert "estimator" in html.lower() and "rescue" in html.lower()
