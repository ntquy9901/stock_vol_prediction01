"""Tests for the daily volatility-estimator comparison (scripts/eda/volatility_estimators.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volatility_estimators as VE  # noqa: E402


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


def test_invalid_prices_are_nan():
    df = pd.DataFrame({"date": [1], "open": [0.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]})
    est = VE.estimators_from_ohlcv(df)
    assert est["parkinson"].isna().iloc[0]          # non-positive open -> invalid


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
