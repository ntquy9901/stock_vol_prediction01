"""Test-vs-published-formula for each estimator column (INDEPENDENT recompute; not reusing enrich's path)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import enrich
from _synth import clean_frame

_LN2 = np.log(2.0)
_N = 20                                  # yang_zhang window (matches yang_zhang_n20)
_K = 0.34 / (1.34 + (_N + 1) / (_N - 1))


def _independent_estimators(df: pd.DataFrame) -> dict:
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    ln_hl = np.log(h / lo)
    park = ln_hl ** 2 / (4 * _LN2)
    gk = 0.5 * ln_hl ** 2 - (2 * _LN2 - 1) * np.log(c / o) ** 2
    rs = np.clip(np.log(h / c) * np.log(h / o) + np.log(lo / c) * np.log(lo / o), 0.0, None)
    return {"parkinson": park, "garman_klass": np.clip(gk, 0, None), "rogers_satchell": rs}


def _independent_yang_zhang(df: pd.DataFrame, cap: float = 0.20) -> np.ndarray:
    o = df["open"].to_numpy(float); c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float); lo = df["low"].to_numpy(float)
    prev_c = np.concatenate([[np.nan], c[:-1]])
    r_on = np.clip(np.log(o / prev_c), -cap, cap)
    ln_co = np.log(c / o)
    rs = np.clip(np.log(h / c) * np.log(h / o) + np.log(lo / c) * np.log(lo / o), 0.0, None)
    var_on = pd.Series(r_on).rolling(_N).var()
    var_oc = pd.Series(ln_co).rolling(_N).var()
    mean_rs = pd.Series(rs).rolling(_N).mean()
    return np.clip((var_on + _K * var_oc + (1 - _K) * mean_rs).to_numpy(), 0.0, None)


def test_parkinson_gk_rs_match_published_formula():
    df = clean_frame(n=50, seed=3)
    out, _, _ = enrich.build_ticker(df)
    exp = _independent_estimators(df)
    assert np.allclose(out["parkinson_variance"], exp["parkinson"], atol=1e-12)
    assert np.allclose(out["garman_klass_variance"], exp["garman_klass"], atol=1e-12)
    assert np.allclose(out["rogers_satchell_variance"], exp["rogers_satchell"], atol=1e-12)


def test_parkinson_equals_logrange_identity():
    df = clean_frame(n=40, seed=4)
    out, _, _ = enrich.build_ticker(df)
    assert np.allclose(out["parkinson_variance"], out["log_range"] ** 2 / (4 * _LN2), atol=1e-12)


def test_yang_zhang_n20_matches_windowed_formula():
    df = clean_frame(n=50, seed=5)
    out, _, _ = enrich.build_ticker(df)
    exp = _independent_yang_zhang(df)
    got = out["yang_zhang_n20"].to_numpy(float)
    fin = np.isfinite(exp) & np.isfinite(got)
    assert fin.sum() > 10
    assert np.allclose(got[fin], exp[fin], atol=1e-10)


def test_har_terms_are_trailing_means_of_parkinson():
    df = clean_frame(n=60, seed=6)
    out, _, _ = enrich.build_ticker(df)
    pk = out["parkinson_variance"]
    assert np.allclose(out["har_daily"], pk, atol=1e-12)
    assert np.allclose(out["har_weekly"].dropna(),
                       pk.rolling(5, min_periods=5).mean().dropna(), atol=1e-12)
    assert np.allclose(out["har_monthly"].dropna(),
                       pk.rolling(22, min_periods=22).mean().dropna(), atol=1e-12)
