"""Plan section 65: Parkinson formula + no-negative-PK tests, and real-data cross-check."""

import numpy as np
import pandas as pd
import pytest

from graph_eda import parkinson
from graph_eda.io_data import load_ticker
from graph_eda.leakage import assert_no_negative_pk


def test_parkinson_formula_known_values():
    # closed-form: (ln(H/L))^2 / (4 ln 2)
    high, low = 4.83, 3.83
    expected = (np.log(high / low) ** 2) / (4 * np.log(2))
    assert parkinson.parkinson_var(np.array([high]), np.array([low]))[0] == pytest.approx(
        expected
    )
    assert parkinson.parkinson_vol(np.array([high]), np.array([low]))[0] == pytest.approx(
        np.sqrt(expected)
    )


def test_parkinson_var_is_variance_matches_processed():
    # data/processed parkinson_volatility is stored as VARIANCE (sigma^2); confirm.
    raw = load_ticker("data/raw/prices/ACB_ohlcv.csv")
    proc = pd.read_csv("data/processed/ACB_processed.csv", parse_dates=["date"])
    raw = raw.assign(pk_var=parkinson.parkinson_var(raw["high"], raw["low"]))
    merged = raw.merge(proc, on="date", how="inner").dropna(
        subset=["pk_var", "parkinson_volatility"]
    )
    assert len(merged) > 1000
    diff = (merged["pk_var"] - merged["parkinson_volatility"]).abs()
    assert diff.median() < 1e-6  # variance formula reproduces processed column


def test_no_negative_pk_on_real_slice():
    raw = load_ticker("data/raw/prices/VNM_ohlcv.csv")
    pk = parkinson.parkinson_var(raw["high"], raw["low"])
    assert_no_negative_pk(pk)


def test_build_features_leakage_safe_shift():
    d = load_ticker("data/raw/prices/FPT_ohlcv.csv").head(60)
    f = parkinson.build_features(d)
    # pk_lag1 at row t equals pk_var at row t-1 (uses past only)
    assert f["pk_lag1"].iloc[5] == pytest.approx(f["pk_var"].iloc[4])
    # rolling means need a full trailing window -> first rows are NaN (no future fill)
    assert np.isnan(f["pk_mean_22"].iloc[10])
