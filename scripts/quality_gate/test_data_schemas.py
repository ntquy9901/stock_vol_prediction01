"""Adjacent tests for the enriched Pandera schema (data_schemas.validate_enriched / _check_enriched_file)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - test import bootstrap
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.quality_gate.data_schemas import (  # noqa: E402
    INVALID,
    MISSING,
    VALID,
    _check_enriched_file,
    validate_enriched,
)


def _good_frame(n: int = 8) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": np.linspace(100.0, 110.0, n),
        "high": np.linspace(101.0, 112.0, n),
        "low": np.linspace(99.0, 108.0, n),
        "close": np.linspace(100.5, 111.0, n),
        "volume": np.linspace(1e5, 2e5, n),
        "parkinson_variance": np.linspace(0.001, 0.02, n),
        "garman_klass_variance": np.linspace(0.001, 0.02, n),
        "rogers_satchell_variance": np.linspace(0.001, 0.02, n),
        "yang_zhang_n20": [np.nan] * 3 + list(np.linspace(0.001, 0.02, n - 3)),
        "log_range": np.linspace(0.0, 0.1, n),
        "daily_return": np.linspace(-0.05, 0.05, n),
        "har_daily": np.linspace(0.001, 0.02, n),
        "har_weekly": [np.nan] * 2 + list(np.linspace(0.001, 0.02, n - 2)),
        "har_monthly": [np.nan] * 4 + list(np.linspace(0.001, 0.02, n - 4)),
        "market_pk": np.linspace(0.001, 0.02, n),
        "volume_zscore_22": np.linspace(-2, 2, n),
        "volume_zscore_20": np.linspace(-2, 2, n),
        "dirty_flag": [False] * n,
        "cleaning_applied": ["none"] * n,
        "zero_range_flag": [False] * n,
        "zero_volume_flag": [False] * n,
    })


def _write(tmp_path, market, ticker, df):
    d = tmp_path / market
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{ticker}.csv"
    df.to_csv(p, index=False)
    return p


def test_valid_enriched_file(tmp_path):
    p = _write(tmp_path, "vn30", "AAA", _good_frame())
    name, status, _ = _check_enriched_file(p)
    assert status == VALID and name == "enriched:vn30/AAA.csv"


def test_read_failure(tmp_path):
    d = tmp_path / "vn30"
    d.mkdir(parents=True)
    p = d / "BAD.csv"
    p.write_text("this,is\nnot,valid,csv,shape\n\"unterminated", encoding="utf-8")
    name, status, detail = _check_enriched_file(p)
    assert status == INVALID


def test_negative_estimator_fails_schema(tmp_path):
    df = _good_frame()
    df.loc[2, "parkinson_variance"] = -1.0
    p = _write(tmp_path, "vn30", "NEG", df)
    _, status, detail = _check_enriched_file(p)
    assert status == INVALID and "schema" in detail


def test_valid_enriched_file_has_ohlcv(tmp_path):
    df = _good_frame()
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns
    p = _write(tmp_path, "vn30", "OHLCV", df)
    _, status, _ = _check_enriched_file(p)
    assert status == VALID


def test_nonpositive_price_fails_schema(tmp_path):
    df = _good_frame()
    df.loc[2, "close"] = 0.0        # cleaned OHLC must be strictly positive
    p = _write(tmp_path, "vn30", "ZERO", df)
    _, status, detail = _check_enriched_file(p)
    assert status == INVALID and "schema" in detail


def test_high_lt_low_geometry_fails(tmp_path):
    df = _good_frame()
    df.loc[3, "high"] = df.loc[3, "low"] - 1.0     # violate high >= low
    p = _write(tmp_path, "vn30", "GEO", df)
    _, status, detail = _check_enriched_file(p)
    assert status == INVALID and "high < low" in detail


def test_enriched_without_ohlc_skips_geometry_check(tmp_path):
    # OHLC columns are optional; a frame lacking them must still validate (geometry check skipped).
    df = _good_frame().drop(columns=["open", "high", "low", "close", "volume"])
    p = _write(tmp_path, "vn30", "NOOHLC", df)
    _, status, _ = _check_enriched_file(p)
    assert status == VALID


def test_unparseable_date(tmp_path):
    df = _good_frame()
    df.loc[1, "date"] = "not-a-date"
    p = _write(tmp_path, "vn30", "DT", df)
    _, status, detail = _check_enriched_file(p)
    assert status == INVALID and "unparseable" in detail


def test_non_monotonic_dates(tmp_path):
    df = _good_frame()
    df.loc[0, "date"], df.loc[1, "date"] = df.loc[1, "date"], df.loc[0, "date"]
    p = _write(tmp_path, "vn30", "MONO", df)
    _, status, detail = _check_enriched_file(p)
    assert status == INVALID and "monotonic" in detail


def test_duplicate_dates(tmp_path):
    df = _good_frame()
    df.loc[2, "date"] = df.loc[1, "date"]
    p = _write(tmp_path, "vn30", "DUP", df)
    _, status, detail = _check_enriched_file(p)
    assert status == INVALID and "duplicate" in detail


def test_weekend_date(tmp_path):
    df = _good_frame()
    df.loc[7, "date"] = "2020-01-11"     # a Saturday, still after row 6 -> monotonic + unique
    p = _write(tmp_path, "vn30", "WKND", df)
    _, status, detail = _check_enriched_file(p)
    assert status == INVALID and "weekday" in detail


def test_validate_enriched_missing_dir(tmp_path):
    res = validate_enriched(tmp_path / "nope")
    assert res == [("enriched_dir", MISSING, res[0][2])]
    assert res[0][1] == MISSING


def test_validate_enriched_no_csvs(tmp_path):
    (tmp_path).mkdir(exist_ok=True)
    res = validate_enriched(tmp_path)
    assert res[0][1] == MISSING and "no enriched" in res[0][2]


def test_validate_enriched_scans_and_skips_rejections(tmp_path):
    _write(tmp_path, "vn30", "AAA", _good_frame())
    # a rejections sidecar must NOT be validated as a price series
    (tmp_path / "vn30" / "AAA_rejections.csv").write_text("date,reason\n2020-01-01,naninf\n", encoding="utf-8")
    res = validate_enriched(tmp_path)
    assert len(res) == 1 and res[0][1] == VALID
