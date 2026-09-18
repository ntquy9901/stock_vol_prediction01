"""Unit + real-data tests for the earnings parquet data-quality guard (codifies the provenance audit)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from earnings_data_quality import _load, earnings_quality  # noqa: E402


def _clean_df():
    rows = []
    for tk in ("AAA", "BBB"):
        base = np.datetime64("2020-01-31")
        for i in range(6):
            rows.append({"ticker": tk, "earnings_date": base + np.timedelta64(91 * i, "D")})
    return pd.DataFrame(rows)


def test_clean_table_reports_no_issues():
    q = earnings_quality(_clean_df())
    assert q["schema_ok"] and q["n_null"] == 0 and q["n_dup_rows"] == 0 and q["n_dup_pairs"] == 0
    assert q["n_tickers"] == 2 and q["n_rows"] == 12
    assert q["pct_gap_quarterly"] == 100.0            # 91-day gaps are inside the quarterly band
    assert q["tickers_lt_min_history"] == 0


def test_detects_dup_null_and_short_history():
    df = _clean_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)               # a duplicate row
    df.loc[len(df)] = {"ticker": "CCC", "earnings_date": pd.NaT}        # a null date
    df.loc[len(df)] = {"ticker": "DDD", "earnings_date": pd.Timestamp("2021-01-01")}  # short-history ticker
    q = earnings_quality(df)
    assert q["n_dup_rows"] >= 1 and q["n_dup_pairs"] >= 1
    assert q["n_null"] == 1                                             # CCC's NaT
    assert q["tickers_lt_min_history"] >= 1                             # DDD (1 valid event; CCC dropped as NaT)


def test_non_quarterly_gaps_lower_the_regularity():
    base = np.datetime64("2020-01-31")
    dates = [base + np.timedelta64(30 * i, "D") for i in range(6)]      # monthly, not quarterly
    q = earnings_quality(pd.DataFrame({"ticker": "AAA", "earnings_date": dates}))
    assert q["pct_gap_quarterly"] == 0.0 and q["n_gaps"] == 5


def test_empty_table_safe():
    q = earnings_quality(pd.DataFrame({"ticker": [], "earnings_date": []}))
    assert q["n_rows"] == 0 and q["pct_gap_quarterly"] == 0.0 and q["date_min"] is None


# ---- real-data guard: the crawled parquets must stay clean (audit invariants) ----

@pytest.mark.parametrize("market", ["hose", "sp500"])
def test_real_parquets_are_clean(market):
    q = earnings_quality(_load(market))
    assert q["schema_ok"], "schema drifted from [ticker, earnings_date]"
    assert q["n_null"] == 0, "null/NaT earnings dates present"
    assert q["n_dup_pairs"] == 0, "duplicate (ticker, date) rows present"
    assert q["n_rows"] > 1000 and q["n_tickers"] > 300


def test_sp500_regular_hose_irregular():
    """Codifies the audit: SP500 dates are quarterly-regular; HOSE mixes filing types -> irregular."""
    sp = earnings_quality(_load("sp500"))
    ho = earnings_quality(_load("hose"))
    assert sp["pct_gap_quarterly"] > 90.0                               # audit: ~99%
    assert ho["pct_gap_quarterly"] < 60.0                              # audit: ~45%
    assert sp["pct_gap_quarterly"] > ho["pct_gap_quarterly"]
