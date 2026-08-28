"""Smoke tests for scripts/etl_hose_hnx.py pure helpers (no network / heavy deps)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import etl_hose_hnx as ETL  # noqa: E402


def test_ticker_of_strips_suffix():
    assert ETL._ticker_of(Path("FPT_ohlcv.csv")) == "FPT"
    assert ETL._ticker_of(Path("/data/raw/ABC_ohlcv.csv")) == "ABC"


def test_leading_flat_run_counts_leading_H_equals_L():
    assert ETL._leading_flat_run(np.array([2.0, 3.0]), np.array([1.0, 2.0])) == 0     # no leading flat
    assert ETL._leading_flat_run(np.array([1.0, 1.0, 3.0]), np.array([1.0, 1.0, 2.0])) == 2  # 2 leading flat
    assert ETL._leading_flat_run(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 2     # all flat
    assert ETL._leading_flat_run(np.array([]), np.array([])) == 0                     # empty
