"""Real-data-sample smoke per market: read a small slice of REAL raw data, assert the build runs + is sane."""
from __future__ import annotations

import numpy as np
import pytest

import enrich

MARKETS = list(enrich.PRICE_DIRS)


@pytest.mark.parametrize("market", MARKETS)
def test_real_market_slice_builds(market):
    price_dir = enrich.PRICE_DIRS[market]
    if not price_dir.exists() or not list(price_dir.glob("*_ohlcv.csv")):  # pragma: no cover - env guard
        pytest.skip(f"no real raw for {market} on this machine")
    summary = enrich.build_market(market, out_root=None, write=False, limit=2)
    assert summary["n_tickers"] >= 1
    assert summary["rows_out"] > 100
    # market_pk finite and non-negative where defined
    assert np.isfinite(summary["market_pk"]["mean"])
    assert summary["market_pk"]["min"] >= 0.0
    # parkinson mean is a small positive variance
    assert 0 < summary["estimator_mean"]["parkinson_variance"] < 1.0


def test_real_vn30_single_ticker_columns_and_sanity():
    price_dir = enrich.PRICE_DIRS["vn30"]
    files = sorted(price_dir.glob("*_ohlcv.csv"))
    if not files:  # pragma: no cover - env guard
        pytest.skip("no real vn30 raw")
    import pandas as pd
    out, rej, _ = enrich.build_ticker(pd.read_csv(files[0]))
    assert list(out.columns) == enrich.ENRICHED_COLUMNS
    pk = out["parkinson_variance"].to_numpy(float)
    fin = pk[np.isfinite(pk)]
    assert (fin >= 0).all()
    assert out["date"].is_monotonic_increasing
