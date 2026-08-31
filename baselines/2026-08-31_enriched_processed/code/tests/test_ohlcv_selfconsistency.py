"""OHLCV cleaned columns present + row-level self-consistency with parkinson_variance + geometry."""
from __future__ import annotations

import numpy as np

import enrich
from _synth import clean_frame, dirty_frame

_LN2 = np.log(2.0)
_OHLCV = ["open", "high", "low", "close", "volume"]


def test_ohlcv_columns_present_and_positioned_after_date():
    out, _, _ = enrich.build_ticker(clean_frame(n=40, seed=3))
    for col in _OHLCV:
        assert col in out.columns
    # positioned right after date, before parkinson_variance
    assert list(out.columns[:6]) == ["date"] + _OHLCV
    assert list(out.columns) == enrich.ENRICHED_COLUMNS


def test_ohlcv_finite_and_positive_on_valid_bars():
    out, _, _ = enrich.build_ticker(clean_frame(n=40, seed=3))
    for col in _OHLCV:
        v = out[col].to_numpy(float)
        assert np.isfinite(v).all()
    for col in ("open", "high", "low", "close"):
        assert (out[col].to_numpy(float) > 0).all()
    assert (out["volume"].to_numpy(float) >= 0).all()


def test_parkinson_self_consistent_with_row_own_high_low():
    # on CLEAN bars the row's own high/low reproduce parkinson_variance exactly (the estimator input).
    out, _, _ = enrich.build_ticker(clean_frame(n=50, seed=7))
    clean_rows = ~out["dirty_flag"].to_numpy(bool)
    h = out["high"].to_numpy(float)[clean_rows]
    lo = out["low"].to_numpy(float)[clean_rows]
    pk = out["parkinson_variance"].to_numpy(float)[clean_rows]
    recomputed = np.log(h / lo) ** 2 / (4 * _LN2)
    assert np.allclose(pk, recomputed, atol=1e-12)


def test_cleaned_geometry_high_ge_low_and_open_close_bracketed():
    # non-dropped rows carry cleaned geometry: high>=low and open/close within [low, high].
    out, _, _ = enrich.build_ticker(dirty_frame())
    h = out["high"].to_numpy(float)
    lo = out["low"].to_numpy(float)
    o = out["open"].to_numpy(float)
    c = out["close"].to_numpy(float)
    fin = np.isfinite(h) & np.isfinite(lo) & np.isfinite(o) & np.isfinite(c)
    assert (h[fin] >= lo[fin]).all()
    assert (o[fin] >= lo[fin] - 1e-9).all() and (o[fin] <= h[fin] + 1e-9).all()
    assert (c[fin] >= lo[fin] - 1e-9).all() and (c[fin] <= h[fin] + 1e-9).all()


def test_ohlcv_equals_cleaned_input_on_clean_frame():
    # a fully-clean frame is unmodified by ETL -> enriched OHLCV == raw input row-for-row.
    raw = clean_frame(n=30, seed=11)
    out, _, _ = enrich.build_ticker(raw)
    assert len(out) == len(raw)
    for col in ("open", "high", "low", "close"):
        assert np.allclose(out[col].to_numpy(float), raw[col].to_numpy(float), atol=1e-12)
    assert np.allclose(out["volume"].to_numpy(float), raw["volume"].to_numpy(float), atol=1e-12)
