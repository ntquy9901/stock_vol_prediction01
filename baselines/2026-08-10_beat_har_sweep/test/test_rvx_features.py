"""Tests for the C4 HAR-RV-X range/overnight variance estimators (sigma^2 units)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_CODE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rvx_features import (  # noqa: E402
    compute_rvx_frame, garman_klass_variance, overnight_variance, rogers_satchell_variance,
)


def test_gk_zero_when_no_range_and_no_move():
    # O=H=L=C -> both terms zero
    assert garman_klass_variance(10.0, 10.0, 10.0, 10.0) == pytest.approx(0.0)


def test_rs_zero_when_flat():
    assert rogers_satchell_variance(10.0, 10.0, 10.0, 10.0) == pytest.approx(0.0)


def test_rs_matches_manual():
    o, h, low, c = 100.0, 105.0, 98.0, 102.0
    expected = (np.log(h / c) * np.log(h / o)) + (np.log(low / c) * np.log(low / o))
    assert rogers_satchell_variance(o, h, low, c) == pytest.approx(expected)


def test_overnight_is_squared_log_gap_nonnegative():
    assert overnight_variance(102.0, 100.0) == pytest.approx(np.log(102.0 / 100.0) ** 2)
    assert overnight_variance(98.0, 100.0) >= 0.0


def test_variance_scale_matches_parkinson_order():
    """On a realistic daily bar the RV-X terms are ~1e-4 (variance), same order as Parkinson."""

    o, h, low, c = 20.0, 20.3, 19.8, 20.1
    gk = garman_klass_variance(o, h, low, c)
    rs = rogers_satchell_variance(o, h, low, c)
    assert 1e-6 < abs(gk) < 1e-2 and 1e-6 < abs(rs) < 1e-2


def test_compute_frame_aligns_and_drops_first_overnight():
    frame = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "open": [10.0, 10.2, 10.1], "high": [10.5, 10.6, 10.3],
        "low": [9.8, 10.0, 9.9], "close": [10.2, 10.1, 10.2],
    })
    out = compute_rvx_frame(frame)
    assert list(out.columns) == ["date", "gk_variance", "rs_variance", "overnight_variance"]
    assert len(out) == 2  # first row dropped (no prior close for overnight)
    assert np.isfinite(out[["gk_variance", "rs_variance", "overnight_variance"]].to_numpy()).all()
    assert (out["overnight_variance"] >= 0).all()


def test_rejects_missing_columns():
    with pytest.raises(ValueError):
        compute_rvx_frame(pd.DataFrame({"date": ["2020-01-01"], "open": [1.0]}))
