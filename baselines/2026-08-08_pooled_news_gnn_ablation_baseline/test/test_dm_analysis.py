"""Per-observation QLIKE/SE loss extraction for the Diebold-Mariano comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from dm_analysis import paired_losses  # noqa: E402


def test_paired_losses_computes_qlike_and_squared_error_per_observation() -> None:
    """QLIKE_i = r - ln r - 1 (r = target/pred, clamped) and SE_i = (target-pred)**2."""
    g0_rows = [
        {"ticker_id": 0, "target_date": "d1", "target_raw": 10.0, "prediction_raw": 12.0},
        {"ticker_id": 1, "target_date": "d1", "target_raw": 100.0, "prediction_raw": 90.0},
    ]
    g1_rows = [
        {"ticker_id": 0, "target_date": "d1", "target_raw": 10.0, "prediction_raw": 11.0},
        {"ticker_id": 1, "target_date": "d1", "target_raw": 100.0, "prediction_raw": 95.0},
    ]

    out = paired_losses(g0_rows, g1_rows)

    np.testing.assert_allclose(out["qlike_g0"], [0.015654890127287935, 0.005750595453284824])
    np.testing.assert_allclose(out["qlike_g1"], [0.00440108889523394, 0.0013382845598179927])
    np.testing.assert_allclose(out["se_g0"], [4.0, 100.0])
    np.testing.assert_allclose(out["se_g1"], [1.0, 25.0])
    assert out["n"] == 2


def test_paired_losses_rejects_misaligned_observation_keys() -> None:
    """G0/G1 must be the identical (ticker_id, target_date) set in the same order."""
    g0_rows = [{"ticker_id": 0, "target_date": "d1", "target_raw": 10.0, "prediction_raw": 12.0}]
    g1_rows = [{"ticker_id": 9, "target_date": "d1", "target_raw": 10.0, "prediction_raw": 11.0}]

    with pytest.raises(ValueError, match="aligned"):
        paired_losses(g0_rows, g1_rows)


def test_paired_losses_rejects_mismatched_targets() -> None:
    """Same (id,date) must carry the same raw target across G0 and G1 (same eval set)."""
    g0_rows = [{"ticker_id": 0, "target_date": "d1", "target_raw": 10.0, "prediction_raw": 12.0}]
    g1_rows = [{"ticker_id": 0, "target_date": "d1", "target_raw": 11.0, "prediction_raw": 11.0}]

    with pytest.raises(ValueError, match="target"):
        paired_losses(g0_rows, g1_rows)
