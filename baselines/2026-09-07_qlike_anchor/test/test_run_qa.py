"""Test the driver's model-selection parser (importing the driver pulls in delivered modules)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_qlike_anchor as R  # noqa: E402


def test_select_models_orders_and_rejects():
    assert R._select_models("VolGA,LSTM") == ("LSTM", "VolGA")   # fixed order
    assert R._select_models("LSTM") == ("LSTM",)
    for bad in ("", "MASTER", "LSTM,foo"):
        with pytest.raises(ValueError):
            R._select_models(bad)


def test_pool_maps_masked_test_cells():
    import numpy as np
    from types import SimpleNamespace
    D = SimpleNamespace(y_te=np.array([[1.0, 2.0]]), tmask_te=np.array([[True, False]]),
                        d_te=np.array(["2026-01-01"]), N=2)
    pooled = R._pool(np.array([[1.1, 9.9]]), D)                  # only masked cell (0, d0) kept
    assert set(pooled) == {(0, "2026-01-01")} and pooled[(0, "2026-01-01")] == (1.0, 1.1)
