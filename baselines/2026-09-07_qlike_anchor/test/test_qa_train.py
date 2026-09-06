"""Tests for the pure forecast/loss helpers of the QLIKE-anchor trainer."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import qa_train as QT  # noqa: E402


def test_node_floor_matches_formula():
    import pipeline_config as pc
    tm = np.array([1e-3, 2e-3])
    assert np.allclose(QT.node_floor(tm), pc.POS_FLOOR_FRAC * tm + pc.POS_FLOOR_EPS)


def test_zscore_forecast_denorms_and_floors():
    z = np.array([[0.0, 2.0]]); mean = np.array([1e-3, 1e-3]); std = np.array([1e-3, 1e-3])
    floor = np.array([1e-4, 1e-4])
    f = QT.zscore_forecast(z, mean, std, floor)
    assert np.allclose(f, [[1e-3, 3e-3]])                       # 0*std+mean=1e-3 ; 2*std+mean=3e-3
    f2 = QT.zscore_forecast(np.array([[-5.0]]), np.array([1e-3]), np.array([1e-3]), np.array([1e-4]))
    assert f2[0, 0] == 1e-4                                     # negative denorm floored


def test_anchor_forecast_z0_is_harx_and_clips():
    harx = np.array([1.27e-4, 2.0e-4])
    assert np.allclose(QT.anchor_forecast([0.0, 0.0], harx, clip=0.5), harx)   # z=0 -> HAR-X (no collapse)
    big_neg = QT.anchor_forecast([-10.0], np.array([1.27e-4]), clip=0.5)       # clipped to -0.5
    assert np.allclose(big_neg, 1.27e-4 * np.exp(-0.5))
    floored = QT.anchor_forecast([-10.0], np.array([1.27e-4]), clip=5.0, floor=np.array([1e-5]))
    assert floored[0] == 1e-5                                   # floor applied


def test_qlike_np_zero_at_equality_and_penalizes_under():
    y = np.array([1e-3, 2e-3])
    assert np.allclose(QT.qlike_np(y, y), 0.0)
    assert QT.qlike_np([1e-2], [1e-3])[0] > QT.qlike_np([1e-3], [1e-2])[0]   # under-forecast costs more


def test_residual_target_log_ratio():
    y = np.array([2e-3]); harx = np.array([1e-3])
    assert abs(QT.residual_target(y, harx, eps=0.0)[0] - np.log(2.0)) < 1e-9
