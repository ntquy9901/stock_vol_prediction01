"""Tests for the HARQ walk-forward baseline: quarticity-feature formula + causality, OLS recovery, and a
real-data smoke that the delivered HAR-X is reproduced and HAR-X-Q does not do worse at h5."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import harq_walkforward as H  # noqa: E402


def test_quarticity_formula_and_shape():
    """harq[t,n] = pk[t,n] * sqrt(mean of pk^2 over the last QWIN days)."""
    T, N = 8, 2
    feats = np.zeros((T, N, 5))
    feats[:, 0, 0] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])   # node 0 daily pk
    feats[:, 1, 0] = 2.0                                                  # node 1 constant
    hq = H.quarticity_panel(feats)
    assert hq.shape == (T, N)
    # node 1 constant pk=2 -> rq=mean(4)=4 -> sqrt=2 -> hq = 2*2 = 4 everywhere
    assert np.allclose(hq[:, 1], 4.0)
    # node 0 at t=6 (0-indexed): last QWIN=5 days pk = [3,4,5,6,7], rq=mean of squares
    rq = np.mean(np.array([3.0, 4.0, 5.0, 6.0, 7.0]) ** 2)
    assert np.isclose(hq[6, 0], 7.0 * np.sqrt(rq))


def test_quarticity_is_causal():
    """Changing a FUTURE pk must not change the current harq value."""
    T, N = 10, 1
    feats = np.ones((T, N, 5)); feats[:, 0, 0] = np.arange(1.0, T + 1)
    hq_a = H.quarticity_panel(feats)
    feats2 = feats.copy(); feats2[7:, 0, 0] = 999.0                        # change only t>=7
    hq_b = H.quarticity_panel(feats2)
    assert np.allclose(hq_a[:7, 0], hq_b[:7, 0])                           # t<7 unchanged


def test_ols_predict_recovers_linear():
    rng = np.random.default_rng(0)
    x_tr = rng.normal(size=(200, 2)); y_tr = 3.0 * x_tr[:, 0] - 2.0 * x_tr[:, 1] + 0.5
    x_te = rng.normal(size=(50, 2)); y_true = 3.0 * x_te[:, 0] - 2.0 * x_te[:, 1] + 0.5
    pred = H._ols_predict(x_tr, y_tr, x_te, np.full(50, -1e9), (50,))       # low floor -> no clamping
    assert np.allclose(pred, y_true, atol=1e-8)


def test_ols_predict_applies_floor():
    x_tr = np.zeros((10, 1)); y_tr = np.full(10, -5.0)                      # fit predicts negative
    pred = H._ols_predict(x_tr, y_tr, np.zeros((3, 1)), np.full(3, 1e-8), (3,))
    assert np.all(pred >= 1e-8)                                             # floored to positive


def test_appended_harq_column_changes_predictions():
    """The HARQ column must actually alter the fit — guards against a masking/zeroing regression that would
    silently collapse HAR-X-Q back to HAR-X (the exact failure this baseline is claimed to avoid)."""
    rng = np.random.default_rng(1)
    h5_tr = rng.normal(size=(300, 5)); hq_tr = rng.normal(size=300)
    y_tr = 1.0 * h5_tr[:, 0] + 2.0 * hq_tr + 0.3                            # target DEPENDS on the HARQ column
    h5_te = rng.normal(size=(40, 5)); hq_te = rng.normal(size=40)
    lowfloor = np.full(40, -1e9)
    p_x = H._ols_predict(h5_tr, y_tr, h5_te, lowfloor, (40,))               # HAR-X (no HARQ)
    p_q = H._ols_predict(np.column_stack([h5_tr, hq_tr]), y_tr,
                         np.column_stack([h5_te, hq_te]), lowfloor, (40,))  # HAR-X-Q
    assert not np.allclose(p_x, p_q)                                        # column is active
    y_true = 1.0 * h5_te[:, 0] + 2.0 * hq_te + 0.3
    assert np.mean((p_q - y_true) ** 2) < np.mean((p_x - y_true) ** 2)      # and it helps when y~HARQ


@pytest.mark.smoke
def test_smoke_reproduces_harx_and_no_worse_h5():
    """Real-data smoke: HAR-X matches the canonical VN100 h5 QLIKE and HAR-X-Q is not worse."""
    import glob
    if not glob.glob(str(H.REPO / "data" / "processed_enriched" / "vn100" / "*.csv")):
        pytest.skip("vn100 enriched data not present")
    r = H.run("vn100", 5, folds_target=7, out=str(H.REPO / "results" / "harq" / "_smoke_vn100_h5.json"))
    q = {m: r["metrics"][m]["qlike"] for m in ("HAR-X", "HAR-X-Q")}
    assert abs(q["HAR-X"] - 0.5607) < 0.002                                 # canonical 7-fold HAR-X (exact repro)
    assert q["HAR-X-Q"] < q["HAR-X"]                                        # HARQ STRICTLY improves (column active)
    assert r["dm_HARXQ_vs_HARX"]["qlike"]["mean_diff"] < 0                  # and favours HAR-X-Q
    assert 0.0 <= r["dm_HARXQ_vs_HARX"]["qlike"]["p_value"] <= 1.0
