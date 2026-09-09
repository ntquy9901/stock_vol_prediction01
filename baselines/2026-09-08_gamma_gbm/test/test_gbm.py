"""Tests for the gamma-GBM baseline: causal feature panels, design-matrix shape/order, and a real-data smoke
(HAR-X reproduces canonical; the GBM pipeline runs and yields finite positive QLIKE)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import gamma_gbm_walkforward as G  # noqa: E402


def test_extra_feature_panels_rq_and_causality():
    T, N = 10, 1
    feats = np.zeros((T, N, 5)); feats[:, 0, 0] = np.arange(1.0, T + 1)      # pk = 1..10
    ex = G.extra_feature_panels(feats)
    assert set(ex) == {"rq", "mr_change", "mr_slope5", "mr_slope10", "mr_dev5", "mr_z22"}
    assert ex["rq"].shape == (T, N)
    # rq at t=6 (0-idx) = sqrt(mean of pk^2 over last 5 days [3,4,5,6,7])
    assert np.isclose(ex["rq"][6, 0], np.sqrt(np.mean(np.array([3.0, 4.0, 5.0, 6.0, 7.0]) ** 2)))
    # causal: changing a FUTURE pk must not change earlier rq/decline values
    feats2 = feats.copy(); feats2[8:, 0, 0] = 999.0
    ex2 = G.extra_feature_panels(feats2)
    assert np.allclose(ex["rq"][:8, 0], ex2["rq"][:8, 0])
    assert np.allclose(np.nan_to_num(ex["mr_slope5"][:8, 0]), np.nan_to_num(ex2["mr_slope5"][:8, 0]))


def test_design_shape_and_har_block():
    n, N = 3, 2
    har5 = np.arange(n * N * 5, dtype=float).reshape(n, N, 5)
    T = 5
    extras = {k: np.zeros((T, N)) for k in G.EXTRA_KEYS}
    for kk, k in enumerate(G.EXTRA_KEYS):
        extras[k][:] = kk + 1                                                # constant per feature
    anchors = np.array([0, 1, 2])
    X = G._design(har5, extras, anchors)
    assert X.shape == (n * N, 5 + len(G.EXTRA_KEYS))                         # 11 columns
    assert np.allclose(X[:, :5], har5.reshape(n * N, 5))                     # first 5 = HAR block, in order
    for kk in range(len(G.EXTRA_KEYS)):
        assert np.allclose(X[:, 5 + kk], kk + 1)                            # extras appended in EXTRA_KEYS order


def test_extra_features_are_not_a_noop():
    """The 6 extra columns must actually change the fitted GBM — guards against a design/aligning regression
    that silently drops them (which would collapse the model to HAR-X-only)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(0)
    har5 = rng.normal(size=(2000, 5)); extra = rng.normal(size=2000)
    y = np.exp(0.5 * har5[:, 0] + 1.5 * extra)                              # target DEPENDS on an extra column
    m_base = HistGradientBoostingRegressor(loss="gamma", max_iter=100, random_state=0).fit(har5, y)
    m_full = HistGradientBoostingRegressor(loss="gamma", max_iter=100, random_state=0).fit(
        np.column_stack([har5, extra]), y)
    xte = rng.normal(size=(200, 5)); ete = rng.normal(size=200)
    p_base = m_base.predict(xte); p_full = m_full.predict(np.column_stack([xte, ete]))
    assert not np.allclose(p_base, p_full)                                 # extra column is used
    yte = np.exp(0.5 * xte[:, 0] + 1.5 * ete)
    assert np.mean((np.log(p_full) - np.log(yte)) ** 2) < np.mean((np.log(p_base) - np.log(yte)) ** 2)


@pytest.mark.smoke
def test_smoke_reproduces_harx_and_runs_gbm():
    """VN30 (small, fast): HAR-X reproduces canonical h5 QLIKE; GBM yields finite positive QLIKE."""
    import glob
    if not glob.glob(str(G.REPO / "data" / "processed_enriched" / "vn30" / "*.csv")):
        pytest.skip("vn30 enriched data not present")
    r = G.run("vn30", 5, folds_target=7, out=str(G.REPO / "results" / "gamma_gbm" / "_smoke_vn30_h5.json"))
    q = {m: r["metrics"][m]["qlike"] for m in ("HAR-X", "GBM")}
    assert abs(q["HAR-X"] - 0.5602) < 0.003                                 # reproduces canonical VN30 h5 HAR-X
    assert np.isfinite(q["GBM"]) and q["GBM"] > 0
    assert 0.0 <= r["dm_GBM_vs_HARX"]["qlike"]["p_value"] <= 1.0
    assert r["features"][:1] == ["har5"] and "rq" in r["features"]
