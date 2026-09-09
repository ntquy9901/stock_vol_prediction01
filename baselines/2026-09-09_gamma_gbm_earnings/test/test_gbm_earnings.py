"""Tests for the earnings + calm-calibration logic (pure helpers; the SP500-only walk-forward `run` is a
pragma-no-cover driver validated by the committed results JSON)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import gbm_earnings_walkforward as E  # noqa: E402


def test_dist_to_earnings_nearest_and_cap():
    ed = np.array(["2025-01-10", "2025-04-10"], dtype="datetime64[D]")
    tgt = np.array(["2025-01-08", "2025-01-10", "2025-02-01", "2025-12-31"], dtype="datetime64[D]")
    d = E._dist(tgt, ed)
    assert d[0] == 2 and d[1] == 0                       # 2 days before / on the earnings day
    assert d[2] == min(22, 60)                           # ~22 days to Jan-10 (nearest)
    assert d[3] == E.CAP                                 # far beyond any earnings -> capped
    assert np.all(E._dist(tgt, None) == E.CAP)           # no earnings -> all capped


def test_earnings_panels_ramp_and_soon():
    panel = types.SimpleNamespace(
        target_dates=np.array(["2025-01-08", "2025-01-10", "2025-02-20"], dtype="datetime64[D]"),
        N=1, tickers=["ACB"])
    earn = {"ACB": np.array(["2025-01-10"], dtype="datetime64[D]")}
    p = E.earnings_panels(panel, earn)
    # dist = [2, 0, 41]; prox = max(0, 1 - dist/RAMP=5): [0.6, 1.0, 0.0]; soon(<=3) = [1,1,0]
    assert np.allclose(p["earn_prox"][:, 0], [0.6, 1.0, 0.0])
    assert np.allclose(p["earn_soon"][:, 0], [1.0, 1.0, 0.0])


def test_design_column_counts():
    n, N = 2, 3
    har5 = np.zeros((n, N, 5)); extras = {k: np.zeros((40, N)) for k in E.G.EXTRA_KEYS}
    earn = {k: np.zeros((40, N)) for k in E.EARN_KEYS}
    anchors = np.array([30, 31]); pos = np.array([0, 1])
    assert E._design(har5, extras, earn, anchors, pos, False).shape == (n * N, 11)   # base
    assert E._design(har5, extras, earn, anchors, pos, True).shape == (n * N, 13)    # + earn_prox, earn_soon


def test_design_earnings_aligned_by_position_not_dateindex():
    """Regression for the index-space bug: earnings must be pulled by POSITIONAL index (pos), not the
    date-index anchors (which start ~30). Distinct value per (position, node) must land on row pos*N+node."""
    n, N = 2, 3
    har5 = np.zeros((n, N, 5)); extras = {k: np.zeros((40, N)) for k in E.G.EXTRA_KEYS}
    A = 40
    ep = np.arange(A * N, dtype=float).reshape(A, N)                    # earn_prox distinct per (pos,node)
    earn = {"earn_prox": ep, "earn_soon": np.zeros((A, N))}
    anchors = np.array([30, 31]); pos = np.array([0, 1])               # date-index != position
    X = E._design(har5, extras, earn, anchors, pos, True)
    # earn_prox is the 12th column (index 11 = after 5 HAR + 6 extras); row i*N+j -> ep[pos[i], j]
    for i in range(n):
        for j in range(N):
            assert X[i * N + j, 11] == ep[pos[i], j]                    # positional alignment (not ep[anchors[i]])


def test_calibrate_is_qlike_optimal_multiplier():
    """For QLIKE, argmin_c QLIKE(y, c*f) = mean(y/f). A forecast 2x too high should be shrunk ~0.5x."""
    rng = np.random.default_rng(0)
    yv = np.exp(rng.normal(-8, 0.1, 4000)); pv = 2.0 * yv                      # forecast 2x realized
    pt = 2.0 * np.exp(rng.normal(-8, 0.1, 1000))
    cal = E._calibrate(pv, yv, pt, np.full(1000, 1e-12))
    assert 0.4 < np.median(cal / pt) < 0.6                                     # multiplier ~0.5 (correct 2x bias)


def test_calibrate_respects_floor_and_clip():
    yv = np.full(200, 1e-6); pv = np.full(200, 1e-6); pt = np.full(50, 1e-6)
    cal = E._calibrate(pv, yv, pt, np.full(50, 1e-3))
    assert np.all(cal >= 1e-3)                                                 # floored


def test_earn_by_ticker_groups_and_sorts(tmp_path, monkeypatch):
    import pandas as pd
    p = tmp_path / "e.parquet"
    pd.DataFrame({"ticker": ["B", "A", "A"],
                  "earnings_date": pd.to_datetime(["2025-05-01", "2025-04-01", "2025-01-01"])}).to_parquet(p)
    monkeypatch.setattr(E, "EARN_PARQUET", p)
    d = E._earn_by_ticker()
    assert set(d) == {"A", "B"}
    assert list(d["A"]) == list(np.array(["2025-01-01", "2025-04-01"], dtype="datetime64[D]"))  # sorted


def test_gbm_fit_predict_shapes():
    rng = np.random.default_rng(0)
    xtr = rng.normal(size=(500, 3)); ytr = np.exp(0.5 * xtr[:, 0])            # positive for gamma loss
    xa = rng.normal(size=(40, 3)); xb = rng.normal(size=(10, 3))
    pa, pb = E._gbm_fit_predict(xtr, ytr, xa, xb)
    assert pa.shape == (40,) and pb.shape == (10,)
