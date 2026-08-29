"""Unit tests for the combined corr+lift edge (arXiv:2502.15813 sec 3.2).

The Pearson-rho and lift tests INDEPENDENTLY recompute the published formula on a tiny fixture (NOT reusing
the module's own code path) and match -- per CLAUDE.md "named method must use the published formula".
UNIQUE basenames (test_corrlift_edge.py) to avoid the pytest prepend-import duplicate-basename shadowing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import corrlift_edge as CL  # noqa: E402


# ------------------------- load_close_wide + daily_returns -------------------------

def _write_ohlcv(dirpath: Path, tk: str, dates, close):
    df = pd.DataFrame({"date": dates, "open": close, "high": close, "low": close,
                       "close": close, "volume": [1] * len(close)})
    df.to_csv(dirpath / f"{tk}_ohlcv.csv", index=False)


def test_load_close_wide_missing_file_is_all_nan(tmp_path):
    d = pd.date_range("2020-01-01", periods=4, freq="D")
    _write_ohlcv(tmp_path, "AAA", d, [10.0, 11.0, 12.0, 13.0])
    wide = CL.load_close_wide(["AAA", "ZZZ"], tmp_path)     # ZZZ has no file
    assert list(wide.columns) == ["AAA", "ZZZ"]
    assert wide["ZZZ"].isna().all()
    assert wide["AAA"].tolist() == [10.0, 11.0, 12.0, 13.0]


def test_load_close_wide_dedups_dates(tmp_path):
    d = ["2020-01-01", "2020-01-01", "2020-01-02"]
    _write_ohlcv(tmp_path, "AAA", d, [10.0, 99.0, 12.0])    # keep="last" -> 99.0 on 01-01
    wide = CL.load_close_wide(["AAA"], tmp_path)
    assert wide.loc[pd.Timestamp("2020-01-01"), "AAA"] == 99.0
    assert len(wide) == 2


def test_daily_returns_own_dates_pct_change():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    wide = pd.DataFrame({"A": [10.0, 11.0, np.nan, 13.2]}, index=idx)
    r = CL.daily_returns(wide)
    # own dates for A = {d0,d1,d3}; pct_change: d0 NaN, d1=(11-10)/10=0.1, d3=(13.2-11)/11=0.2
    assert np.isnan(r[0, 0])
    assert r[1, 0] == pytest.approx(0.1)
    assert np.isnan(r[2, 0])                                # off own date
    assert r[3, 0] == pytest.approx(0.2)


def test_daily_returns_too_few_points_all_nan():
    idx = pd.date_range("2020-01-01", periods=2, freq="D")
    wide = pd.DataFrame({"A": [10.0, np.nan]}, index=idx)   # only 1 finite point
    assert np.isnan(CL.daily_returns(wide)).all()


# ------------------------- Pearson rho (independent recompute) -------------------------

def test_pearson_matches_independent_formula():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(200, 3))
    r[:, 2] = 0.9 * r[:, 0] + 0.1 * rng.normal(size=200)    # col2 correlated with col0
    corr = CL.pearson_corr(r, min_overlap=50)
    # INDEPENDENT recompute of Pearson rho (Eq.3) WITHOUT calling the module -- mean-centred cov / std product
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            a, b = r[:, i], r[:, j]
            am, bm = a - a.mean(), b - b.mean()
            expected = float((am * bm).sum() / np.sqrt((am ** 2).sum() * (bm ** 2).sum()))
            assert corr[i, j] == pytest.approx(expected, abs=1e-9)
    assert corr[0, 2] == pytest.approx(corr[2, 0])          # symmetric


def test_pearson_min_overlap_and_constant_give_nan():
    r = np.full((120, 3), np.nan)
    r[:40, 0] = np.arange(40.0)                             # only 40 finite -> below min_overlap=100
    r[:, 1] = np.arange(120.0)
    r[:, 2] = 5.0                                            # constant series -> std 0
    corr = CL.pearson_corr(r, min_overlap=100)
    assert np.isnan(corr[0, 1])                             # < min_overlap
    assert np.isnan(corr[1, 2])                             # constant col2
    assert np.isnan(corr[0, 0])                             # diagonal untouched (NaN)


# ------------------------- move_events + lift (independent recompute) -------------------------

def test_move_events_threshold_is_train_median_abs():
    r = np.array([[0.01], [-0.02], [0.05], [np.nan], [-0.005]])
    event, valid = CL.move_events(r)
    absr = np.array([0.01, 0.02, 0.05, np.nan, 0.005])
    thr = np.median(absr[np.isfinite(absr)])               # median of {0.01,0.02,0.05,0.005}=0.015
    assert thr == pytest.approx(0.015)
    assert event[:, 0].tolist() == [False, True, True, False, False]  # |r|>0.015 strictly
    assert valid[:, 0].tolist() == [True, True, True, False, True]


def test_move_events_empty_column():
    r = np.full((3, 1), np.nan)
    event, valid = CL.move_events(r)
    assert not event.any() and not valid.any()


def test_lift_matches_independent_formula():
    # hand-built co-move events over 40 co-observed days for 2 stocks
    event = np.zeros((40, 2), dtype=bool)
    valid = np.ones((40, 2), dtype=bool)
    event[:10, 0] = True; event[:10, 1] = True              # 10 shared move days
    event[10:16, 0] = True                                  # 6 more solo moves for stock0
    event[16:20, 1] = True                                  # 4 more solo moves for stock1
    lift = CL.pairwise_lift(event, valid, min_pairs=30)
    # INDEPENDENT market-basket recompute: support = P(event), lift = P(i&j)/(P(i)P(j))
    si = event[:, 0].mean(); sj = event[:, 1].mean(); sij = (event[:, 0] & event[:, 1]).mean()
    expected = sij / (si * sj)
    assert lift[0, 1] == pytest.approx(expected)
    assert lift[0, 1] == pytest.approx(lift[1, 0])          # symmetric
    assert np.isnan(lift[0, 0])                             # diagonal untouched


def test_lift_min_pairs_and_zero_support_give_nan():
    event = np.zeros((40, 3), dtype=bool)
    valid = np.ones((40, 3), dtype=bool)
    valid[20:, 2] = False                                    # stock2 co-observed with others only 20 days (<30)
    event[:10, 0] = True                                    # stock1 never moves -> support 0
    lift = CL.pairwise_lift(event, valid, min_pairs=30)
    assert np.isnan(lift[0, 2])                             # < min_pairs
    assert np.isnan(lift[0, 1])                             # stock1 support 0 -> undefined


# ------------------------- build_corrlift_adjacency (combine + leakage) -------------------------

def _corrlift_wide(seed=1, n_days=400):
    """A 4-ticker close panel: A,B highly correlated + co-moving; C,D independent noise."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.02, size=n_days)
    ra = base + rng.normal(0, 0.001, size=n_days)
    rb = base + rng.normal(0, 0.001, size=n_days)           # B ~ A (corr AND lift fire)
    rc = rng.normal(0, 0.02, size=n_days)
    rd = rng.normal(0, 0.02, size=n_days)
    idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
    def px(r):
        return 10.0 * np.cumprod(1 + r)
    return pd.DataFrame({"A": px(ra), "B": px(rb), "C": px(rc), "D": px(rd)}, index=idx)


def test_build_adjacency_symmetry_selfloop_and_density():
    wide = _corrlift_wide()
    cutoff = wide.index[-1] + pd.Timedelta(days=1)          # all rows are train
    adj, diag = CL.build_corrlift_adjacency(wide, cutoff)
    assert adj.shape == (4, 4)
    assert adj.dtype == np.float32
    assert np.allclose(np.diag(adj), 1.0)                   # self-loop
    assert np.allclose(adj, adj.T)                          # symmetric
    assert (adj[np.triu_indices(4, 1)] >= 0).all() and (adj <= 1.0 + 1e-6).all()  # weights in [0,1]
    assert adj[0, 1] > 0                                    # A-B edge present (correlated + co-moving)
    assert diag["n_either_edges"] >= 1
    assert diag["n_corr_edges"] <= diag["n_either_edges"]
    assert diag["n_both_edges"] <= min(diag["n_corr_edges"], diag["n_lift_edges"])
    assert diag["n_nodes"] == 4 and diag["n_pairs"] == 6


def test_build_adjacency_empty_graph_when_thresholds_too_high():
    wide = _corrlift_wide()
    cutoff = wide.index[-1] + pd.Timedelta(days=1)
    adj, diag = CL.build_corrlift_adjacency(wide, cutoff, corr_thresh=0.999, lift_thresh=100.0)
    assert diag["n_either_edges"] == 0
    off = adj.copy(); np.fill_diagonal(off, 0.0)
    assert (off == 0).all()                                # only self-loops
    assert diag["n_singletons"] == 4
    assert np.allclose(np.diag(adj), 1.0)


def test_build_adjacency_is_train_only_leakage_frozen():
    wide = _corrlift_wide()
    cutoff = wide.index[len(wide) // 2]                     # split mid-panel
    adj_ref, _ = CL.build_corrlift_adjacency(wide, cutoff)
    # corrupt every POST-cutoff row with extreme values -> must NOT change the frozen (train-only) adjacency
    poisoned = wide.copy()
    poisoned.loc[poisoned.index >= cutoff] = 1e6
    adj_poison, _ = CL.build_corrlift_adjacency(poisoned, cutoff)
    assert np.array_equal(adj_ref, adj_poison)


def test_build_adjacency_lift_only_edge_uses_lift_strength():
    # A shared VOLATILITY factor drives both stocks' |return| identically, but the SIGN of each day's move
    # is independent -> the "notable move" (above-median-|return|) days coincide (lift fires) while the
    # signed Pearson rho ~ 0 (linear criterion does NOT fire). Isolates the non-linear lift criterion.
    rng = np.random.default_rng(7)
    n = 400
    common_vol = np.abs(rng.normal(0, 0.03, size=n)) + 1e-4     # shared magnitude both stocks feel
    ra = common_vol * rng.choice([-1.0, 1.0], size=n)          # random independent signs
    rb = common_vol * rng.choice([-1.0, 1.0], size=n)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    wide = pd.DataFrame({"A": 10.0 * np.cumprod(1 + ra), "B": 10.0 * np.cumprod(1 + rb)}, index=idx)
    cutoff = idx[-1] + pd.Timedelta(days=1)
    adj, diag = CL.build_corrlift_adjacency(wide, cutoff)
    assert diag["n_lift_edges"] >= 1                        # co-movement detected by lift
    assert diag["n_corr_edges"] == 0                       # signed correlation does NOT fire
    assert adj[0, 1] > 0                                    # edge present via the non-linear criterion only
