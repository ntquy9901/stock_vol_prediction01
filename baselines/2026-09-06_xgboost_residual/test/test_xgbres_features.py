"""Causal feature builders: backward-looking windows, no look-ahead, per-ticker alignment, graph orientation."""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))

import xgb_features as F  # noqa: E402


@dataclass
class FakePanel:
    tickers: list
    dates: object
    pk: np.ndarray
    feats: np.ndarray

    @property
    def N(self):
        return len(self.tickers)


def _panel(t=80, n=3, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=t, freq="B")
    pk = np.abs(rng.normal(0.02, 0.01, size=(t, n)))
    feats = np.zeros((t, n, 5))
    feats[:, :, 0] = pk
    feats[:, :, 1] = pk * 0.9        # har_weekly stand-in
    feats[:, :, 2] = pk * 0.8        # har_monthly stand-in
    feats[:, :, 3] = np.nanmean(pk, axis=1, keepdims=True)   # market_pk (shared)
    feats[:, :, 4] = rng.normal(size=(t, n))                 # volume_zscore
    return FakePanel(["A", "B", "C"][:n], dates, pk, feats)


def test_pk_lag_and_rollmean_are_backward_looking():
    p = _panel()
    ret = np.zeros_like(p.pk)
    Fsm, names, groups = F.build_sm_features(p, ret, np.zeros_like(p.pk), p.feats[:, :, 4])
    lag1 = names.index("pk_lag1")
    rm5 = names.index("pk_rollmean5")
    j, t = 0, 40
    assert Fsm[t, j, lag1] == np.float32(p.pk[t - 1, j])
    np.testing.assert_allclose(Fsm[t, j, rm5], p.pk[t - 4:t + 1, j].mean(), rtol=1e-5)


def test_no_lookahead_future_change_leaves_feature_unchanged():
    p = _panel()
    ret = np.zeros_like(p.pk)
    F0, names, _ = F.build_sm_features(p, ret, np.zeros_like(p.pk), p.feats[:, :, 4])
    p2 = _panel()
    p2.pk[60:] *= 3.0                       # perturb only the future
    p2.feats[60:, :, 0] = p2.pk[60:]
    F1, _, _ = F.build_sm_features(p2, ret, np.zeros_like(p2.pk), p2.feats[:, :, 4])
    # every stock feature at t=40 (< 60) must be identical
    np.testing.assert_allclose(np.nan_to_num(F0[40]), np.nan_to_num(F1[40]), rtol=1e-6, atol=1e-9)


def test_market_features_are_cross_sectional_at_t():
    p = _panel()
    ret = np.arange(p.pk.size, dtype=float).reshape(p.pk.shape) - p.pk.size / 2
    Fsm, names, groups = F.build_sm_features(p, ret, np.zeros_like(p.pk), p.feats[:, :, 4])
    mret = names.index("market_ret")
    t = 30
    assert groups[mret] == "market"
    np.testing.assert_allclose(Fsm[t, :, mret], np.nanmean(ret[t]), rtol=1e-5)  # same for every node


def test_graph_neighbor_orientation():
    p = _panel(n=3)
    ret = np.zeros_like(p.pk)
    adj = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.2, 0.0, 1.0]], np.float32)
    G, gnames = F.build_graph_features(p, ret, adj)
    npk = gnames.index("neighbor_pk")
    t = 20
    expect_j0 = float(adj[0] @ np.nan_to_num(p.pk[t]))       # sum_i adj[0,i]*pk_i(t)
    np.testing.assert_allclose(G[t, 0, npk], expect_j0, rtol=1e-5)


def test_pernode_skips_all_nan_column():
    mat = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    out = F._pernode(mat, lambda s: s.shift(1))
    assert np.isnan(out[:, 1]).all()            # all-NaN column produces all-NaN output (no crash)
    assert out[1, 0] == 1.0


def test_read_aligned_columns_maps_dates(tmp_path):
    p = _panel(n=2)
    files = []
    for j, tk in enumerate(p.tickers):
        df = pd.DataFrame({"date": p.dates, "daily_return": np.arange(len(p.dates)) + j,
                           "log_range": np.full(len(p.dates), float(j))})
        fp = tmp_path / f"{tk}.csv"
        df.to_csv(fp, index=False)
        files.append(str(fp))
    out = F.read_aligned_columns(files, p, ["daily_return", "log_range"])
    assert out["daily_return"].shape == (len(p.dates), p.N)
    np.testing.assert_allclose(out["daily_return"][:, 0], np.arange(len(p.dates)))
    np.testing.assert_allclose(out["log_range"][:, 1], 1.0)
