"""Panel build + train-only-scaler + packer coverage (real-data slice + synthetic branch cases)."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import wf_panel as WP                       # noqa: E402
from wf_folds import Fold                    # noqa: E402
from wf_panel import build_wf_panel, pack_fold  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
DATA_FILES = str(_REPO / "submission" / "soict_lstm_gat" / "data" / "vn100" / "*_processed.csv")
PRICE_DIR = str(_REPO / "data" / "raw" / "prices" / "vn100_vnstock")


def _real_slice(n=8):
    files = sorted(glob.glob(DATA_FILES))
    if len(files) < n:
        pytest.skip("vn100 processed files unavailable")
    keep = [Path(f).name.replace("_processed.csv", "") for f in files][:n]
    return files, keep


def _synth_panel(T=120, N=4, lookback=10, horizon=1, drop_train_node=None):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=T, freq="D")
    pk = np.abs(rng.normal(0.01, 0.005, size=(T, N))) + 1e-4
    feats = np.zeros((T, N, 5))
    for j in range(N):
        feats[:, j, 0] = pk[:, j]
        feats[:, j, 1] = pd.Series(pk[:, j]).rolling(5, min_periods=1).mean().to_numpy()
        feats[:, j, 2] = pd.Series(pk[:, j]).rolling(22, min_periods=1).mean().to_numpy()
        feats[:, j, 3] = np.median(np.sqrt(pk), axis=1)
        feats[:, j, 4] = rng.normal(0, 1, size=T)
    anchors = np.arange(lookback + 21, T - horizon)
    node_ok = np.ones((len(anchors), N), bool)
    if drop_train_node is not None:
        node_ok[:, drop_train_node] = False           # this node has no valid train rows
    target_dates = dates.to_numpy()[anchors + horizon]
    return WP.WFPanel(list("ABCDEFGH"[:N]), dates, pk, feats, anchors,
                      node_ok.copy(), node_ok.copy(), node_ok, target_dates)


# -------------------- real-data-sample smoke --------------------

def test_build_wf_panel_real_slice_shapes():
    files, keep = _real_slice(8)
    panel = build_wf_panel(files, PRICE_DIR, lookback=10, horizon=1, keep_tickers=keep)
    assert panel.N == 8
    assert panel.feats.shape[2] == 5
    assert len(panel.anchors) == len(panel.target_dates) > 100
    # anchors are within bounds for windowing and target
    assert panel.anchors.min() - 10 + 1 >= 0
    assert panel.anchors.max() + 1 < panel.pk.shape[0]


def test_build_wf_panel_too_few_files_raises():
    files, keep = _real_slice(8)
    with pytest.raises(ValueError, match="<2"):
        build_wf_panel(files, PRICE_DIR, 10, 1, keep_tickers={keep[0]})


def test_pack_fold_real_slice_train_only_scaler():
    files, keep = _real_slice(8)
    panel = build_wf_panel(files, PRICE_DIR, 10, 1, keep)
    fold = Fold(0, slice(0, 200), slice(200, 240), slice(240, 270), slice(239, 240))
    D = pack_fold(panel, fold, 10, 1)
    assert D.X_tr.shape == (200, 8, 10, 5)
    assert D.X_te.shape == (30, 8, 10, 5)
    assert D.har5_te.shape == (30, 8, 5) and D.har_te.shape == (30, 8, 3)
    assert len(D.d_te) == 30 and D.adj_vol2pk.shape == (8, 8)
    # perturbing a FORECAST-region feature must not move the train-fit scalers (leakage guard)
    tm0, ts0, fm0, fs0 = WP._fit_scalers(panel, fold.train, 1)
    saved = panel.feats.copy()
    panel.feats[panel.anchors[250], 0, 0] += 1e6
    tm1, _, fm1, _ = WP._fit_scalers(panel, fold.train, 1)
    panel.feats[:] = saved
    assert np.allclose(tm0, tm1) and np.allclose(fm0, fm1)


# -------------------- synthetic branch coverage --------------------

def test_fit_scalers_node_without_train_rows_uses_neutral_defaults():
    panel = _synth_panel(drop_train_node=0)
    fold = Fold(0, slice(0, 40), slice(40, 60), slice(60, 80), slice(59, 60))
    t_mean, t_std, f_mean, f_std = WP._fit_scalers(panel, fold.train, 1)
    assert t_mean[0] == 0.0 and abs(t_std[0] - 1.0) < 1e-6
    assert np.allclose(f_mean[0], 0.0) and np.allclose(f_std[0], 1.0, atol=1e-6)
    # a node WITH train rows gets real (non-default) statistics
    assert t_std[1] != 1.0


def test_pack_fold_empty_split_returns_zero_length_arrays():
    panel = _synth_panel()
    fold = Fold(0, slice(0, 40), slice(50, 50), slice(60, 80), slice(59, 60))  # empty val slice
    D = pack_fold(panel, fold, 10, 1)
    assert D.X_va.shape == (0, panel.N, 10, 5)
    assert D.y_va.shape == (0, panel.N) and D.har5_va.shape == (0, panel.N, 5)
    assert D.d_va == []
    assert D.X_tr.shape[0] == 40 and D.X_te.shape[0] == 20
