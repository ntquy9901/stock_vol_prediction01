"""Tests for topology.global_feats / build_topo_windows / build_topo_daily (real ln-volume path monkeypatched)."""
import numpy as np
import pandas as pd
import pytest

import config
import topology
import volume_io


def _synth_frames(n_tickers=5, n_rows=80, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_rows)
    frames = {}
    for t in range(n_tickers):
        frames[f"T{t}"] = pd.DataFrame({
            "date": dates,
            "daily_return": rng.standard_normal(n_rows),
        })
    return frames, dates


def _patch_lnvol(monkeypatch, dates, n_tickers=5, seed=1):
    """Monkeypatch load_log_volume to a synthetic ln(volume) panel aligned to `dates`."""
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame({f"T{t}": rng.standard_normal(len(dates)) for t in range(n_tickers)},
                         index=pd.DatetimeIndex(dates))
    monkeypatch.setattr(volume_io, "load_log_volume", lambda market, tickers: panel[list(tickers)])


def test_global_feats_connected_matrix():
    # 4 nodes fully correlated above THR -> edges present (covers diameter / avg_w / betw paths)
    C = np.array([[1.0, 0.9, 0.8, 0.7],
                  [0.9, 1.0, 0.85, 0.75],
                  [0.8, 0.85, 1.0, 0.9],
                  [0.7, 0.75, 0.9, 1.0]])
    feats = topology.global_feats(C)
    assert len(feats) == 6
    assert all(np.isfinite(v) for v in feats)
    assert 0.0 <= feats[0] <= 1.0


def test_global_feats_no_edges():
    # all off-diagonal below THR -> no edges (covers avg_w=0 / diam=0 no-edge branches)
    C = np.eye(4) + 0.1 * (np.ones((4, 4)) - np.eye(4))
    feats = topology.global_feats(C)
    assert feats[0] == 0.0            # density 0
    assert feats[3] == 0.0           # avg_w 0
    assert feats[4] == 0.0           # diameter 0
    assert all(np.isfinite(v) for v in feats)


def test_global_feats_two_nodes():
    # n == 2 -> betweenness else-branch (0.0)
    C = np.array([[1.0, 0.9], [0.9, 1.0]])
    feats = topology.global_feats(C)
    assert feats[5] == 0.0           # betweenness 0 for n<=2
    assert len(feats) == 6


def test_build_topo_windows_produces_finite_rows_and_counts(monkeypatch):
    monkeypatch.setattr(config, "MIN_COMMON_TICKERS", 3)
    frames, dates = _synth_frames()
    _patch_lnvol(monkeypatch, dates)
    F, n = topology.build_topo_windows(frames, "hose", alpha=config.ALPHA, win=config.WIN)
    assert list(F.columns) == config.TOPO
    assert len(F) >= 1
    assert set(F.index).issubset(set(dates))       # window-end dates are panel dates
    assert np.isfinite(F.to_numpy(float)).all()
    # n_by_window: one count per emitted window, in [MIN_COMMON, n_tickers]
    assert list(n.index) == list(F.index)
    assert (n >= 3).all() and (n <= 5).all()


def test_build_topo_windows_skips_when_too_few_common(monkeypatch):
    monkeypatch.setattr(config, "MIN_COMMON_TICKERS", 999)   # force the < MIN skip branch
    frames, dates = _synth_frames()
    _patch_lnvol(monkeypatch, dates)
    F, n = topology.build_topo_windows(frames, "hose", alpha=config.ALPHA, win=config.WIN)
    assert F.empty
    assert list(F.columns) == config.TOPO
    assert n.empty


def test_build_topo_daily_ffills_to_all_dates(monkeypatch):
    monkeypatch.setattr(config, "MIN_COMMON_TICKERS", 3)
    frames, dates = _synth_frames()
    _patch_lnvol(monkeypatch, dates)
    D = topology.build_topo_daily(frames, "hose", alpha=config.ALPHA, win=config.WIN)
    assert len(D) == len(dates)                    # reindexed to every date
    assert list(D.columns) == config.TOPO


@pytest.mark.smoke
def test_build_topo_windows_real_hose_slice(monkeypatch):
    """Real-data-sample smoke (M2): run the topology core end-to-end on real HOSE frames + real ln(volume)."""
    enr = config.REPO / "data" / "processed_enriched" / "hose"
    raw = config.REPO / config.RAW_VOL_DIR["hose"]                # only tickers that also have raw volume
    tickers = sorted(p.stem for p in enr.glob("*.csv")
                     if not p.stem.endswith("_rejections")        # skip audit sidecar CSVs
                     and (raw / f"{p.stem}_ohlcv.csv").exists())[:12]
    frames = {tk: pd.read_csv(enr / f"{tk}.csv", parse_dates=["date"])[["date", "daily_return"]]
              for tk in tickers}
    monkeypatch.setattr(config, "MIN_COMMON_TICKERS", 5)
    F, n = topology.build_topo_windows(frames, "hose", alpha=config.ALPHA, win=30)   # real load_log_volume
    assert not F.empty and list(F.columns) == config.TOPO
    assert np.isfinite(F.to_numpy(float)).all()                  # real correlations produce finite metrics
    assert (F.index.to_series().diff().dropna() > pd.Timedelta(0)).all()   # window-end dates strictly increasing
    assert (n >= 5).all() and (n <= len(tickers)).all()
