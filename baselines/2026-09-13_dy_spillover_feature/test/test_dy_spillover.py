"""Tests for the causal Diebold-Yilmaz spillover feature builder: GFEVD sign on a known VAR, the log-mean
sector panel, per-window degeneracy paths, causal rolling construction (perturbation test) and broadcast."""
import numpy as np
import pandas as pd
import pytest

import config
import dy_spillover as DS


def _driven_var(n=300, seed=0):
    """Two series where B_t = 0.8 A_{t-1} + noise -> A is the exogenous net transmitter."""
    rng = np.random.default_rng(seed)
    a = np.zeros(n); b = np.zeros(n)
    ea = rng.standard_normal(n); eb = 0.2 * rng.standard_normal(n)
    for t in range(1, n):
        a[t] = 0.3 * a[t - 1] + ea[t]
        b[t] = 0.8 * a[t - 1] + eb[t]
    return pd.DataFrame({"A": a, "B": b})


def test_gfevd_sign_known_var():
    """B is driven by lagged A -> net directional spillover of A is positive, of B negative."""
    s, net = DS.spillover_from_window(_driven_var(), lag=1, gfevd_h=config.DY_GFEVD_H,
                                      min_sectors=2, min_std=config.DY_VAR_MIN_STD, min_obs=50)
    assert net["A"] > 0 > net["B"]
    assert 0.0 < s <= 100.0                          # a real, bounded system-wide spillover index


def test_gfevd_identity_is_zero_spillover():
    """Identity MA + diagonal Sigma -> each series explained only by itself -> theta = I, S = 0, net = 0."""
    ma = np.stack([np.eye(2), np.zeros((2, 2))])
    theta = DS.gfevd(ma, np.diag([1.0, 2.0]))
    assert np.allclose(theta.sum(axis=1), 1.0)       # rows normalised
    assert np.allclose(np.diag(theta), 1.0)
    s, net = DS.spillover_scalars(theta)
    assert s == pytest.approx(0.0) and np.allclose(net, 0.0)


def _sector_frames(n_rows=260, seed=0):
    """9 tickers across 3 sectors (3 each) with distinct volatility dynamics per sector."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_rows)
    frames = {}
    base = {0: rng.standard_normal(n_rows), 1: rng.standard_normal(n_rows), 2: rng.standard_normal(n_rows)}
    for i in range(9):
        sec = i % 3
        pv = np.abs(base[sec] + 0.3 * rng.standard_normal(n_rows)) * 1e-4 + 1e-6
        d = pd.DataFrame({"date": dates, "parkinson_variance": pv, "sector": sec, "ticker": f"T{i}"})
        frames[f"T{i}"] = d
    return frames


def test_sector_logvar_panel_qualifies_and_logs():
    frames = _sector_frames()
    panel = DS.sector_logvar_panel(frames, min_stocks=3)
    assert list(panel.columns) == [0, 1, 2]                       # all three sectors have 3 constituents
    # value at a date = log(mean parkinson_variance over sector-0 stocks that day)
    d0 = panel.index[100]
    mean_pv = np.mean([frames[f"T{i}"].set_index("date").loc[d0, "parkinson_variance"] for i in (0, 3, 6)])
    assert panel.loc[d0, 0] == pytest.approx(np.log(max(mean_pv, DS.FL)))


def test_sector_logvar_panel_empty_when_none_qualify():
    panel = DS.sector_logvar_panel(_sector_frames(), min_stocks=99)   # no sector reaches 99 constituents
    assert panel.empty


def test_spillover_from_window_short_and_thin(monkeypatch):
    win = _driven_var(n=40)
    nan_s, nan_net = DS.spillover_from_window(win, 1, 10, 2, 1e-9, min_obs=100)   # too few rows
    assert np.isnan(nan_s) and nan_net.isna().all()
    # one degenerate (constant) column leaves <2 non-degenerate series
    win2 = win.copy(); win2["B"] = 5.0
    s2, net2 = DS.spillover_from_window(win2, 1, 10, min_sectors=2, min_std=1e-9, min_obs=20)
    assert np.isnan(s2) and net2.isna().all()


def test_spillover_from_window_var_failure_is_caught(monkeypatch):
    def boom(*a, **k):
        raise ValueError("singular")
    monkeypatch.setattr(DS, "VAR", boom)
    s, net = DS.spillover_from_window(_driven_var(), 1, 10, 2, 1e-9, min_obs=50)
    assert np.isnan(s) and net.isna().all()


def test_rolling_spillover_is_causal():
    """Perturbing volatilities AFTER a cutoff must not change any spillover value dated <= cutoff."""
    frames = _sector_frames(n_rows=320)
    panel = DS.sector_logvar_panel(frames, min_stocks=3)
    net_df, total, diag = DS.rolling_spillover(panel, window=60, step=5, lag=1, gfevd_h=10,
                                               min_sectors=2, win_min_frac=0.8, min_std=1e-9)
    cut = panel.index[200]
    pert = panel.copy()
    pert.loc[pert.index > cut] += 10.0                            # corrupt only the future
    net_df2, total2, _ = DS.rolling_spillover(pert, window=60, step=5, lag=1, gfevd_h=10,
                                              min_sectors=2, win_min_frac=0.8, min_std=1e-9)
    m = net_df.index <= cut
    assert total[m].notna().any()                                # pre-cut region has real values -> not vacuous
    pd.testing.assert_frame_equal(net_df[m], net_df2[m])
    pd.testing.assert_series_equal(total[m], total2[m])
    assert diag["n_sectors"] == 3 and diag["n_anchor_windows"] > 0


def test_merge_spillover_broadcast_and_missing_sector():
    frames = _sector_frames()
    panel = DS.sector_logvar_panel(frames, min_stocks=3)
    net_df, total, _ = DS.rolling_spillover(panel, 60, 5, 1, 10, 2, 0.8, 1e-9)
    # add a ticker in a sector that is NOT a VAR series -> should receive NaN net
    frames["Z"] = frames["T0"].assign(sector=7, ticker="Z")
    merged = DS.merge_spillover(frames, net_df, total)
    d = merged["T0"]
    row = d[d[config.FEAT_NET].notna()].iloc[0]
    assert row[config.FEAT_NET] == pytest.approx(net_df.loc[row["date"], 0])
    assert row[config.FEAT_TOTAL] == pytest.approx(total.loc[row["date"]])
    assert merged["Z"][config.FEAT_NET].isna().all()             # sector 7 not in the VAR system


def test_merge_spillover_handles_empty_frame():
    frames = _sector_frames()
    panel = DS.sector_logvar_panel(frames, min_stocks=3)
    net_df, total, _ = DS.rolling_spillover(panel, 60, 5, 1, 10, 2, 0.8, 1e-9)
    frames["EMPTY"] = frames["T0"].iloc[0:0].copy()              # zero-row frame -> sector is None
    merged = DS.merge_spillover(frames, net_df, total)
    assert config.FEAT_NET in merged["EMPTY"].columns and len(merged["EMPTY"]) == 0


def test_build_spillover_wires_from_config(monkeypatch):
    monkeypatch.setattr(config, "DY_SECTOR_MIN_STOCKS", 3)
    monkeypatch.setattr(config, "DY_WINDOW", 60)
    monkeypatch.setattr(config, "DY_STEP", 5)
    monkeypatch.setattr(config, "DY_MIN_SECTORS", 2)
    merged, diag = DS.build_spillover(_sector_frames(n_rows=300))
    assert set(diag) == {"n_sectors", "n_anchor_windows", "n_failed_windows"}
    assert any(f[config.FEAT_TOTAL].notna().any() for f in merged.values())
