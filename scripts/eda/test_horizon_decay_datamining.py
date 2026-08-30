"""Unit + smoke tests for the VN100 graph horizon-decay data-mining (horizon_decay_datamining.py).

Synthetic fixtures exercise every pure function and branch; a real-data smoke runs the full pipeline on a
small VN100 slice and SKIPS cleanly when that data is absent (so coverage holds regardless of data presence).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import horizon_decay_datamining as HD  # noqa: E402


def _synth_wide(n_days=400, n_stocks=8, seed=0):
    """Wide Parkinson-variance panel with a shared market factor + a 1-day spillover shock + idiosyncratic
    persistence, so the transient lead-lag exists at h1 and decays."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    market_shock = rng.normal(0, 0.4, n_days)
    base = np.zeros((n_days, n_stocks))
    for t in range(1, n_days):
        # AR(1) persistence + yesterday's market shock spilling into today (transient, 1-day)
        base[t] = 0.85 * base[t - 1] + 0.5 * market_shock[t] + 0.3 * market_shock[t - 1] \
            + rng.normal(0, 0.2, n_stocks)
    pk = np.exp(base * 0.5 - 6.0)  # positive variance-scale values
    return pd.DataFrame(pk, index=dates, columns=[f"S{j}" for j in range(n_stocks)])


def test_leave_one_out_mean_matches_manual_and_handles_sparse():
    M = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan], [5.0, np.nan, 7.0]])
    out = HD.leave_one_out_mean(M)
    assert out[0, 0] == pytest.approx((2.0 + 3.0) / 2)
    assert out[0, 1] == pytest.approx((1.0 + 3.0) / 2)
    assert np.isnan(out[1]).all()            # all-NaN row -> NaN everywhere
    assert out[2, 0] == pytest.approx(7.0)   # peers of col0 = {NaN, 7} -> 7.0
    assert out[2, 1] == pytest.approx(6.0)   # self NaN; peers = {5, 7} -> 6.0
    assert out[2, 2] == pytest.approx(5.0)   # peers of col2 = {5, NaN} -> 5.0


def test_build_features_shapes_and_keys():
    feat = HD.build_features(_synth_wide())
    for k in ("logpk", "logharw", "logharm", "peer_lev", "peer_shock"):
        assert feat[k].shape == (feat["T"], feat["N"])
    assert feat["market"].shape == (feat["T"],)
    assert feat["mktshock"].shape == (feat["T"],)
    # monthly HAR is NaN before index 21; defined afterwards
    assert np.isnan(feat["logharm"][0, 0])
    assert np.isfinite(feat["logharm"][100, 0])


def test_fit_ols_predict_r2_recovers_linear_signal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 2))
    y = 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1] + rng.normal(0, 1e-6, 500)
    beta = HD.fit_ols(X, y)
    assert beta == pytest.approx([1.0, 2.0, -3.0], abs=1e-3)
    assert HD.r2_score(y, HD.predict(X, beta)) > 0.999


def test_r2_score_zero_variance_returns_nan():
    y = np.ones(10)
    assert np.isnan(HD.r2_score(y, y))


def test_incremental_r2_structure_and_smallness():
    feat = HD.build_features(_synth_wide(n_days=600, n_stocks=10))
    r = HD.incremental_r2(feat, 1)
    for k in ("har_r2_in", "incr_level_in", "incr_shock_in", "incr_both_in",
              "har_r2_oos", "incr_both_oos", "n_train", "n_oos"):
        assert k in r
    assert r["n_train"] > 0 and r["n_oos"] > 0
    assert r["incr_both_in"] >= r["incr_shock_in"] - 1e-9   # both block >= shock-only block in-sample


def test_row_split_is_leakage_free_with_purge_gap():
    """The train/OOS split must be strictly leakage-free: every TRAIN target date is before the 80%
    boundary, every OOS anchor is at/after it, the two masks are disjoint, and an h-day purge gap sits
    between the last train anchor and the boundary (this is the #1 correctness property of the analysis)."""
    feat = HD.build_features(_synth_wide(n_days=500, n_stocks=6))
    T = feat["T"]
    tb = int(T * HD.TRAIN_FRAC)
    for h in (1, 5, 22):
        train_sel, oos_sel = HD._row_split(feat, h)
        train_anchors = np.flatnonzero(train_sel[:, 0])
        oos_anchors = np.flatnonzero(oos_sel[:, 0])
        # no anchor is in both sets
        assert not (train_sel & oos_sel).any()
        # every TRAIN target (anchor + h) lands strictly before the boundary -> no future leaks into fit
        assert (train_anchors + h).max() < tb
        # every OOS anchor is at/after the boundary
        assert oos_anchors.min() >= tb
        # a purge gap of h days: last train anchor is at least h before the first OOS anchor
        assert oos_anchors.min() - train_anchors.max() >= h


def test_pooled_lag_autocorr_positive_for_persistent_series():
    feat = HD.build_features(_synth_wide())
    ac = HD.pooled_lag_autocorr(feat["logpk"], 1, feat)
    assert 0.0 < ac <= 1.0


def test_pooled_lag_autocorr_nan_when_no_pairs():
    feat = HD.build_features(_synth_wide(n_days=200))
    # horizon so large that no train anchor row qualifies -> empty -> NaN
    assert np.isnan(HD.pooled_lag_autocorr(feat["logpk"], feat["T"], feat))


def test_leadlag_corr_shock_decays_with_horizon():
    feat = HD.build_features(_synth_wide(n_days=800, n_stocks=12))
    c1 = HD.leadlag_corr(feat["mktshock"], feat, 1)
    c10 = HD.leadlag_corr(feat["mktshock"], feat, 10)
    assert abs(c1) > abs(c10)                       # transient shock spillover decays
    # 2-D signal path (peer_shock) also runs
    assert np.isfinite(HD.leadlag_corr(feat["peer_shock"], feat, 1))


def test_leadlag_corr_nan_when_empty():
    feat = HD.build_features(_synth_wide(n_days=200))
    assert np.isnan(HD.leadlag_corr(feat["market"], feat, feat["T"]))


def test_har_residual_matrix_and_pairwise_corr():
    feat = HD.build_features(_synth_wide(n_days=600, n_stocks=10))
    res = HD.har_residual_matrix(feat, 1)
    assert res.shape == (feat["T"], feat["N"])
    assert np.isfinite(res).any()
    med, npairs = HD.median_pairwise_corr(res, min_overlap=30)
    assert npairs > 0 and -1.0 <= med <= 1.0


def test_har_residual_matrix_skips_fully_invalid_row():
    wide = _synth_wide(n_days=600, n_stocks=10)
    wide.iloc[300, :] = np.nan          # an interior date with no valid target -> ok.any() is False there
    feat = HD.build_features(wide)
    res = HD.har_residual_matrix(feat, 1)
    assert np.isnan(res[299]).all()     # anchor whose h1 target is the all-NaN row stays unfilled
    assert np.isfinite(res).any()       # other rows still filled


def test_median_pairwise_corr_edge_branches():
    # too little overlap -> no pairs
    M = np.full((50, 3), np.nan)
    M[:5, 0] = np.arange(5.0)
    M[:5, 1] = np.arange(5.0)
    med, n = HD.median_pairwise_corr(M, min_overlap=100)
    assert np.isnan(med) and n == 0
    # a constant column (std 0) is skipped
    M2 = np.random.default_rng(2).normal(size=(200, 3))
    M2[:, 2] = 4.0
    med2, n2 = HD.median_pairwise_corr(M2, min_overlap=10)
    assert n2 == 1     # only the (0,1) pair survives; pairs with the constant col are skipped


def test_first_factor_share_common_factor_dominates():
    rng = np.random.default_rng(3)
    common = rng.normal(size=(300, 1))
    M = common + rng.normal(0, 0.05, size=(300, 6))   # strong single common factor
    share = HD.first_factor_share(M)
    assert 0.8 < share <= 1.0
    # degenerate all-constant matrix -> NaN share (skips every column, zero total variance)
    assert np.isnan(HD.first_factor_share(np.zeros((10, 4))))


def test_run_analyses_and_leadlag_only_shapes():
    wide = _synth_wide(n_days=700, n_stocks=10)
    res = HD.run_analyses(wide, horizons=(1, 5), fine_h=(1, 2, 5))
    assert res["N"] == 10 and res["T"] == 700
    assert len(res["incremental"]) == 2
    assert len(res["persistence"]) == 2
    assert len(res["leadlag"]) == 3
    assert len(res["structure"]) == 2
    ll = HD.leadlag_only(wide, fine_h=(1, 5))
    assert len(ll) == 2 and "mkt_shock" in ll[0]


def test_charts_and_render_produce_outputs():
    wide = _synth_wide(n_days=700, n_stocks=10)
    vn = HD.run_analyses(wide, horizons=(1, 5), fine_h=(1, 2, 5))
    contrasts = {"VN100": vn["leadlag"], "HNX": HD.leadlag_only(wide, fine_h=(1, 2, 5))}
    charts = HD.make_charts(vn, contrasts)
    for key in ("leadlag", "incremental", "persistence", "structure", "contrast"):
        assert isinstance(charts[key], str) and len(charts[key]) > 100
    html = HD.render_html(vn, contrasts, charts)
    assert "<html" in html and "Executive summary" in html and "data:image/png;base64," in html
    md = HD.render_md(vn, contrasts)
    assert md.startswith("# VN100") and "Proven mechanism" in md and "Caveats" in md


@pytest.mark.smoke
def test_real_vn100_slice_smoke():
    import glob
    vn_dir = HD.REPO / "submission" / "soict_lstm_gat" / "data" / "vn100"
    files = sorted(glob.glob(str(vn_dir / "*_processed.csv")))
    if len(files) < 8:  # pragma: no cover - data-absent guard
        pytest.skip("VN100 processed data not present")
    wide = HD._load_panel(vn_dir).iloc[:, :12]
    res = HD.run_analyses(wide, horizons=(1, 5), fine_h=(1, 2, 5))
    assert res["N"] == 12
    h1 = next(d["mkt_shock"] for d in res["leadlag"] if d["h"] == 1)
    assert np.isfinite(h1)
