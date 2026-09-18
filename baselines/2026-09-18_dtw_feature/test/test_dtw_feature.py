"""Tests for the DTW self-similarity feature baseline: banded DTW correctness, causal template/window
construction, feature-frame plumbing, DM/verdict/success logic, and a walk-forward run smoke carrying the
gate-required over/under-fit evidence keys. Synthetic panels keep the driver fast without real data."""
import numpy as np
import pandas as pd
import pytest

import dtw_config as C
import dtw_feature as DF
import run_dtw_feature as R
import vn_gbm_graph_stage1 as S1


# --------------------------------------------------------------------------- synthetic data
def _ticker_frame(seed, start="2021-01-01", end="2023-06-30"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    pk = np.empty(n)
    pk[0] = 3e-4
    for t in range(1, n):
        pk[t] = max(1e-6, 0.85 * pk[t - 1] + 0.15 * 3e-4 + rng.normal(0, 3e-5))
    s = pd.Series(pk)
    base = pd.DataFrame({
        "date": dates, "parkinson_variance": pk,
        "har_daily": pk, "har_weekly": s.rolling(5, min_periods=1).mean().to_numpy(),
        "har_monthly": s.rolling(22, min_periods=1).mean().to_numpy(),
        "volume_zscore_22": rng.normal(0, 1, n), "market_pk": pk * 0.9,
        "daily_return": rng.normal(0, 0.01, n)})
    return S1._feat(base)


def _frames(n_tickers=3):
    return {f"TK{i}": _ticker_frame(seed=i).assign(ticker=f"TK{i}", sector=0) for i in range(n_tickers)}


def _fake_loader(market):
    return _frames(), {f"TK{i}": 0 for i in range(3)}, {}


def _tiny(monkeypatch):
    monkeypatch.setattr(C, "MIN_ROWS", {"sp500": 10, "default": 10})


# --------------------------------------------------------------------------- banded DTW
def test_banded_dtw_identical_is_zero():
    q = np.array([[0.0, 1.0, 2.0, 3.0]])
    assert DF.banded_dtw_batch(q, q[0], band=2)[0] == pytest.approx(0.0)


def test_banded_dtw_known_pair_value():
    # every local cost is 1 (all zeros vs all ones); the cheapest monotonic path is the diagonal of W=3
    # cells -> accumulated 3 -> distance sqrt(3).
    q = np.zeros((1, 3))
    t = np.ones(3)
    assert DF.banded_dtw_batch(q, t, band=2)[0] == pytest.approx(np.sqrt(3.0))


def test_banded_dtw_batch_axis_and_shapes():
    q = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = DF.banded_dtw_batch(q, np.ones(3), band=1)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(np.sqrt(3.0)) and out[1] == pytest.approx(0.0)


def test_banded_dtw_bad_shape_raises():
    with pytest.raises(ValueError):
        DF.banded_dtw_batch(np.zeros((2, 3)), np.ones(4), band=1)
    with pytest.raises(ValueError):
        DF.banded_dtw_batch(np.zeros(3), np.ones(3), band=1)


# --------------------------------------------------------------------------- windows + templates
def test_windows_front_pad_and_trailing():
    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    w = DF._windows(z, np.array([0, 4]), 3)
    assert list(w[0]) == [0.0, 0.0, 0.0]        # early row front-pads with the first value (causal)
    assert list(w[1]) == [2.0, 3.0, 4.0]        # trailing window ends at the position


def test_build_templates_bands_and_empty_band():
    tw = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    temps = DF.build_templates(tw, ((0.0, 0.5), (0.5, 1.0)))
    assert temps.shape == (2, 2)
    assert temps[0].mean() < temps[1].mean()    # low band centroid below high band centroid
    # a degenerate band where floor(lo*n) == ceil(hi*n) exercises the single-window `else` fallback
    one = DF.build_templates(tw, ((0.5, 0.5),))
    assert one.shape == (1, 2)


# --------------------------------------------------------------------------- fold features (causality)
def _series_and_combo():
    frames = _frames(2)
    series = DF.prepare_series(frames)
    # combo = a slice of rows for both tickers spanning train + later dates
    rows = []
    for tk, d in frames.items():
        rows.append(d.iloc[100:140].assign(ticker=tk))
    combo = pd.concat(rows).reset_index(drop=True)
    return frames, series, combo


def test_fold_features_columns_and_causal():
    frames, series, combo = _series_and_combo()
    boundary = pd.Timestamp("2022-01-01")
    out = DF.fold_features(series, combo, boundary, C)
    real, plac = DF.feature_names(C)
    assert list(out.columns) == real + plac
    assert len(out) == len(combo)
    # CAUSALITY: perturb FUTURE values (after the combo dates) -> an earlier row's feature is unchanged
    frames2 = _frames(2)
    for tk in frames2:
        m = frames2[tk]["date"] > pd.Timestamp("2023-01-01")
        frames2[tk].loc[m, "logpk"] = frames2[tk].loc[m, "logpk"] + 5.0
    out2 = DF.fold_features(DF.prepare_series(frames2), combo, boundary, C)
    np.testing.assert_allclose(out.to_numpy(), out2.to_numpy(), equal_nan=True)


def test_fold_features_skips_missing_ticker_and_short_history(monkeypatch):
    frames, series, combo = _series_and_combo()
    boundary = pd.Timestamp("2022-01-01")
    # a ticker present in combo but absent from `series` -> skipped, its rows stay NaN
    extra = combo.iloc[:5].copy()
    extra["ticker"] = "ZZZ"
    combo2 = pd.concat([combo, extra]).reset_index(drop=True)
    out = DF.fold_features(series, combo2, boundary, C)
    assert out.iloc[-5:].isna().all().all()
    # too few train windows -> NaN features for every row (raise the min threshold above availability)
    monkeypatch.setattr(C, "DTW_MIN_TRAIN_WINDOWS", 10_000)
    assert DF.fold_features(series, combo, boundary, C).isna().all().all()


def test_fold_features_early_boundary_skips_when_train_below_window():
    frames, series, combo = _series_and_combo()
    # boundary before any history -> train_mask.sum() < W -> skip, all NaN
    out = DF.fold_features(series, combo, pd.Timestamp("1990-01-01"), C)
    assert out.isna().all().all()


def test_fold_features_constant_series_zero_std_branch():
    # constant log-variance -> train std 0 -> the `sd if sd>0 else 1.0` guard path
    dates = pd.bdate_range("2021-01-01", "2022-12-31")
    const = pd.DataFrame({"date": dates, "logpk": np.ones(len(dates))})
    series = {"TK": (const["date"].to_numpy(), const["logpk"].to_numpy(float))}
    combo = const.iloc[300:320].assign(ticker="TK").reset_index(drop=True)
    out = DF.fold_features(series, combo, pd.Timestamp("2022-06-01"), C)
    # identical (constant) windows and templates -> DTW distance 0 everywhere
    assert np.allclose(out.to_numpy(), 0.0)


def test_placebo_shift_differs_from_real_on_trending_series():
    # a strongly trending series: the date-shifted placebo window differs from the recent window,
    # so at least one placebo distance differs from its real counterpart.
    dates = pd.bdate_range("2020-01-01", "2022-12-31")
    lpk = np.linspace(-6.0, -2.0, len(dates))
    series = {"TK": (dates.to_numpy(), lpk)}
    combo = pd.DataFrame({"date": dates[400:440], "ticker": "TK"})
    out = DF.fold_features(series, combo, pd.Timestamp("2022-06-01"), C)
    real, plac = DF.feature_names(C)
    assert not np.allclose(out[real].to_numpy(), out[plac].to_numpy())


# --------------------------------------------------------------------------- pure logic
def test_verdict_and_gain():
    assert R.verdict(0.5, 0.01) and not R.verdict(-0.1, 0.01) and not R.verdict(0.5, 0.20)
    assert R._gain(1.0, 0.8) == pytest.approx(20.0)


def test_success_counts_spike_and_placebo():
    good = {"verdict": {"beats": True}, "spike_robustness": {"beats_ex_spike": True},
            "placebo_verdict": {"beats": False}}
    assert R.success({1: good, 5: dict(good)})
    # placebo also beats -> disqualified
    dirty = {"verdict": {"beats": True}, "spike_robustness": {"beats_ex_spike": True},
             "placebo_verdict": {"beats": True}}
    assert not R.success({1: good, 5: dirty})
    # only one clean horizon -> below KILL_MIN_HORIZONS
    assert not R.success({1: good})


def test_safe_dm_degenerate_and_valueerror(monkeypatch):
    e = np.array([1.0, 2.0, 3.0, 4.0])
    d = pd.to_datetime(["2021-01-01"] * 4).to_numpy()
    assert R._safe_dm(e, e.copy(), d, 1)["p_value"] == 1.0
    monkeypatch.setattr(R.ST, "date_clustered_dm", lambda *a, **k: (_ for _ in ()).throw(ValueError("hln")))
    assert R._safe_dm(e, e + 1.0, d, 1)["p_value"] == 1.0


def test_safe_dm_runs_real():
    rng = np.random.default_rng(0)
    e_a = rng.random(60) + 0.5
    e_b = rng.random(60)
    dates = pd.bdate_range("2021-01-01", periods=60).to_numpy()
    assert 0.0 <= R._safe_dm(e_a, e_b, dates, 1)["p_value"] <= 1.0


def test_spike_mask_flags_windows():
    d = pd.to_datetime(["2019-06-01", "2020-03-15", "2022-06-01", "2025-04-10"]).to_numpy()
    assert list(R._spike_mask(d)) == [False, True, True, True]


def test_load_earn_sp500_passthrough_and_hose_paths(monkeypatch, tmp_path):
    assert R._load_earn("sp500", {"A": 1}) == {"A": 1}
    monkeypatch.setattr(R, "REPO", tmp_path)                          # no parquet under tmp
    assert R._load_earn("hose", {}) == {}
    gdir = tmp_path / "results" / "gamma_gbm"
    gdir.mkdir(parents=True)
    pd.DataFrame({"ticker": ["AAA", "AAA"],
                  "earnings_date": pd.to_datetime(["2021-01-01", "2021-04-01"])}).to_parquet(
        gdir / "hose_earnings_combined.parquet")
    got = R._load_earn("hose", {})
    assert "AAA" in got and len(got["AAA"]) == 2


def test_metrics5_keys():
    y = np.array([1e-4, 2e-4, 3e-4])
    p = np.array([1.1e-4, 1.9e-4, 3.2e-4])
    assert set(R._metrics5(y, p)) == {"mse", "rmse", "mae", "r2", "qlike"}


# --------------------------------------------------------------------------- run smoke
def test_run_hose_structure_evidence_perfold_spike(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    monkeypatch.setattr(C, "SPIKE_WINDOWS", (("1990-01-01", "1990-01-02"),))   # non-overlapping -> spike computed
    docs = R.run("hose", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)
    assert set(docs) == {1}
    doc = docs[1]
    assert (tmp_path / "dtw_hose_smoke_h1.json").exists()
    for blk in ("metrics", "train_metrics", "val_metrics"):
        for m in R.ORDER:
            assert set(doc[blk][m]) == {"mse", "rmse", "mae", "r2", "qlike"}
    assert set(doc["dm"]) == {"GBME+dtw_vs_GBME", "GBME+dtw_vs_GBME+dtw_placebo",
                              "GBME+dtw_placebo_vs_GBME"}
    assert "per_fold_qlike" in doc and "spike_robustness" in doc
    assert "verdict" in doc and "placebo_verdict" in doc
    # fit evidence is carried for every arm (train/val/test -> classify_fit verdict)
    for m in R.ORDER:
        assert doc["fit_diagnostics"][m]["status"] in {"ok", "overfit", "underfit", "unknown"}
    # the pre-push over/under-fit gate treats this deterministic-boosting result as a non-learned file
    # (no learned tokens in the model names) and skips it -> no BLOCK.
    import check_overfit_evidence as CE
    assert CE.check_files([str(tmp_path / "dtw_hose_smoke_h1.json")]) == {}


def test_run_sp500_no_spike_block(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)
    doc = docs[1]
    assert "per_fold_qlike" not in doc and "spike_robustness" not in doc
    assert (tmp_path / "dtw_sp500_smoke_h1.json").exists()


def test_run_skips_empty_folds_multi(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    monkeypatch.setattr(C, "HORIZONS", (1,))
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=False)   # horizons=None -> C.HORIZONS
    assert docs[1]["n_folds"] >= 1


def test_run_horizons_override(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=False, horizons=(1,))
    assert set(docs) == {1}


def test_run_omits_robustness_when_all_test_in_spike(monkeypatch, tmp_path):
    _tiny(monkeypatch)                                                # default 2022 window covers fold0 test
    docs = R.run("hose", load_fn=_fake_loader, out_dir=tmp_path, smoke=True)
    assert "spike_robustness" not in docs[1] and "per_fold_qlike" in docs[1]
