"""Tests for the foundation-model feature baseline (Hướng B falsification): causal zero-shot forecasting, the
per-(ticker,date) cache, wrong-ticker placebo, model plumbing, Stage-0/Stage-1 DM + verdict logic, spike mask,
and a walk-forward run smoke carrying the gate-required over/under-fit evidence. A fast fake forecaster keeps the
driver test off the real (slow) Chronos model; one opt-in test exercises the real model on a tiny slice."""
import numpy as np
import pandas as pd
import pytest

import foundation_config as C
import foundation_forecaster as FF
import run_foundation as R
import vn_gbm_graph_stage1 as S1


# --------------------------------------------------------------------------- synthetic data + fake forecaster
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


def _persistence_fn(contexts, pred_len):
    """Fast causal fake: median forecast = last observed value repeated (persistence); spread = 1% of it. Because
    it uses only ``context[-1]`` (the value at t), a leak of any future value would change the output."""
    med = np.array([np.full(pred_len, float(c[-1])) for c in contexts])
    return med, 0.01 * med


def _step_fn(contexts, pred_len):
    """Step-varying causal fake: the step-``k`` forecast = ``context[-1] + k*1e-8`` (k=1..pred_len). Distinct per
    step, so it LOCKS the step->horizon->target alignment (a silent off-by-one in ``step=h-1`` or a mismatched
    panel shift would change which value lands in ``fnd_h{h}``). Increments stay well below PRED_CAP/floor."""
    med = np.array([float(c[-1]) + np.arange(1, pred_len + 1) * 1e-8 for c in contexts])
    return med, np.full_like(med, 1e-9)


def _fake_cache(frames):
    return FF.build_cache(frames, _persistence_fn, C)


def _tiny(monkeypatch):
    monkeypatch.setattr(C, "MIN_ROWS", {"sp500": 10, "default": 10})


# --------------------------------------------------------------------------- forecaster: causality + cache
def test_batched_forecast_chunks():
    ctx = [np.arange(1.0, 5.0)[:i + 2] for i in range(5)]        # 5 variable-length contexts
    med, spr = FF._batched_forecast(_persistence_fn, ctx, pred_len=3, batch=2)   # 2+2+1 chunks
    assert med.shape == (5, 3) and spr.shape == (5, 3)
    assert np.allclose(med[:, 0], [c[-1] for c in ctx])


def test_forecast_series_is_causal_and_respects_min_context(monkeypatch):
    monkeypatch.setattr(C, "MIN_CONTEXT", 8)
    monkeypatch.setattr(C, "CTX_LEN", 20)
    vals = np.linspace(1e-4, 5e-4, 40)
    dates = pd.bdate_range("2022-01-01", periods=40).to_numpy()
    df = FF.forecast_series(_persistence_fn, dates, vals, C)
    # first MIN_CONTEXT-1 rows have too little context and are dropped
    assert len(df) == 40 - (C.MIN_CONTEXT - 1)
    # persistence: fnd_h1 for the row dated date[i] equals vals[i] (context ended at t -> causal, no future leak)
    merged = df.set_index("date")
    for i in range(C.MIN_CONTEXT - 1, 40):
        assert merged.loc[dates[i], "fnd_h1"] == pytest.approx(vals[i])
    for h in C.HORIZONS:
        assert f"fnd_h{h}" in df and f"fspread_h{h}" in df


def test_forecast_series_step_to_horizon_alignment(monkeypatch):
    monkeypatch.setattr(C, "MIN_CONTEXT", 2)
    vals = np.full(12, 3e-4)
    df = FF.forecast_series(_step_fn, pd.bdate_range("2022-01-01", periods=12).to_numpy(), vals, C)
    row = df.iloc[0]
    # fnd_h{h} must carry the step-h forecast (index h-1): context-last 3e-4 + h*1e-8
    for h in C.HORIZONS:
        assert row[f"fnd_h{h}"] == pytest.approx(3e-4 + h * 1e-8, rel=1e-6)


def test_attach_uses_horizon_matched_forecast():
    frames = _frames()
    cache = FF.build_cache(frames, _step_fn, C)
    panel_cols = ["ticker", "date", "y"] + R.OWN + ["_absent_col_"]     # exercises the missing-column filter too
    attached = {}
    for h in (1, 5):
        a = R._attach(R.FM.panel(frames, {}, h), cache, h, panel_cols)
        assert "_absent_col_" not in a.columns
        r = a.iloc[50]
        want = cache[(cache.ticker == r["ticker"]) & (cache.date == r["date"])][f"fnd_h{h}"].iloc[0]
        assert r["fnd"] == pytest.approx(want)                          # attached fnd is the horizon-h forecast
        attached[h] = (r["ticker"], r["date"], r["fnd"])
    # a step-varying forecaster gives a DIFFERENT fnd for h1 vs h5 at the same (ticker,date) -> horizon-matched
    same = cache[(cache.ticker == attached[1][0]) & (cache.date == attached[1][1])]
    assert same["fnd_h1"].iloc[0] != same["fnd_h5"].iloc[0]


def test_forecast_series_without_spread(monkeypatch):
    monkeypatch.setattr(C, "USE_SPREAD", False)
    df = FF.forecast_series(_persistence_fn, pd.bdate_range("2022-01-01", periods=20).to_numpy(),
                            np.linspace(1e-4, 2e-4, 20), C)
    assert "fnd_h1" in df and "fspread_h1" not in df


def test_forecast_series_clips_to_floor_and_cap(monkeypatch):
    monkeypatch.setattr(C, "MIN_CONTEXT", 2)
    # a context whose last value is huge must clip the median forecast to PRED_CAP; a near-zero one to FL
    df_hi = FF.forecast_series(_persistence_fn, pd.bdate_range("2022-01-01", periods=5).to_numpy(),
                               np.full(5, 100.0), C)
    df_lo = FF.forecast_series(_persistence_fn, pd.bdate_range("2022-01-01", periods=5).to_numpy(),
                               np.full(5, 1e-20), C)
    assert (df_hi["fnd_h1"] <= C.PRED_CAP + 1e-12).all() and (df_lo["fnd_h1"] >= FF.FL).all()


def test_build_cache_skips_ticker_shorter_than_min_context(monkeypatch):
    monkeypatch.setattr(C, "MIN_CONTEXT", 8)
    short = pd.DataFrame({"date": pd.bdate_range("2022-01-01", periods=3),
                          "parkinson_variance": [3e-4, 3e-4, 3e-4]})
    frames = {"TK0": _ticker_frame(0), "SHORT": short}
    cache = FF.build_cache(frames, _persistence_fn, C)               # SHORT yields no rows -> skipped
    assert set(cache["ticker"].unique()) == {"TK0"}


def test_placebo_without_spread(monkeypatch):
    monkeypatch.setattr(C, "USE_SPREAD", False)
    cache = _fake_cache(_frames())
    plac = R._placebo(cache, 1)
    assert "fnd_plac" in plac and "fnd_plac_spread" not in plac


def test_build_and_load_cache_roundtrip(tmp_path):
    frames = _frames()
    cache = FF.build_cache(frames, _persistence_fn, C)
    assert set(cache["ticker"].unique()) == {"TK0", "TK1", "TK2"}
    for h in C.HORIZONS:
        assert f"fnd_h{h}" in cache
    p = tmp_path / "cache.parquet"
    built = FF.load_or_build_cache(frames, p, predict_fn=_persistence_fn, cfg=C)   # builds + writes
    assert p.exists() and len(built) == len(cache)
    reread = FF.load_or_build_cache(frames, p, predict_fn=_persistence_fn, cfg=C)   # reads parquet
    assert len(reread) == len(built)
    rebuilt = FF.load_or_build_cache(frames, p, predict_fn=_persistence_fn, cfg=C, rebuild=True)  # forces rebuild
    assert len(rebuilt) == len(built)


# --------------------------------------------------------------------------- pure logic
def test_fnd_cols_toggle(monkeypatch):
    monkeypatch.setattr(C, "USE_SPREAD", True)
    assert R._fnd_cols("fnd") == ["fnd", "fnd_spread"]
    monkeypatch.setattr(C, "USE_SPREAD", False)
    assert R._fnd_cols("fnd") == ["fnd"]


def test_placebo_is_wrong_ticker():
    cache = _fake_cache(_frames())
    plac = R._placebo(cache, 1)
    assert set(plac.columns) >= {"date", "ticker", "fnd_plac"}
    # TK0's placebo value at a shared date must equal TK1's real forecast (cyclic map TK0<-TK1)
    d = cache[cache.ticker == "TK1"]["date"].iloc[10]
    want = cache[(cache.ticker == "TK1") & (cache.date == d)]["fnd_h1"].iloc[0]
    got = plac[(plac.ticker == "TK0") & (plac.date == d)]["fnd_plac"].iloc[0]
    assert got == pytest.approx(want)


def test_attach_adds_paired_feature_columns():
    frames = _frames()
    cache = _fake_cache(frames)
    a = R._attach(R.FM.panel(frames, {}, 1), cache, 1, ["ticker", "date", "y"] + R.OWN)
    for col in ("fnd", "fnd_spread", "fnd_plac", "fnd_plac_spread"):
        assert col in a
    assert a[["fnd", "fnd_plac"]].notna().all().all()


def test_metrics5_keys():
    y = np.array([1e-4, 2e-4, 3e-4]); p = np.array([1.1e-4, 1.9e-4, 3.2e-4])
    assert set(R._metrics5(y, p)) == {"mse", "rmse", "mae", "r2", "qlike"}


def test_verdict_thresholds():
    assert R.verdict(0.5, 0.01) and not R.verdict(-0.1, 0.01) and not R.verdict(0.5, 0.20)


def test_success_all_branches():
    good = {1: {"verdict": {"beats": True}}, 5: {"verdict": {"beats": True},
                                                 "spike_robustness": {"beats_ex_spike": True}}}
    assert R.success(good)
    assert not R.success({1: {"verdict": {"beats": True}}})                        # < KILL_MIN_HORIZONS
    plac = {1: {"verdict": {"beats": True}, "placebo": {"beats": True}},
            5: {"verdict": {"beats": True}}}
    assert not R.success(plac)                                                     # placebo wins -> fail
    spike = {1: {"verdict": {"beats": True}},
             5: {"verdict": {"beats": True}, "spike_robustness": {"beats_ex_spike": False}}}
    assert not R.success(spike)                                                    # spike kills a beat -> fail


def test_safe_dm_degenerate_and_valueerror(monkeypatch):
    e = np.array([1.0, 2.0, 3.0, 4.0]); d = pd.to_datetime(["2021-01-01"] * 4).to_numpy()
    assert R._safe_dm(e, e.copy(), d, 1)["p_value"] == 1.0
    monkeypatch.setattr(R.ST, "date_clustered_dm", lambda *a, **k: (_ for _ in ()).throw(ValueError("hln")))
    assert R._safe_dm(e, e + 1.0, d, 1)["p_value"] == 1.0


def test_spike_mask_flags_windows():
    d = pd.to_datetime(["2019-06-01", "2020-03-15", "2022-06-01", "2025-04-10"]).to_numpy()
    assert list(R._spike_mask(d)) == [False, True, True, True]


def test_load_earn_sp500_passthrough_and_hose(monkeypatch, tmp_path):
    assert R._load_earn("sp500", {"A": 1}) == {"A": 1}
    monkeypatch.setattr(R, "REPO", tmp_path)
    assert R._load_earn("hose", {}) == {}                                          # no parquet -> unchanged
    ep = tmp_path / "results" / "gamma_gbm"
    ep.mkdir(parents=True)
    pd.DataFrame({"ticker": ["AAA", "AAA"], "earnings_date": pd.to_datetime(["2021-01-01", "2021-06-01"])}
                 ).to_parquet(ep / "hose_earnings_combined.parquet")
    got = R._load_earn("hose", {})
    assert "AAA" in got and len(got["AAA"]) == 2


# --------------------------------------------------------------------------- run smoke
def test_run_hose_structure_evidence_perfold_spike(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    monkeypatch.setattr(C, "SPIKE_WINDOWS", (("1990-01-01", "1990-01-02"),))       # non-overlapping -> computed
    docs = R.run("hose", load_fn=_fake_loader, cache_fn=_fake_cache, out_dir=tmp_path, smoke=True)
    assert set(docs) == {1}
    doc = docs[1]
    assert (tmp_path / "foundation_hose_smoke_h1.json").exists()
    for blk in ("metrics", "train_metrics", "val_metrics"):
        for m in R.ORDER:
            assert set(doc[blk][m]) == {"mse", "rmse", "mae", "r2", "qlike"}
    assert set(doc["metrics"]) == {R.GBME, R.FND, R.PLAC, R.HAR, R.ZS}
    assert set(doc["dm"]) == {"GBME+FND_vs_GBME", "GBME+PLAC_vs_GBME", "GBME+FND_vs_GBME+PLAC"}
    assert {"qlike_zeroshot", "qlike_har", "dm_p", "zeroshot_beats_har"} <= set(doc["stage0_zeroshot_vs_har"])
    assert "per_fold_qlike" in doc and "spike_robustness" in doc and "placebo" in doc
    # the GBM family carries full train/val/test + fit evidence: every model's recomputed fit verdict is defined
    import overfit_check as OF
    for m in R.ORDER:
        assert OF.classify_fit(doc["train_metrics"][m], doc["val_metrics"][m], doc["metrics"][m])["status"] \
            in {"ok", "overfit", "underfit"}
    # the real pre-push gate (check_overfit_evidence) treats this GBM-family result as a non-learned baseline and
    # SKIPS it (no "masked" design, no metrics_per_seed, no learned model name) -> a coordinator push is not blocked
    import check_overfit_evidence as COE
    assert COE.check_files([str(tmp_path / "foundation_hose_smoke_h1.json")]) == {}


def test_run_sp500_no_spike_block(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    docs = R.run("sp500", load_fn=_fake_loader, cache_fn=_fake_cache, out_dir=tmp_path, smoke=True)
    doc = docs[1]
    assert "per_fold_qlike" not in doc and "spike_robustness" not in doc
    assert (tmp_path / "foundation_sp500_smoke_h1.json").exists()


def test_run_skips_empty_folds_multi(monkeypatch, tmp_path):
    _tiny(monkeypatch)
    monkeypatch.setattr(C, "HORIZONS", (1,))
    docs = R.run("sp500", load_fn=_fake_loader, cache_fn=_fake_cache, out_dir=tmp_path, smoke=False)
    assert docs[1]["n_folds"] >= 1


def test_run_omits_robustness_when_all_test_in_spike(monkeypatch, tmp_path):
    _tiny(monkeypatch)          # default 2022 window covers fold0 test -> keep empty -> spike_robustness omitted
    docs = R.run("hose", load_fn=_fake_loader, cache_fn=_fake_cache, out_dir=tmp_path, smoke=True)
    assert "spike_robustness" not in docs[1] and "per_fold_qlike" in docs[1]


# --------------------------------------------------------------------------- opt-in: the REAL Chronos model
def test_real_chronos_predict_fn_tiny_slice():
    pytest.importorskip("chronos")
    fn = FF.chronos_predict_fn()
    ctx = [np.abs(np.random.default_rng(0).normal(0, 1, 60)) * 1e-4 + 3e-4 for _ in range(3)]
    med, spr = fn(ctx, C.PRED_LEN)
    assert med.shape == (3, C.PRED_LEN) and spr.shape == (3, C.PRED_LEN)
    assert np.isfinite(med).all() and (spr >= 0).all()
