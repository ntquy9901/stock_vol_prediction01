"""TDD tests for the classical econometric baseline suite.

Covers: metric-correctness vs the ladder scorer (round-trip fidelity), obs-alignment / coverage,
per-baseline math (persistence, EWMA recursion, log-HAR positivity, HARQ design), and a per-ticker
GARCH fit smoke on a tiny real-data slice.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
_ROOT = Path(__file__).resolve().parents[3]
_TRACKB = _ROOT / "submission_track_b" / "trackb_code"
for _p in (str(_CODE), str(_TRACKB), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import classical_baselines as cb  # noqa: E402
from scaling import PreprocessorStore, TickerPreprocessor  # noqa: E402


def _sample(ticker_id, ticker, target_date, obs_date, y):
    return SimpleNamespace(
        key=SimpleNamespace(ticker_id=ticker_id, ticker=ticker, target_date=target_date),
        input_dates=(obs_date,), y_eval_raw=y, y_raw=y,
    )


def _store_two_tickers():
    dates = [f"2020-01-{d:02d}" for d in range(1, 31)]
    frame0 = pd.DataFrame({"date": dates,
                           "parkinson_variance": np.linspace(0.01, 0.02, 30)})
    frame1 = pd.DataFrame({"date": dates,
                           "parkinson_variance": np.linspace(0.03, 0.05, 30)})
    pp0 = TickerPreprocessor.fit(frame0, ["parkinson_variance"], "parkinson_variance")
    pp1 = TickerPreprocessor.fit(frame1, ["parkinson_variance"], "parkinson_variance")
    return PreprocessorStore({0: pp0, 1: pp1})


def test_metric_correctness_matches_ladder_scorer():
    """evaluate_baseline (round-trip through the store) reproduces evaluate_records exactly."""

    from train import evaluate_records

    store = _store_two_tickers()
    samples = [
        _sample(0, "AAA", "2021-01-01", "2020-12-25", 0.012),
        _sample(0, "AAA", "2021-01-02", "2020-12-28", 0.015),
        _sample(0, "AAA", "2021-01-03", "2020-12-29", 0.011),
        _sample(1, "BBB", "2021-01-01", "2020-12-25", 0.041),
        _sample(1, "BBB", "2021-01-02", "2020-12-28", 0.039),
        _sample(1, "BBB", "2021-01-03", "2020-12-29", 0.044),
    ]
    preds = {(0, "2021-01-01"): 0.013, (0, "2021-01-02"): 0.014, (0, "2021-01-03"): 0.012,
             (1, "2021-01-01"): 0.040, (1, "2021-01-02"): 0.042, (1, "2021-01-03"): 0.043}
    got = cb.evaluate_baseline(preds, samples, store)

    # Independent expectation: score the same raw preds directly through evaluate_records.
    records = []
    for s in samples:
        pred_raw = preds[(s.key.ticker_id, s.key.target_date)]
        scaler = store.get(s.key.ticker_id).target_scaler
        records.append({
            "ticker_id": s.key.ticker_id, "target_date": s.key.target_date,
            "prediction_norm": float(scaler.transform(np.array([pred_raw]))[0]),
            "target_raw": s.y_eval_raw,
        })
    expected = evaluate_records(records, store)["metrics"]
    for key in ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy"):
        assert got[key] == pytest.approx(expected[key], abs=1e-12)


def test_roundtrip_is_lossless_on_raw_scale():
    """The scaler transform->inverse round-trip preserves the raw prediction (metric fidelity)."""

    store = _store_two_tickers()
    for tid in (0, 1):
        scaler = store.get(tid).target_scaler
        raw = 0.037
        back = float(scaler.inverse_transform(scaler.transform(np.array([raw])))[0])
        assert back == pytest.approx(raw, abs=1e-12)


def test_assert_full_coverage_detects_missing():
    samples = [_sample(0, "AAA", "2021-01-01", "2020-12-25", 0.012),
               _sample(0, "AAA", "2021-01-02", "2020-12-28", 0.015)]
    cb.assert_full_coverage({(0, "2021-01-01"): 0.1, (0, "2021-01-02"): 0.1}, samples)
    with pytest.raises(ValueError):
        cb.assert_full_coverage({(0, "2021-01-01"): 0.1}, samples)


def _fake_series(ticker="AAA"):
    dates = [f"2020-01-{d:02d}" for d in range(1, 31)]
    vol = np.linspace(0.01, 0.03, 30)
    series = pd.Series(vol)
    return cb.TickerSeries(
        ticker=ticker, dates=dates, vol=vol, sigma_d=vol.copy(),
        sigma_w=series.rolling(5).mean().to_numpy(),
        sigma_m=series.rolling(22).mean().to_numpy(),
        ewma_pred=cb._ewma_variance(vol, cb.EWMA_LAMBDA),
        train_end_date="2020-01-21", pos={d: i for i, d in enumerate(dates)},
        logret=pd.Series(dtype=float),
    )


def test_persistence_returns_today_vol():
    series = {"AAA": _fake_series()}
    samples = [_sample(0, "AAA", "2020-02-01", "2020-01-25", 0.02)]
    preds = cb.predict_persistence(samples, series)
    assert preds[(0, "2020-02-01")] == pytest.approx(series["AAA"].vol[24])


def test_ewma_recursion_matches_manual():
    rv = np.array([0.0002, 0.0003, 0.0001, 0.0004], dtype=float)
    smoothed = cb._ewma_variance(rv, 0.94)
    manual = [rv[0]]
    for i in range(1, 4):
        manual.append(0.94 * manual[-1] + 0.06 * rv[i])
    assert smoothed == pytest.approx(np.array(manual))


def test_log_har_predictions_positive():
    series = {"AAA": _fake_series()}
    train = [_sample(0, "AAA", f"2020-01-{d:02d}", f"2020-01-{d-5:02d}", 0.01 + 0.001 * d)
             for d in range(23, 30)]
    ev = [_sample(0, "AAA", "2020-02-01", "2020-01-28", 0.02)]
    preds = cb.predict_log_har(train, ev, series)
    assert preds[(0, "2020-02-01")] > 0


def test_harq_design_has_quarticity_term():
    series = _fake_series()
    design = cb._har_design(series, "2020-01-25", "harq")
    assert design.shape == (4,)
    d = series.sigma_d[24]
    assert design[3] == pytest.approx(d * np.sqrt(d * d))


def test_har_design_rejects_unknown_kind():
    with pytest.raises(ValueError):
        cb._har_design(_fake_series(), "2020-01-25", "bogus")


def test_har_and_harq_wrappers_predict():
    series = {"AAA": _fake_series()}
    train = [_sample(0, "AAA", f"2020-01-{d:02d}", f"2020-01-{d-5:02d}", 0.01 + 0.001 * d)
             for d in range(23, 30)]
    ev = [_sample(0, "AAA", "2020-02-01", "2020-01-28", 0.02)]
    assert (0, "2020-02-01") in cb.predict_har(train, ev, series)
    assert (0, "2020-02-01") in cb.predict_harq(train, ev, series)


def test_load_log_returns_without_ohlcv_is_empty():
    """A ticker without a raw OHLCV file yields no returns (GARCH-ineligible branch)."""

    assert cb._load_log_returns("NONEXISTENT_TICKER_XYZ").empty


def test_garch_rejects_too_few_returns():
    series = _fake_series()
    series.logret = pd.Series(np.full(30, 0.01), index=series.dates[:30])
    with pytest.raises(ValueError):
        cb.garch_ticker_forecast(series, series.dates[24:27], horizon=5, spec_name="garch")


def test_garch_requires_at_least_one_trading_day_origin():
    """If no requested origin is a trading day, the forecast raises rather than guessing."""

    dates = pd.bdate_range("2020-01-01", periods=200).strftime("%Y-%m-%d").tolist()
    r = pd.Series(np.random.default_rng(1).normal(0.0, 0.01, len(dates)), index=dates)
    vol = np.abs(r.to_numpy())
    series = cb.TickerSeries(
        ticker="X", dates=dates, vol=vol, sigma_d=vol, sigma_w=vol, sigma_m=vol,
        ewma_pred=vol, train_end_date=dates[120], pos={d: i for i, d in enumerate(dates)}, logret=r)
    with pytest.raises(ValueError):
        cb.garch_ticker_forecast(series, ["1800-01-01", "1800-01-02"], horizon=5, spec_name="garch")


def test_garch_carry_forward_over_calendar_gap():
    """An origin on a non-trading day (calendar gap, e.g. LPB holidays) is carried forward, not dropped."""

    dates = pd.bdate_range("2020-01-01", periods=200).strftime("%Y-%m-%d").tolist()
    gap = dates.pop(150)  # a date absent from this ticker's trading calendar
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0, 0.01, len(dates)), index=dates)
    vol = np.abs(r.to_numpy())
    series = cb.TickerSeries(
        ticker="X", dates=dates, vol=vol, sigma_d=vol, sigma_w=vol, sigma_m=vol,
        ewma_pred=vol, train_end_date=dates[120], pos={d: i for i, d in enumerate(dates)}, logret=r)
    fc = cb.garch_ticker_forecast(series, [dates[140], gap, dates[160]], horizon=5, spec_name="garch")
    assert {dates[140], gap, dates[160]}.issubset(fc)
    assert np.isfinite(fc[gap]) and fc[gap] > 0


def test_assert_full_coverage_detects_extra_keys():
    samples = [_sample(0, "AAA", "2021-01-01", "2020-12-25", 0.01)]
    with pytest.raises(ValueError):
        cb.assert_full_coverage({(0, "2021-01-01"): 0.1, (9, "2021-01-01"): 0.1}, samples)


def test_garch_smoke_real_ticker_slice():
    """Fit GARCH(1,1) on a real ticker's returns and forecast a few origins (finite, positive)."""

    proc = cb.data_root() / "processed" / "ACB_processed.csv"
    ohlcv = cb.data_root() / "raw" / "prices" / "ACB_ohlcv.csv"
    if not (proc.exists() and ohlcv.exists()):
        pytest.skip("ACB real-data slice not available")
    series = cb.load_ticker_series("ACB")
    # Origins just after the train boundary (val region), a handful for speed.
    val_origins = [d for d in series.dates if d > series.train_end_date][:5]
    for spec in ("garch", "gjr", "egarch"):  # egarch exercises the simulation path
        fc = cb.garch_ticker_forecast(series, val_origins, horizon=5, spec_name=spec)
        assert set(val_origins).issubset(fc)
        for d in val_origins:
            assert np.isfinite(fc[d]) and fc[d] > 0


def test_ohlcv_dates_normalized_to_date_only():
    """VPB/VRE store tz-aware timestamps; returns index must be plain YYYY-MM-DD to align keys."""

    for ticker in ("VPB", "VRE", "ACB"):
        if not (cb.data_root() / "raw" / "prices" / f"{ticker}_ohlcv.csv").exists():
            pytest.skip(f"{ticker} OHLCV not available")
        idx = list(cb.load_ticker_series(ticker).logret.index)
        assert idx, f"{ticker} produced no returns"
        assert all(len(d) == 10 and d[4] == "-" and d[7] == "-" for d in idx[:20])


def _fake_payload():
    metrics = {"mse": 1e-6, "rmse": 1e-3, "mae": 5e-4, "r2": 0.7, "qlike": 0.5,
               "directional_accuracy": 48.5}
    results = {name: {"val": dict(metrics), "test": dict(metrics),
                      "n_val_obs": 14418, "n_test_obs": 14464}
               for name in ("Persistence", "EWMA", "HAR", "HARQ", "logHAR")}
    for name in ("GARCH", "GJR-GARCH", "EGARCH"):
        results[name] = {"val": dict(metrics), "test": dict(metrics),
                         "n_val_obs": 14247, "n_test_obs": 14292}
    return {"timestamp": "2026-01-01_000000", "horizon": 5, "n_val_obs": 14418,
            "n_test_obs": 14464, "n_tickers": 33, "garch_excluded_tickers": ["LPB"],
            "garch_n_val_obs": 14247, "garch_n_test_obs": 14292, "results": results,
            "notes": {"basis_note": "basis", "target_units": "variance"}}


def test_cli_ladder_schema_and_markdown():
    """The report builders emit the ladder-compatible schema and a 6-metric table for every row."""

    import run_classical_baselines as cli

    payload = _fake_payload()
    schema = cli.to_ladder_schema(payload)
    assert set(schema["rung_metrics"]["val"]) == set(payload["results"])
    for row in schema["rung_metrics"]["val"].values():
        assert set(("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")).issubset(row)
        assert "n_obs" in row
    md = cli.to_markdown(payload)
    for name in payload["results"]:
        assert name in md
    assert "VAL metrics" in md and "TEST metrics" in md


def test_cli_skips_absent_baselines():
    """to_ladder_schema / to_markdown skip rows not present in results (partial-run robustness)."""

    import run_classical_baselines as cli

    payload = _fake_payload()
    payload["results"].pop("EGARCH")
    payload["results"].pop("GJR-GARCH")
    schema = cli.to_ladder_schema(payload)
    assert "EGARCH" not in schema["rung_metrics"]["val"]
    assert "EGARCH" not in cli.to_markdown(payload)


def _series_for(ticker: str) -> "cb.TickerSeries":
    dates = [f"2020-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(60)]
    base = 0.0002 if ticker == "AAA" else 0.0004
    vol = base + 1e-5 * np.sin(np.arange(60))
    series = pd.Series(vol)
    return cb.TickerSeries(
        ticker=ticker, dates=dates, vol=vol, sigma_d=vol.copy(),
        sigma_w=series.rolling(5).mean().to_numpy(), sigma_m=series.rolling(22).mean().to_numpy(),
        ewma_pred=cb._ewma_variance(vol, cb.EWMA_LAMBDA), train_end_date=dates[41],
        pos={d: i for i, d in enumerate(dates)},
        logret=pd.Series(np.full(60, 0.01), index=dates),  # non-empty -> GARCH eligible
    )


def _synthetic_manifest():
    tickers = {"AAA": 0, "BBB": 1}

    def mk(ticker, tid, origin_i, target_i):
        s = _series_for(ticker)
        return SimpleNamespace(
            key=SimpleNamespace(ticker_id=tid, ticker=ticker, target_date=s.dates[target_i]),
            input_dates=(s.dates[origin_i],), y_eval_raw=float(s.vol[target_i]),
            y_raw=float(s.vol[target_i]))

    def split(origins):
        return tuple(mk(t, tid, i, i + 5) for t, tid in tickers.items() for i in origins)

    return SimpleNamespace(samples={"train": split([25, 27, 30, 33]),
                                    "val": split([40, 43, 46]), "test": split([48, 50, 52])})


def test_run_all_and_cli_integration(monkeypatch, tmp_path):
    """Integration: run_all scores every baseline; CLI main writes the canonical JSON + MD."""

    import run_classical_baselines as cli

    store = _store_two_tickers()
    monkeypatch.setattr(cb, "build_manifest", lambda horizon=5: (_synthetic_manifest(), store))
    monkeypatch.setattr(cb, "load_ticker_series", _series_for)
    monkeypatch.setattr(cb, "garch_ticker_forecast",
                        lambda series, origins, horizon, spec, seed=42: {d: 3e-4 for d in origins})

    payload = cb.run_all(horizon=5)
    expected = {"Persistence", "EWMA", "HAR", "HARQ", "logHAR", "GARCH", "GJR-GARCH", "EGARCH"}
    assert set(payload["results"]) == expected
    for row in payload["results"].values():
        assert {"val", "test", "n_val_obs", "n_test_obs"}.issubset(row)
        assert np.isfinite(row["val"]["qlike"]) and np.isfinite(row["test"]["rmse"])

    monkeypatch.setattr(cli, "_ROOT", tmp_path)
    cli.main("2020-01-01_000000", 5)
    out = tmp_path / "docs" / "reports"
    assert (out / "classical_baselines_h5_2020-01-01_000000.json").exists()
    assert (out / "classical_baselines_h5_2020-01-01_000000.md").exists()


@pytest.mark.smoke
def test_obs_alignment_matches_ladder_counts():
    """Smoke: the pooled val/test observation set equals the ladder's 14418 / 14464 at h5."""

    manifest, _store = cb.build_manifest(horizon=5)
    assert len(manifest.samples["val"]) == 14418
    assert len(manifest.samples["test"]) == 14464
