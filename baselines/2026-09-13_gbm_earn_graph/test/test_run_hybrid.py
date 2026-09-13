"""End-to-end smoke of run_hybrid on tiny monkeypatched synthetic frames (no real data), plus the
earnings-resolution helper branches, the earnings-required guard, the all-folds-skip / empty-result
guard (default loader branch), and the _success verdict helper."""
import json

import numpy as np
import pandas as pd
import pytest

import config
import run_hybrid


def _fake_frames(n_tickers=25, n_rows=300, seed=0):
    """n_tickers across 3 sectors carrying every column FM.panel / build_graph / graph_feats read."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_rows)
    base = {s: np.abs(rng.standard_normal(n_rows)) * 1e-4 + 1e-6 for s in range(3)}
    frames = {}
    for i in range(n_tickers):
        sec = i % 3
        pv = base[sec] + np.abs(rng.standard_normal(n_rows)) * 5e-5 + 1e-6
        d = pd.DataFrame({"date": dates, "parkinson_variance": pv, "logpk": np.log(pv),
                          "daily_return": rng.standard_normal(n_rows) * 0.01,
                          "volume_zscore_22": rng.standard_normal(n_rows)})
        for c in ["har_daily", "har_weekly", "har_monthly", "rq", "mr_change", "mr_slope5",
                  "mr_slope10", "mr_dev5", "mr_z22"]:
            d[c] = rng.standard_normal(n_rows)
        d["ticker"] = f"T{i}"; d["sector"] = sec
        frames[f"T{i}"] = d
    return frames


def _earn_parquet(tmp_path, n_tickers=25):
    """A tiny earnings-dates parquet covering the synthetic tickers (a couple of dates each)."""
    rows = []
    for i in range(n_tickers):
        for dt in ("2015-04-15", "2015-07-15"):
            rows.append({"ticker": f"T{i}", "earnings_date": pd.Timestamp(dt)})
    p = tmp_path / "hose_earnings_combined.parquet"
    pd.DataFrame(rows).to_parquet(p)
    return p


def _shrink(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 20})


def test_run_hybrid_smoke(monkeypatch, tmp_path):
    """The three hybrid models + graph_only wire end-to-end on synthetic data with a valid scored fold."""
    _shrink(monkeypatch)
    monkeypatch.setattr(run_hybrid, "EARN_PARQUET", _earn_parquet(tmp_path))
    monkeypatch.setattr(run_hybrid.S1, "FOLDS", ["2015-06-01", "2015-09-01", "2015-10-01", "2100-01-01"])
    out = run_hybrid.run_hybrid("hose", load_fn=lambda m: (_fake_frames(), {}, {}))
    assert "h1" in out
    r = out["h1"]
    assert set(r) >= {"n", "qlike", "dm", "err_corr", "gain_vs_earn_pct", "gain_vs_corr_pct",
                      "train_metrics", "test_metrics", "fit_diagnostics"}
    assert set(r["qlike"]) == {"GBM+earn", "GBM+earn+corr", "GBM+earn+graph", "graph_only"}
    assert set(r["dm"]) == {"GBM+earn+graph_vs_GBM+earn", "GBM+earn+graph_vs_GBM+earn+corr"}
    for m in r["qlike"]:
        assert set(r["fit_diagnostics"][m]) == {"verdict", "train_qlike", "test_qlike"}
    assert -1.0 <= r["err_corr"] <= 1.0
    assert isinstance(out["success"], bool)
    json.dumps(out)                                              # JSON-serialisable


def test_out_path_checkpoints_each_horizon(monkeypatch, tmp_path):
    """out_path flushes JSON after every horizon so a Colab disconnect keeps completed horizons."""
    _shrink(monkeypatch)
    monkeypatch.setattr(run_hybrid, "EARN_PARQUET", _earn_parquet(tmp_path))
    monkeypatch.setattr(run_hybrid.S1, "FOLDS", ["2015-06-01", "2015-09-01", "2015-10-01", "2100-01-01"])
    outp = tmp_path / "gbm_earn_graph_hose.json"
    out = run_hybrid.run_hybrid("hose", load_fn=lambda m: (_fake_frames(), {}, {}), out_path=outp)
    assert outp.exists()                                        # written mid-run, not only at the end
    assert "h1" in json.loads(outp.read_text()) and "h1" in out  # checkpoint holds the scored horizon(s)


def test_earn_dates_sp500_keeps_loader_dates():
    ed = {"AAPL": np.array([np.datetime64("2020-01-01")])}
    assert run_hybrid._earn_dates("sp500", ed) is ed             # sp500 branch: untouched


def test_earn_dates_hose_reads_parquet(monkeypatch, tmp_path):
    monkeypatch.setattr(run_hybrid, "EARN_PARQUET", _earn_parquet(tmp_path, n_tickers=2))
    got = run_hybrid._earn_dates("hose", {})
    assert set(got) == {"T0", "T1"} and got["T0"].size == 2      # exists branch: injected from parquet


def test_earn_dates_hose_missing_parquet_returns_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(run_hybrid, "EARN_PARQUET", tmp_path / "nope.parquet")
    assert run_hybrid._earn_dates("hose", {}) == {}              # not-exists branch: falls back to loader


def test_run_hybrid_requires_earnings(monkeypatch, tmp_path):
    """No earnings dates -> fail loud (the hybrid is meaningless without the earnings block)."""
    _shrink(monkeypatch)
    monkeypatch.setattr(run_hybrid, "EARN_PARQUET", tmp_path / "nope.parquet")
    with pytest.raises(ValueError, match="requires the earnings block"):
        run_hybrid.run_hybrid("hose", load_fn=lambda m: (_fake_frames(), {}, {}))


def test_run_hybrid_all_folds_skip_empty_raises(monkeypatch, tmp_path):
    """Every fold below min_rows -> all skipped -> empty result -> fail loud; also exercises the
    ``load_fn or FM.load`` default branch (load_fn=None with FM.load patched)."""
    _shrink(monkeypatch)
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 10**9})
    monkeypatch.setattr(run_hybrid, "EARN_PARQUET", _earn_parquet(tmp_path))
    monkeypatch.setattr(run_hybrid.S1, "FOLDS", ["2015-06-01", "2015-09-01", "2100-01-01"])
    monkeypatch.setattr(run_hybrid.FM, "load", lambda m: (_fake_frames(), {}, {}))
    with pytest.raises(ValueError, match="no fold scored"):
        run_hybrid.run_hybrid("hose", load_fn=None)


def _row(gain, dm_p, q_graph, q_corr, verdict):
    return {"gain_vs_earn_pct": gain, "dm": {"GBM+earn+graph_vs_GBM+earn": dm_p},
            "qlike": {"GBM+earn+graph": q_graph, "GBM+earn+corr": q_corr},
            "fit_diagnostics": {"GBM+earn+graph": {"verdict": verdict}}}


def test_success_verdict_branches():
    assert run_hybrid._success({"success": None}) is False                       # no horizon keys
    good = {"h1": _row(0.5, 0.01, 0.10, 0.11, "ok")}
    assert run_hybrid._success(good) is True
    assert run_hybrid._success({"h1": _row(0.05, 0.01, 0.10, 0.11, "ok")}) is False   # gain below floor
    assert run_hybrid._success({"h1": _row(0.5, 0.5, 0.10, 0.11, "ok")}) is False     # DM not significant
    assert run_hybrid._success({"h1": _row(0.5, 0.01, 0.12, 0.11, "ok")}) is False    # loses to +corr
    assert run_hybrid._success({"h1": _row(0.5, 0.01, 0.10, 0.11, "overfit")}) is False  # overfit
