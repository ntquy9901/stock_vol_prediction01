"""Smoke test for the extended model-free feature/volume-graph screening runner."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SUB = Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(_SUB))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import screen_features as SF  # noqa: E402


@pytest.mark.smoke
def test_screen_features_runs_and_reports(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", periods=600)
    proc = tmp_path / "proc" / "vn30"; proc.mkdir(parents=True)
    raw = tmp_path / "raw" / "data" / "raw" / "prices"; raw.mkdir(parents=True)
    files = []
    for k in range(12):
        pk = np.abs(rng.normal(0.02, 0.006, 600)) + 1e-4
        close = 100 + np.cumsum(rng.normal(0, 1, 600))
        vol = np.abs(rng.normal(1e6, 3e5, 600)) + 1.0
        f = proc / f"T{k:02d}_processed.csv"
        pd.DataFrame({"date": dates, "parkinson_volatility": pk}).to_csv(f, index=False)
        pd.DataFrame({"date": dates, "open": close, "high": close + 1,
                      "low": close - 1, "close": close, "volume": vol}).to_csv(
            raw / f"T{k:02d}_ohlcv.csv", index=False)
        files.append(str(f))

    res = SF.screen(files, "vn30", raw.parents[2], min_common=300)
    assert res["num_nodes"] >= 8
    assert res["n_tickers_with_volume"] == 12
    assert res["horizons"], "no horizon produced results"
    for h, r in res["horizons"].items():
        for key in ("S5_vshock_weighted", "S5_vshock_mean", "PLACEBO_S5"):
            assert key in r["S5"] and np.isfinite(r["S5"][key]["test_incr_R2"])
        for key in ("own_return", "market_pk", "vol_ratio", "own_vshock", "ALL_richer"):
            assert key in r["richer"] and np.isfinite(r["richer"][key]["test_incr_R2"])


@pytest.mark.smoke
def test_causal_zscore_is_causal():
    """A spike at row t must not change the z-score of any earlier row (no lookahead)."""
    x = np.ones((60, 2)) + np.linspace(0, 0.1, 60)[:, None]
    z0 = SF._causal_zscore(x.copy(), win=22)
    x2 = x.copy(); x2[50] += 100.0
    z1 = SF._causal_zscore(x2, win=22)
    assert np.allclose(z0[:50], z1[:50], equal_nan=True)
