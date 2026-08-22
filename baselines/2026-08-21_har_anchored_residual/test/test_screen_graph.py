"""Smoke test for the model-free graph screening runner."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SUB = Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(_SUB))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import screen_graph as SG  # noqa: E402


@pytest.mark.smoke
def test_screen_runs_and_reports_all_signals(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", periods=600)
    d = tmp_path / "vn"; d.mkdir()
    files = []
    for k in range(12):
        pk = np.abs(rng.normal(0.02, 0.006, 600)) + 1e-4
        f = d / f"T{k:02d}_processed.csv"
        pd.DataFrame({"date": dates, "parkinson_volatility": pk}).to_csv(f, index=False)
        files.append(str(f))
    res = SG.screen(files, "vn30", min_common=300)
    assert res["num_nodes"] >= 8
    assert res["horizons"], "no horizon produced results"
    for h, r in res["horizons"].items():
        for key in ("S0_mean", "S1_weighted", "S2_signed", "S3_resid", "S4_leadlag", "PLACEBO_S1"):
            assert key in r and "test_incr_R2" in r[key]
            assert np.isfinite(r[key]["test_incr_R2"])
