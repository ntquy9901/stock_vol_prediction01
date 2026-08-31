"""Smoke: full E0-E10 orchestrator on synthetic data with the SMOKE config."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SUB = Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(_SUB))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from config import SMOKE  # noqa: E402
import run_experiment as RX  # noqa: E402


def _make_files(tmp, n_tickers=12, n_days=440, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    d = tmp / "vn"; d.mkdir()
    files = []
    for k in range(n_tickers):
        base = np.abs(rng.normal(0.02, 0.006, n_days))
        pk = base + 0.3 * np.roll(base, 1) + 1e-4       # mild persistence so HAR is meaningful
        f = d / f"T{k:02d}_processed.csv"
        pd.DataFrame({"date": dates, "parkinson_variance": pk}).to_csv(f, index=False)
        files.append(str(f))
    return files


@pytest.mark.smoke
def test_orchestrator_smoke(tmp_path):
    files = _make_files(tmp_path)
    out = tmp_path / "out"
    res = RX.run("vn30", files, lookback=10, horizon=5, cfg=SMOKE, out_dir=out)
    # all ladder ids present with finite qlike
    for name in ("E0_HAR", "E1", "E2", "E5", "E6", "E7", "E8", "E3_blend", "E9_gate_static", "E10_gate_dyn"):
        assert name in res["metrics"], name
        assert np.isfinite(res["metrics"][name]["qlike"])
    assert 0.0 <= res["alpha_E3"] <= 1.0
    assert res["lambda_E9_static"] >= 0.0
    assert "mcs_set" in res["mcs"]
    assert (out / "result.json").exists() and (out / "row_predictions.csv").exists()
    # DM vs HAR present for a hybrid
    assert "dm_qlike" in res["dm_vs_har"]["E7"]
    assert "residual_r2_oos" in res["diagnostics"]
