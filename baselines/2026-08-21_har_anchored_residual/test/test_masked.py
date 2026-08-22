"""Smoke test for the masked union-date panel builder + runner (staggered listing dates)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SUB = Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(_SUB))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from config import SMOKE  # noqa: E402
import masked_snapshots as MS  # noqa: E402
import run_masked as RM  # noqa: E402


def _make_files(tmp, n=12, base_days=520, seed=0):
    rng = np.random.default_rng(seed)
    all_dates = pd.bdate_range("2018-01-01", periods=base_days)
    d = tmp / "vn"; d.mkdir()
    files = []
    for k in range(n):
        start = 0 if k < n - 3 else rng.integers(80, 200)      # last 3 tickers list late -> ragged union
        dts = all_dates[start:]
        pk = np.abs(rng.normal(0.02, 0.006, len(dts))) + 1e-4
        f = d / f"T{k:02d}_processed.csv"
        pd.DataFrame({"date": dts, "parkinson_volatility": pk}).to_csv(f, index=False)
        files.append(str(f))
    return files


def test_build_masked_union_and_masks(tmp_path):
    files = _make_files(tmp_path)
    D = MS.build_masked(files, lookback=10, horizon=5, min_valid=6, min_train_rows=30)
    assert D.N == 12
    # masks are boolean-ish and target-valid implies at least min_valid nodes per kept anchor
    assert D.tmask_te.shape[1] == 12
    assert (D.tmask_tr.sum(1) >= 6).all()
    # late-listed tickers must be invalid on some early train anchors (ragged panel exercised)
    assert D.nmask_tr.min() == 0.0
    # invalid feature rows are zero-filled, not NaN
    assert np.isfinite(D.X_tr).all() and np.isfinite(D.X_te).all()


@pytest.mark.smoke
def test_run_masked_all_metrics_and_dm(tmp_path):
    files = _make_files(tmp_path)
    res = RM.run("vn30", files, horizon=5, cfg=SMOKE, lookback=10)
    for m in ("HAR", "LSTM", "LSTM_GAT"):
        for metric in ("mse", "rmse", "mae", "qlike", "r2"):
            assert np.isfinite(res["metrics"][m][metric])
    for cmp in ("LSTM_vs_HAR", "LSTM_GAT_vs_HAR", "LSTM_GAT_vs_LSTM"):
        for fam in ("qlike", "se", "ae"):
            assert fam in res["dm_date_clustered"][cmp]
    assert res["n_test_dates"] > 0
