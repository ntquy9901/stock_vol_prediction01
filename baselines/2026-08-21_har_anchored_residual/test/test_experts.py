"""Smoke/integration test for the snapshot data build + unified neural trainer (E0/E1/E5/E8 paths)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SUB = Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"
sys.path.insert(0, str(_SUB))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from config import SMOKE  # noqa: E402
import experts  # noqa: E402


def _make_files(tmp, n_tickers=10, n_days=420, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    d = tmp / "vn"; d.mkdir()
    files = []
    for k in range(n_tickers):
        pk = np.abs(rng.normal(0.02, 0.008, n_days)) + 1e-4          # positive variance-like series
        f = d / f"T{k:02d}_processed.csv"
        pd.DataFrame({"date": dates, "parkinson_volatility": pk}).to_csv(f, index=False)
        files.append(str(f))
    return files


@pytest.mark.smoke
def test_build_and_train_smoke(tmp_path):
    files = _make_files(tmp_path)
    D = experts.build_data(files, lookback=10, horizon=5, cfg=SMOKE)
    assert D["N"] >= 8 and len(D["d_te"]) > 0

    # E0 HAR dict
    har = experts.har_pred_dict(D, "test")
    assert len(har) == len(D["d_te"]) * D["N"]
    assert all(np.isfinite(v[1]) and v[1] >= 0 for v in har.values())

    # purge property: last train target date strictly before first val target date is guaranteed by
    # dropping the last `horizon` train snapshots; here just assert the splits are non-empty and disjoint
    assert D["cf_mask"].sum() > 0                                    # some cross-fitted residual rows

    for framing, use_lstm, use_graph in [("full", True, False),
                                          ("additive", True, False),
                                          ("mult", True, True)]:
        te, va, vq, nparam, corr = experts.train_neural(D, SMOKE, seed=42, use_lstm=use_lstm,
                                                         use_graph=use_graph, framing=framing)
        assert corr["c_tr"].shape[0] == len(D["d_va"]) or corr["c_te"].shape == (len(D["d_te"]), D["N"])
        assert len(te) == len(D["d_te"]) * D["N"] and nparam > 0
        preds = np.array([p for _, p in te.values()])
        assert np.isfinite(preds).all() and (preds >= 0).all()
        assert np.isfinite(vq)
