"""``run_training`` orchestration on a tiny real HNX slice with the DEEP training stubbed (no GPU, no
epochs): the metric / DM / over-under-fit plumbing and the real HAR/HAR-X anchors run on CPU in a fraction
of a second. HAR/HAR-X are computed for real (not stubbed) so ``_har_context`` is covered.
"""
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("GWN_FORCE_CPU", "1")
_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import run_gwn_ablation as R  # noqa: E402
from config import Config, SMOKE  # noqa: E402

PANEL = "hnx"


def _tiny_D():
    keep = R.EFA.screened_tickers(PANEL)
    if keep is None:
        pytest.skip(f"{PANEL} panel not available")
    keep = set(sorted(keep)[:8])
    return R.build_panel(PANEL, SMOKE, horizon=1, out_dir=tempfile.mkdtemp(), keep_tickers=keep)


def _stub_lstm(D, cfg, seed, use_graph, adj, output_param="zscore_floor", return_splits=False):
    goff = 0.0 if not use_graph else 1e-7
    test = D.y_te + 1e-5 + goff
    if return_splits:
        return {"test": test, "val": D.y_va + 5e-6 + goff, "train": D.y_tr + 1e-6 + goff,
                "train_curve": [1e-6, 5e-7], "val_curve": [1.1e-6, 6e-7], "best_epoch": 2}
    return test


def _stub_gwn(D, cfg, seed, adaptive, bs, skip_channels=64, end_channels=128,
             residual_channels=32, node_dim=10, return_splits=False):
    goff = 3e-7 if adaptive else 4e-7
    test = D.y_te + 1e-5 + goff
    if return_splits:
        return {"test": test, "val": D.y_va + 5e-6 + goff, "train": D.y_tr + 1e-6 + goff,
                "train_curve": [2e-6, 9e-7], "val_curve": [2.1e-6, 1e-6], "best_epoch": 2}
    return test


def _run(monkeypatch, tmp_path, out_dir):
    D = _tiny_D()
    monkeypatch.setattr(R.RMR, "train_masked_rich", _stub_lstm)
    monkeypatch.setattr(R, "train_gwn", _stub_gwn)
    monkeypatch.setattr(R, "build_panel",
                        lambda panel, cfg, horizon, td, keep_tickers=None: D)
    cfg = replace(Config(), seeds=(42, 123), epochs=2, min_epochs=1, patience=1)
    return D, R.run_training(PANEL, cfg, horizon=1, gwn_batch=8, out_dir=out_dir)


def test_run_training_writes_result_and_metrics(monkeypatch, tmp_path):
    D, res = _run(monkeypatch, tmp_path, str(tmp_path))
    expected = {"HAR", "HAR-X", "LSTM", "LSTM_wGAT_vol2pk", "GWN_adaptive", "GWN_no_adaptive"}
    assert set(res["metrics"]) == expected
    assert res["num_nodes"] == D.N
    assert res["device"] in {"cpu", "gpu"}
    assert (tmp_path / "graphwavenet_ablation_hnx_h1.json").exists()
    for m in res["metrics"].values():
        assert np.isfinite(m["qlike"])
        assert {"mse", "rmse", "mae", "qlike", "r2"} <= set(m)


def test_run_training_no_out_dir_writes_nothing(monkeypatch, tmp_path):
    _, res = _run(monkeypatch, tmp_path, None)          # out_dir=None branch
    assert not list(tmp_path.iterdir())
    assert "metrics" in res


def test_run_training_dm_and_gate_keys(monkeypatch, tmp_path):
    _, res = _run(monkeypatch, tmp_path, str(tmp_path))
    dm = res["dm_date_clustered"]
    assert {"GWN_adaptive_vs_GWN_no_adaptive", "GWN_adaptive_vs_LSTM", "GWN_no_adaptive_vs_LSTM",
            "GWN_adaptive_vs_HAR", "GWN_adaptive_vs_HARX", "GWN_no_adaptive_vs_HAR"} == set(dm)
    # gate keys (OF.LEARNED) present with fit evidence
    for k in ("LSTM", "LSTM_wGAT_vol2pk"):
        assert k in res["train_metrics"] and k in res["val_metrics"] and k in res["fit_diagnostics"]


def test_run_training_overfit_evidence_structure(monkeypatch, tmp_path):
    _, res = _run(monkeypatch, tmp_path, str(tmp_path))
    learned = {"LSTM", "LSTM_wGAT_vol2pk", "GWN_adaptive", "GWN_no_adaptive"}
    for block in ("train_metrics", "val_metrics", "fit_diagnostics", "learning_curves"):
        assert block in res
    assert learned <= set(res["fit_diagnostics"])
    assert {"HAR", "HAR-X"} <= set(res["train_metrics"])       # deterministic anchors carry train/val too
    for k in learned:
        assert {"qlike", "r2"} <= set(res["train_metrics"][k])
        assert res["fit_diagnostics"][k]["status"] in {"ok", "overfit", "underfit", "unknown"}
        lc = res["learning_curves"][k]
        assert len(lc["train"]) == 2 and len(lc["best_epoch"]) == 2
        assert isinstance(lc["train"][0], list)
        # stub error grows train<val<test -> correctly-wired split feed yields this MSE ordering
        assert res["train_metrics"][k]["mse"] < res["val_metrics"][k]["mse"] < res["metrics"][k]["mse"]


def test_build_panel_too_few_raises(monkeypatch):
    monkeypatch.setattr(R.EFA, "_write_estimator_processed", lambda *a, **k: ["only.csv"])
    with pytest.raises(RuntimeError):
        R.build_panel(PANEL, SMOKE, horizon=1, out_dir="x", keep_tickers={"AAA"})
