"""Pooled walk-forward runner + GPU-politeness coverage (synthetic panel, STUBBED LSTM training).

``train_masked_rich`` is monkeypatched to a deterministic stub so the whole pool/metric/DM/evidence
plumbing runs on CPU in a fraction of a second (no epochs, no GPU) -- mirrors the sector-ablation
``test_run_training_stubbed`` pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import run_walkforward as RW      # noqa: E402
import wf_panel as WP             # noqa: E402


def _synth_panel(T=600, N=5, lookback=10, horizon=1):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2019-01-01", periods=T, freq="D")
    pk = np.abs(rng.normal(0.01, 0.004, size=(T, N))) + 1e-4
    feats = np.zeros((T, N, 5))
    for j in range(N):
        feats[:, j, 0] = pk[:, j]
        feats[:, j, 1] = pd.Series(pk[:, j]).rolling(5, min_periods=1).mean().to_numpy()
        feats[:, j, 2] = pd.Series(pk[:, j]).rolling(22, min_periods=1).mean().to_numpy()
        feats[:, j, 3] = np.median(np.sqrt(pk), axis=1)
        feats[:, j, 4] = rng.normal(0, 1, size=T)
    anchors = np.arange(lookback + 21, T - horizon)
    ok = np.ones((len(anchors), N), bool)
    target_dates = dates.to_numpy()[anchors + horizon]
    return WP.WFPanel(list("ABCDE"[:N]), dates, pk, feats, anchors, ok.copy(), ok.copy(), ok, target_dates)


def _stub_train():
    # split-distinguishable multiplicative error: train < val < test, so a mis-wired split feed would flip
    # the fit ordering and be caught by the evidence assertions below.
    def fake(D, cfg, seed, use_graph, adj, output_param="zscore_floor", return_splits=False):
        test = D.y_te * 1.08 + 1e-4
        if return_splits:
            return {"test": test, "val": D.y_va * 1.05 + 1e-4, "train": D.y_tr * 1.02 + 1e-5,
                    "train_curve": [1e-6, 5e-7], "val_curve": [1.1e-6, 6e-7], "best_epoch": 2}
        return test
    return fake


@pytest.fixture
def patched(monkeypatch):
    panel = _synth_panel()
    monkeypatch.setattr(RW, "build_wf_panel", lambda *a, **k: panel)
    monkeypatch.setattr(RW.RMR, "train_masked_rich", _stub_train())
    return panel


def test_training_config_overrides():
    cfg = RW.training_config(epochs=16, patience=5, batch=32, seeds=(42, 123))
    assert cfg.epochs == 16 and cfg.patience == 5 and cfg.batch_size == 32
    assert cfg.seeds == (42, 123) and cfg.min_epochs == 5


def test_run_walkforward_pools_and_writes_json(patched, tmp_path):
    wf = RW.WFConfig(lookback=10, horizon=1, K=80, val=40, test_start=400)
    cfg = RW.training_config(epochs=2, patience=1, seeds=(42, 123))
    out = tmp_path / "wf.json"
    res = RW.run_walkforward(["x"], "pdir", wf, cfg, out_path=out, keep_tickers=list("ABCDE"))
    assert out.exists()
    assert res["n_folds"] == len(res["per_fold"]) == 3
    assert res["num_nodes"] == 5
    # pooled OOS dates == union of the tiled forecast blocks
    assert res["n_oos_dates"] == res["metrics_pooled"]["HAR"]["n"] / 5
    for m in ("HAR", "HAR-X", "LSTM"):
        assert {"mse", "rmse", "mae", "qlike", "r2", "n"} <= set(res["metrics_pooled"][m])
    # primary comparison present + finite
    q = res["dm_date_clustered"]["LSTM_vs_HARX"]["qlike"]
    assert np.isfinite(q["p_value"]) and q["favors"] in {"A", "B"}
    assert "LSTM" in res["metrics_per_seed"]
    # over/under-fit evidence: per-fold blocks + summary counts
    for fold_ev in res["per_fold"]:
        for block in ("train_metrics", "val_metrics", "test_metrics", "fit_diagnostics",
                      "lstm_learning_curves"):
            assert block in fold_ev
        # stub error grows train<val<test -> correctly-wired splits yield this MSE ordering
        assert (fold_ev["train_metrics"]["LSTM"]["mse"] < fold_ev["val_metrics"]["LSTM"]["mse"]
                < fold_ev["test_metrics"]["LSTM"]["mse"])
    for mdl in ("HAR", "HAR-X", "LSTM"):
        s = res["fit_summary"][mdl]
        assert s["n_ok"] + s["n_overfit"] + s["n_underfit"] + s["n_unknown"] == 3
    # expanding train windows across folds
    stops = [f["n_train"] for f in res["per_fold"]]
    assert stops[0] < stops[1] < stops[2]


def test_run_walkforward_default_test_start_and_no_out(patched):
    wf = RW.WFConfig(lookback=10, horizon=1, K=80, val=40)   # test_start None -> int(n*0.9)
    cfg = RW.training_config(epochs=2, patience=1, seeds=(42,))
    res = RW.run_walkforward(["x"], "pdir", wf, cfg, out_path=None, keep_tickers=list("ABCDE"))
    assert res["test_start_anchor"] == int(res["n_anchors"] * 0.90)
    assert res["n_folds"] >= 1
    assert res["fixed_split_reference"]["favors"] == "HAR-X"


# -------------------- GPU politeness --------------------

def test_gpu_is_free():
    assert RW.gpu_is_free(0, 100, util_max=15, mem_max_mib=1200)
    assert not RW.gpu_is_free(50, 100, util_max=15, mem_max_mib=1200)   # busy util
    assert not RW.gpu_is_free(0, 2000, util_max=15, mem_max_mib=1200)   # high VRAM


def test_wait_for_gpu_returns_true_after_hold():
    samples = iter([(50, 50), (0, 100), (0, 100), (0, 100)])   # busy then free x3
    calls = []
    ok = RW.wait_for_gpu(query=lambda: next(samples), hold=3, sleep=lambda s: calls.append(s), poll=0.0)
    assert ok is True and len(calls) == 3   # busy(sleep) + free,free(sleep) then 3rd free -> return pre-sleep


def test_wait_for_gpu_gives_up_after_max_polls():
    ok = RW.wait_for_gpu(query=lambda: (99, 9999), hold=3, sleep=lambda s: None, max_polls=4)
    assert ok is False
