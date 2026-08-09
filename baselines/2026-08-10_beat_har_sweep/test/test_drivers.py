"""Unit tests for the report / analysis / data-loader drivers and validation branches.

Covers build_report.summarize, analyze.main (cached-P0 path, no basis rebuild),
spillover.load_train_volatility_panel, and the input-validation raises in qlike_torch / adjacency_ops /
spillover — the non-GPU driver surface. The GPU orchestration in sweep.py (run_all / main /
build_backbone_path / load_backbone) is a run entry point exercised end-to-end by the real sweep and is
outside unit scope (see code_review F7 / the results report).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

_CODE = Path(__file__).resolve().parents[1] / "code"
_PILOT = Path(__file__).resolve().parents[2] / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_CODE), str(_PILOT), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adjacency_ops  # noqa: E402
import analyze  # noqa: E402
import build_report  # noqa: E402
import qlike_torch  # noqa: E402
import spillover  # noqa: E402


# ---------------------------------------------------------------- fixtures

def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _results_payload(qlike, rmse):
    metrics = {"mse": rmse ** 2, "rmse": rmse, "mae": rmse * 0.3, "r2": 0.76,
               "qlike": qlike, "directional_accuracy": 48.3}
    return {"validation_metrics": {**metrics, "qlike": qlike - 0.06}, "test_metrics": metrics}


def _keys(n):
    return [f"{i % 3}:2022-{i:04d}" for i in range(n)]


# ---------------------------------------------------------------- build_report

def test_build_report_summarize(tmp_path, monkeypatch):
    monkeypatch.setattr(build_report, "_ROOT", tmp_path)
    ts = "T"
    results = tmp_path / "results"
    for c, q in (("C1", 0.573), ("C2", 0.566), ("C3", 0.591), ("C6", 0.590)):
        for seed in (42, 123, 2026):
            _write(results / f"beat_har_{c}_{ts}" / f"seed{seed}" / "results.json",
                   _results_payload(q, 0.00229))
    for k in ("k8", "k16"):
        for seed in (42, 123, 2026):
            _write(results / f"beat_har_C5_{ts}" / f"seed{seed}" / k / "results.json",
                   _results_payload(0.575, 0.00230))
    _write(results / f"beat_har_sweep_{ts}" / "analysis.json",
           {"configs": {"C5": {"best_k": "k16"}}})
    out = build_report.summarize(ts)
    assert out["configs"]["C1"]["test"]["qlike"]["mean"] == pytest.approx(0.573)
    assert out["configs"]["C5"]["best_k"] == "k16"
    assert "k8" in out["configs"]["C5"]["per_k"]
    assert (results / f"beat_har_sweep_{ts}" / "report_summary.json").exists()


def test_build_report_marks_missing_config(tmp_path, monkeypatch):
    monkeypatch.setattr(build_report, "_ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    out = build_report.summarize("Z")
    assert out["configs"]["C1"]["status"] == "no_results"


# ---------------------------------------------------------------- analyze.main

def test_analyze_main_cached_p0(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_ROOT", tmp_path)
    ts = "T"
    results = tmp_path / "results"
    n = 90
    keys = _keys(n)
    rng = np.random.default_rng(0)
    targets = rng.uniform(2e-3, 4e-3, size=n)
    p0_pred = targets * rng.uniform(1.05, 1.12, size=n)
    # cached P0 files (string keys)
    for split in ("val", "test"):
        _write(results / f"beat_har_sweep_{ts}" / "P0" / split / "results.json",
               {"ordered_validation_keys": keys, "targets_raw": targets.tolist(),
                "predictions_raw": p0_pred.tolist()})
    # one config beating P0 on every seed
    for seed in (42, 123, 2026):
        cp = targets * rng.uniform(1.001, 1.01, size=n)
        rows = [{"ticker_id": int(k.split(":")[0]), "target_date": k.split(":")[1],
                 "target_raw": t, "prediction_raw": p} for k, t, p in zip(keys, targets, cp)]
        _write(results / f"beat_har_C1_{ts}" / f"seed{seed}" / "predictions_test.json", rows)
    out_path = analyze.main(ts, "cpu", ["C1"])
    analysis = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert analysis["configs"]["C1"]["beats_P0_qlike_dm"] is True


def test_analyze_main_no_results(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "_ROOT", tmp_path)
    ts = "T"
    results = tmp_path / "results"
    keys = _keys(10)
    for split in ("val", "test"):
        _write(results / f"beat_har_sweep_{ts}" / "P0" / split / "results.json",
               {"ordered_validation_keys": keys, "targets_raw": [2e-3] * 10,
                "predictions_raw": [2.1e-3] * 10})
    out_path = analyze.main(ts, "cpu", ["C2"])
    analysis = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert analysis["configs"]["C2"]["status"] == "no_results"


# ---------------------------------------------------------------- spillover loader

def test_load_train_volatility_panel(tmp_path):
    dates = pd.date_range("2020-01-01", periods=60).strftime("%Y-%m-%d")
    rng = np.random.default_rng(0)
    for t in ("AAA", "BBB"):
        pd.DataFrame({"date": dates, "parkinson_volatility": rng.uniform(1e-4, 3e-4, size=60)}).to_csv(
            tmp_path / f"{t}_processed.csv", index=False)
    panel = spillover.load_train_volatility_panel(tmp_path, ["AAA", "BBB"], "2020-02-10")
    assert panel.shape[1] == 2 and panel.shape[0] > 4
    assert np.isfinite(panel).all()


def test_load_train_volatility_panel_rejects_short(tmp_path):
    pd.DataFrame({"date": ["2020-01-01", "2020-01-02"],
                  "parkinson_volatility": [1e-4, 2e-4]}).to_csv(tmp_path / "AAA_processed.csv", index=False)
    pd.DataFrame({"date": ["2020-01-01", "2020-01-02"],
                  "parkinson_volatility": [1e-4, 2e-4]}).to_csv(tmp_path / "BBB_processed.csv", index=False)
    with pytest.raises(ValueError):
        spillover.load_train_volatility_panel(tmp_path, ["AAA", "BBB"], "2020-02-10")


# ---------------------------------------------------------------- validation branches

def test_qlike_rejects_bad_shapes():
    ok = torch.ones(2, 3)
    with pytest.raises(ValueError):  # mean/std shape mismatch
        qlike_torch.snapshot_qlike_loss(ok, ok, torch.ones(2, 4), torch.ones(2, 3), torch.ones(2, 3).bool())
    with pytest.raises(ValueError):  # presence shape mismatch
        qlike_torch.snapshot_qlike_loss(ok, ok, ok, ok, torch.ones(2, 4).bool())
    with pytest.raises(ValueError):  # non-positive eps
        qlike_torch.snapshot_qlike_loss(ok, ok, ok, ok, torch.ones(2, 3).bool(), eps=0.0)


def test_qlike_rejects_empty_present_row():
    ok = torch.ones(1, 3)
    with pytest.raises(ValueError):
        qlike_torch.snapshot_qlike_loss(ok, ok, ok, ok, torch.zeros(1, 3).bool())


def test_mask_static_rejects_no_present():
    static = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError):
        adjacency_ops.mask_static_adjacency(static, np.zeros(3))


def test_learned_adjacency_no_sparsify_when_k_covers_all():
    module = adjacency_ops.LearnedAdjacency(num_nodes=3, dim=4, top_k=5)  # k>=n-1 -> no threshold path
    adjacency = module()
    assert adjacency.shape == (3, 3)
