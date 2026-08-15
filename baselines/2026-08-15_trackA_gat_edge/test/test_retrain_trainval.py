import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import run_retrain_trainval as rr  # noqa: E402

N, SEQ, NEWS_DIM = 4, 22, 146


class _Node:
    def __init__(self, tid: int) -> None:
        self.ticker_id = tid
        self.y_norm = 0.1
        self.y_raw = 0.5


class _Snap:
    def __init__(self, split: str) -> None:
        self.nodes = [_Node(i) for i in range(N)]
        self.x_price = np.random.RandomState(0).randn(N, SEQ, 5).astype(np.float32)
        self.x_news = np.random.RandomState(1).randn(N, SEQ, NEWS_DIM).astype(np.float32)
        self.news_mask = np.ones((N, SEQ), dtype=np.float32)
        self.adjacency = np.ones((N, N), dtype=np.float32)
        self.presence_mask = np.ones(N, dtype=np.int8)
        self.split = split
        self.target_date = "2020-01-01"


class _Scaler:
    mean = np.asarray([0.0])
    std = np.asarray([1e-4])


class _Store:
    def get(self, ticker_id):  # noqa: ARG002
        return SimpleNamespace(target_scaler=_Scaler())


def _fake_basis(stamp):  # noqa: ARG001
    graph = SimpleNamespace(
        ticker_to_id={f"T{i}": i for i in range(N)},
        snapshots=[_Snap("train"), _Snap("train"), _Snap("val"), _Snap("test"), _Snap("test")],
    )
    pooled = SimpleNamespace(samples={"train": [1, 2], "val": [3]})
    return pooled, graph, _Store(), (), 0


def _fake_run_e0(pooled, allowed, store, out):  # noqa: ARG001
    return {"validation_metrics": {"qlike": 0.3}, "test_metrics": {"qlike": 0.3, "rmse": 0.1}}


@pytest.mark.smoke
def test_run_horizon_retrains_trainval(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "ROOT", tmp_path)
    monkeypatch.setattr(rr.combo_ladder, "build_basis", _fake_basis)
    monkeypatch.setattr(rr, "run_e0", _fake_run_e0)

    result = rr.run_horizon(5, seed=0, epochs=1, ts="T", device=torch.device("cpu"))

    assert set(result["rungs"]) == {"HAR", "FULL", "minus_graph", "minus_gate",
                                    "minus_news", "lstm_only"}
    assert all(result["rungs"][r]["trained_on"] == "train+val" for r in result["rungs"])
    out_base = tmp_path / "results" / "trackA_retrain_h5_seed0_T"
    assert (out_base / "retrain_metrics.json").exists()
    assert (out_base / "FULL" / "predictions_test.json").exists()
    assert (out_base / "minus_graph" / "predictions_test.json").exists()


def test_train_uses_trainval_partition(tmp_path, monkeypatch):
    """The retrain set is split in {train,val}; test is held out (partition-logic unit check)."""

    captured = {}

    def _spy_train(basis, train_snaps, ckpt, epochs, seed, device, **flags):  # noqa: ARG001
        captured.setdefault("splits", [s["split"] for s in train_snaps])
        return SimpleNamespace(eval=lambda: None)

    def _spy_eval(model, snaps, store, device, apply_graph, out, dump):  # noqa: ARG001
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "predictions_test.json").write_text("[]", encoding="utf-8")
        captured["test_splits"] = [s["split"] for s in snaps]
        return {"metrics": {"qlike": 0.2}, "floor_hit_fraction": 0.0}

    monkeypatch.setattr(rr, "ROOT", tmp_path)
    monkeypatch.setattr(rr.combo_ladder, "build_basis", _fake_basis)
    monkeypatch.setattr(rr, "run_e0", _fake_run_e0)
    monkeypatch.setattr(rr, "_train", _spy_train)
    monkeypatch.setattr(rr, "_evaluate_rung", _spy_eval)

    rr.run_horizon(5, seed=0, epochs=1, ts="U", device=torch.device("cpu"))

    assert set(captured["splits"]) == {"train", "val"}
    assert set(captured["test_splits"]) == {"test"}
