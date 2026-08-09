"""Smoke test for the sweep training/eval helpers on tiny synthetic snapshots (no GPU, no basis).

Exercises the real prediction, loss, and eval-record code paths (C1 monolithic and C2 HAR-residual)
against fabricated GraphSnapshot objects and a minimal PreprocessorStore, so a loop bug surfaces in
seconds instead of after the multi-minute basis build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

_CODE = Path(__file__).resolve().parents[1] / "code"
_PILOT = Path(__file__).resolve().parents[2] / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_CODE), str(_PILOT), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sweep  # noqa: E402
from data import GraphNode, GraphSnapshot  # noqa: E402
from scaling import ArrayStandardizer, PreprocessorStore, TickerPreprocessor  # noqa: E402

N = 3          # tickers/nodes
SEQ = 4
PRICE_DIM = 3
NEWS_DIM = 5
HIDDEN = 8


class _FakeModel(nn.Module):
    """Minimal GraphAblationModel surface used by _predict_batch / _loss / eval."""

    def __init__(self):
        super().__init__()
        self.price_encoder = nn.LSTM(PRICE_DIM, 2)  # only .parameters() is inspected (frozen check)
        for p in self.price_encoder.parameters():
            p.requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
        self.message_passing = None
        self._positivity_configured = True
        self._positivity_epsilon = 1e-6
        self.register_buffer("target_mean", torch.full((N,), 1.3e-4))
        self.register_buffer("target_std", torch.full((N,), 4e-5))

    def _apply_positivity(self, output, ticker_ids):
        mean = self.target_mean[ticker_ids].reshape(output.shape)
        std = self.target_std[ticker_ids].reshape(output.shape)
        raw = output * std + mean
        eps = self._positivity_epsilon
        raw_pos = eps * torch.nn.functional.softplus(raw / eps) + eps
        return (raw_pos - mean) / std

    def apply_graph_head(self, base, adjacency, ticker_ids, presence_mask=None,
                         apply_message_passing=True):
        out = self.head(base).squeeze(-1)
        return self._apply_positivity(out, ticker_ids.reshape(-1).long())


def _snapshot(split: str, date: str, y_scale: float) -> GraphSnapshot:
    rng = np.random.default_rng(abs(hash((split, date))) % 2**32)
    x_price = rng.normal(size=(N, SEQ, PRICE_DIM)).astype(np.float32)
    x_news = rng.normal(size=(N, SEQ, NEWS_DIM)).astype(np.float32)
    news_mask = np.ones((N, SEQ), dtype=np.int8)
    adjacency = np.eye(N, dtype=np.float32)  # self-loops satisfy the MP invariant
    adjacency[0, 1] = adjacency[1, 0] = 0.5
    presence = np.ones(N, dtype=np.int8)
    nodes = tuple(
        GraphNode(i, f"T{i}", split, y_raw=float(1e-4 * (1 + 0.1 * i) * y_scale), y_norm=float(0.1 * i))
        for i in range(N)
    )
    return GraphSnapshot(date, split, (), nodes, x_price, x_news, news_mask, adjacency, presence)


def _store() -> PreprocessorStore:
    pre = {}
    for i in range(N):
        scaler = ArrayStandardizer(np.array([1.3e-4]), np.array([4e-5]))
        pre[i] = TickerPreprocessor(("v",), "v", 0.0, 1.0, scaler, scaler)
    return PreprocessorStore(pre)


def _base_cache(snaps):
    return [torch.randn(N, HIDDEN) for _ in snaps]


@pytest.mark.smoke
@pytest.mark.parametrize("cfg_name", ["C1", "C2"])
def test_train_and_eval_runs_and_produces_metrics(tmp_path, monkeypatch, cfg_name):
    monkeypatch.setattr(sweep, "GRAPH_EPOCHS", 2)
    cfg = sweep.CONFIGS[cfg_name]
    train = [_snapshot("train", f"2020-01-0{d}", 1.0) for d in range(1, 6)]
    val = [_snapshot("val", f"2021-01-0{d}", 1.1) for d in range(1, 4)]
    test = [_snapshot("test", f"2022-01-0{d}", 1.2) for d in range(1, 4)]
    snapshots = tuple(train + val + test)

    class _Graph:
        ticker_to_id = {f"T{i}": i for i in range(N)}
        train_end_date = "2020-01-05"
    graph = _Graph()
    graph.snapshots = snapshots
    base_cache = {"train": _base_cache(train), "val": _base_cache(val), "test": _base_cache(test)}
    model = _FakeModel()
    monkeypatch.setattr(model, "configure_positivity", lambda store: model, raising=False)

    result = sweep.train_and_eval(cfg_name, cfg, None, graph, _store(), model, base_cache,
                                  tmp_path, seed=42, device=torch.device("cpu"))
    for split in ("validation_metrics", "test_metrics"):
        metrics = result[split]
        assert set(sweep._METRIC_KEYS).issubset(metrics)
        assert np.isfinite(list(metrics.values())).all()
    assert (tmp_path / "predictions_test.json").exists()
    assert len(result["train_losses"]) == 2


@pytest.mark.smoke
def test_qlike_loss_path_gradients_reach_head():
    """A monolithic C1 step must push gradients into the trainable head (not the frozen encoder)."""

    model = _FakeModel()
    cfg = sweep.CONFIGS["C1"]
    snaps = [_snapshot("train", "2020-02-01", 1.0)]
    presence, ticker_ids, targets = sweep._stack_targets(snaps, torch.device("cpu"))
    base = [torch.randn(N, HIDDEN)]
    adjacency = torch.from_numpy(np.stack([s.adjacency for s in snaps]))
    predictions = sweep._predict_batch(cfg, model, None, base, adjacency, ticker_ids, presence, None)
    loss = sweep._loss(cfg, model, predictions, targets, ticker_ids, presence)
    loss.backward()
    assert model.head[0].weight.grad is not None
    assert model.head[0].weight.grad.abs().sum() > 0
