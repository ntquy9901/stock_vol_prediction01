"""Tests for the differentiable snapshot QLIKE loss (C1 shared lever)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_CODE = Path(__file__).resolve().parents[1] / "code"
_PILOT = Path(__file__).resolve().parents[2] / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_CODE), str(_PILOT), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qlike_torch import snapshot_qlike_loss  # noqa: E402
from src.common.evaluation import qlike_loss  # noqa: E402


def _denorm(pred_norm, mean, std):
    return pred_norm * std + mean


def test_matches_numpy_qlike_on_uniform_present():
    """With equal present-node counts per snapshot and floors inactive, the torch loss equals the
    numpy eval QLIKE over the flattened present values."""

    rng = np.random.default_rng(0)
    batch, nodes = 4, 5
    mean = torch.full((batch, nodes), 1.3e-4, dtype=torch.float64)
    std = torch.full((batch, nodes), 5e-5, dtype=torch.float64)
    # positive raw values well above the 1e-6 floor
    pred_raw = torch.tensor(rng.uniform(5e-5, 3e-4, size=(batch, nodes)))
    target_raw = torch.tensor(rng.uniform(5e-5, 3e-4, size=(batch, nodes)))
    pred_norm = (pred_raw - mean) / std
    target_norm = (target_raw - mean) / std
    presence = torch.ones(batch, nodes, dtype=torch.bool)

    got = snapshot_qlike_loss(pred_norm, target_norm, mean, std, presence).item()
    # numpy per-snapshot mean then mean over snapshots (equal counts -> equals flat mean)
    per_snap = [qlike_loss(target_raw[i].numpy(), pred_raw[i].numpy()) for i in range(batch)]
    expected = float(np.mean(per_snap))
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_zero_at_perfect_prediction():
    mean = torch.full((2, 3), 1e-4, dtype=torch.float64)
    std = torch.full((2, 3), 3e-5, dtype=torch.float64)
    target_norm = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.5, -0.1]], dtype=torch.float64)
    presence = torch.ones(2, 3, dtype=torch.bool)
    loss = snapshot_qlike_loss(target_norm.clone(), target_norm, mean, std, presence).item()
    assert loss == pytest.approx(0.0, abs=1e-10)


def test_gradient_flows_to_predictions():
    mean = torch.full((2, 3), 1e-4, dtype=torch.float64)
    std = torch.full((2, 3), 3e-5, dtype=torch.float64)
    pred_norm = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    target_norm = torch.tensor([[0.2, -0.1, 0.4], [0.1, 0.0, -0.2]], dtype=torch.float64)
    presence = torch.ones(2, 3, dtype=torch.bool)
    loss = snapshot_qlike_loss(pred_norm, target_norm, mean, std, presence)
    loss.backward()
    assert pred_norm.grad is not None
    assert torch.isfinite(pred_norm.grad).all()
    assert pred_norm.grad.abs().sum() > 0


def test_absent_nodes_excluded():
    """A present node's loss is unaffected by arbitrary values on masked-absent nodes."""

    mean = torch.full((1, 3), 1e-4, dtype=torch.float64)
    std = torch.full((1, 3), 3e-5, dtype=torch.float64)
    pred_norm = torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64)
    target_norm = torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float64)
    presence = torch.tensor([[True, False, False]])
    base = snapshot_qlike_loss(pred_norm, target_norm, mean, std, presence).item()
    poisoned = pred_norm.clone()
    poisoned[0, 1] = 9.0  # absurd value on an absent node
    poisoned[0, 2] = -9.0
    got = snapshot_qlike_loss(poisoned, target_norm, mean, std, presence).item()
    assert got == pytest.approx(base, abs=1e-12)


def test_floor_keeps_loss_finite_for_nonpositive_denorm_pred():
    """A prediction that denormalizes below zero is floored, so the loss stays finite."""

    mean = torch.full((1, 2), 1e-4, dtype=torch.float64)
    std = torch.full((1, 2), 3e-5, dtype=torch.float64)
    pred_norm = torch.tensor([[-100.0, -100.0]], dtype=torch.float64)  # denorm well below 0
    target_norm = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    presence = torch.ones(1, 2, dtype=torch.bool)
    loss = snapshot_qlike_loss(pred_norm, target_norm, mean, std, presence)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_rejects_shape_mismatch():
    mean = torch.ones(2, 3, dtype=torch.float64)
    std = torch.ones(2, 3, dtype=torch.float64)
    presence = torch.ones(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError):
        snapshot_qlike_loss(torch.ones(2, 3), torch.ones(2, 4), mean, std, presence)
