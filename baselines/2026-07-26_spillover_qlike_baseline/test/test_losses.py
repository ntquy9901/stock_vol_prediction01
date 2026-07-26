"""Tests for losses.combined_loss / qlike (design.md §2.2)."""
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import numpy as np
import torch

from losses import combined_loss, qlike


def test_qlike_zero_at_perfect_prediction():
    y = torch.tensor([1.0, 2.0, 0.5])
    loss = qlike(y, y.clone())
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_qlike_matches_manual_formula():
    y_true = torch.tensor([1.0, 2.0])
    y_pred = torch.tensor([1.5, 1.0])
    ratio = y_true / y_pred
    expected = (ratio - torch.log(ratio) - 1).mean()
    assert torch.isclose(qlike(y_true, y_pred), expected, atol=1e-5)


def test_qlike_clamps_negative_and_zero_without_nan():
    y_true = torch.tensor([1.0, -0.5, 0.0])
    y_pred = torch.tensor([-0.2, 0.0, 1.0])
    loss = qlike(y_true, y_pred, eps=1e-6)
    assert torch.isfinite(loss)


def test_combined_loss_reduces_to_mse_when_weight_zero():
    pred = torch.randn(4, 3)
    y = torch.randn(4, 3)
    mean_t = torch.zeros(3)
    std_t = torch.ones(3)
    mse = torch.nn.functional.mse_loss(pred, y)
    combined = combined_loss(pred, y, mean_t, std_t, qlike_weight=0.0)
    assert torch.isclose(combined, mse, atol=1e-5)


def test_combined_loss_is_finite_and_differentiable():
    torch.manual_seed(0)
    pred = torch.randn(8, 5, requires_grad=True)
    y = torch.randn(8, 5)
    mean_t = torch.tensor([0.01, 0.02, 0.015, 0.03, 0.01])
    std_t = torch.tensor([0.005, 0.01, 0.008, 0.02, 0.005])
    loss = combined_loss(pred, y, mean_t, std_t, qlike_weight=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_combined_loss_qlike_weight_changes_value():
    torch.manual_seed(1)
    pred = torch.randn(4, 2)
    y = torch.randn(4, 2)
    mean_t = torch.tensor([0.01, 0.02])
    std_t = torch.tensor([0.005, 0.01])
    low = combined_loss(pred, y, mean_t, std_t, qlike_weight=0.0)
    high = combined_loss(pred, y, mean_t, std_t, qlike_weight=1.0)
    assert not torch.isclose(low, high)
