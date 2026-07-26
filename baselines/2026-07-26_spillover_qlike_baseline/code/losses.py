"""QLIKE-augmented training loss (see design.md §2.2).

MSE stays the primary loss (proven stable, per CLAUDE.md's documented LSTM-GNN normalization
failure — do NOT force positivity via an output activation like Softplus, it collapsed
predictions to 0 in that incident). QLIKE is added as a SECONDARY term computed on the
inverse-transformed (denormalized, naturally-positive-in-real-data) scale, clamped away from 0 to
avoid NaN/Inf, weighted small (`qlike_weight`, default 0.1) so it nudges training toward the
academic-standard volatility loss without destabilizing it.
"""
from __future__ import annotations

import torch


def build_denorm_tensors(dataset, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-stock (mean, std) tensors, ordered like `dataset.stock_names`, for vectorized
    inverse-transform of a [batch, num_stocks] normalized tensor back to the original volatility
    scale. `dataset.target_normalizers[stock]` is a fitted `VolatilityNormalizer`
    (see src/common/data_normalization.py) — plain scalar mean/std, affine, differentiable."""
    means, stds = [], []
    for sname in dataset.stock_names:
        norm = dataset.target_normalizers[sname]
        means.append(float(norm.mean))
        stds.append(float(norm.std))
    mean_t = torch.tensor(means, dtype=torch.float32, device=device)
    std_t = torch.tensor(stds, dtype=torch.float32, device=device)
    return mean_t, std_t


def qlike(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable QLIKE, on values assumed already on the (positive) original scale.
    Clamped to `eps` to avoid division-by-zero / log(0)."""
    y_true_c = torch.clamp(y_true, min=eps)
    y_pred_c = torch.clamp(y_pred, min=eps)
    ratio = y_true_c / y_pred_c
    return (ratio - torch.log(ratio) - 1).mean()


def combined_loss(
    pred_norm: torch.Tensor,
    y_norm: torch.Tensor,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    qlike_weight: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """MSE(normalized) + qlike_weight * QLIKE(denormalized, clamped).

    Args:
        pred_norm, y_norm: [batch, num_stocks], normalized scale (model's native output scale).
        mean_t, std_t: [num_stocks] per-stock denormalization constants (see
            `build_denorm_tensors`), broadcast over the batch dim.
        qlike_weight: weight of the QLIKE term relative to MSE (default 0.1 — MSE dominates,
            QLIKE is a regularizer, not the primary objective; see design.md §2.2, not tuned).
    """
    mse = torch.nn.functional.mse_loss(pred_norm, y_norm)
    if qlike_weight == 0.0:
        return mse
    pred_denorm = pred_norm * std_t.unsqueeze(0) + mean_t.unsqueeze(0)
    y_denorm = y_norm * std_t.unsqueeze(0) + mean_t.unsqueeze(0)
    return mse + qlike_weight * qlike(y_denorm, pred_denorm, eps=eps)
