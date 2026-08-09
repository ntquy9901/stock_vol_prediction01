"""Differentiable snapshot QLIKE loss on denormalized, positivity-floored predictions.

QLIKE is the headline volatility-forecast metric and is level-sensitive: it must be computed on the
DENORMALIZED prediction (physical variance scale), not on the per-ticker standardized prediction. This
module denormalizes with the same train-fit per-ticker target scaler used everywhere else
(``GraphAblationModel.target_mean/target_std``), floors the prediction to strict positivity so the loss
and its gradient stay finite, and averages per snapshot over PRESENT nodes only -- the identical
weighting to the pilot's ``_mean_snapshot_mse`` so the two losses are directly swappable.
"""

from __future__ import annotations

import torch

# Matches models.POSITIVITY_EPSILON (denormalized-scale positive floor, three orders below the
# ~1e-3 typical Parkinson variance) and evaluation.qlike_loss's target epsilon.
FLOOR_EPSILON = 1e-6
TARGET_EPSILON = 1e-8


def snapshot_qlike_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    presence: torch.Tensor,
    eps: float = TARGET_EPSILON,
    floor_eps: float = FLOOR_EPSILON,
) -> torch.Tensor:
    """Equal-weighted per-snapshot QLIKE over present nodes.

    Args:
        pred_norm, target_norm: ``[batch, nodes]`` standardized predictions / model targets.
        target_mean, target_std: ``[batch, nodes]`` per-node train-fit target scaler stats
            (``raw = norm*std + mean``).
        presence: ``[batch, nodes]`` boolean mask; only present nodes enter the loss.
        eps: floor applied to the denormalized TARGET (QLIKE undefined for non-positive truth).
        floor_eps: soft positivity floor applied to the denormalized PREDICTION.

    Returns:
        Scalar loss: mean over snapshots of (mean over each snapshot's present-node QLIKE).
    """

    if pred_norm.shape != target_norm.shape or pred_norm.ndim != 2 or not pred_norm.shape[0]:
        raise ValueError("pred_norm and target_norm must be non-empty [batch, nodes] tensors")
    if target_mean.shape != pred_norm.shape or target_std.shape != pred_norm.shape:
        raise ValueError("target_mean/target_std must match the [batch, nodes] prediction shape")
    if presence.shape != pred_norm.shape:
        raise ValueError("presence must match the [batch, nodes] prediction shape")
    if floor_eps <= 0 or eps <= 0:
        raise ValueError("eps and floor_eps must be positive")

    present = presence.to(dtype=torch.bool)
    pred_raw = pred_norm * target_std + target_mean
    target_raw = target_norm * target_std + target_mean
    # Hard positive clamps identical to evaluation.qlike_loss (an identity for the healthy bulk, so
    # the loss is a faithful QLIKE). Smoothness in the sub-floor tail is supplied UPSTREAM by
    # GraphAblationModel._apply_positivity (softplus floor before the prediction reaches this loss);
    # this clamp is only a finite-value safety net for the pathological non-positive case.
    pred_pos = torch.clamp(pred_raw, min=floor_eps)
    target_pos = torch.clamp(target_raw, min=eps)
    ratio = target_pos / pred_pos
    qlike = ratio - torch.log(ratio) - 1.0

    per_snapshot = []
    for row_qlike, row_present in zip(qlike, present, strict=True):
        if not row_present.any():
            raise ValueError("each snapshot must contain at least one present node")
        per_snapshot.append(row_qlike[row_present].mean())
    return torch.stack(per_snapshot).mean()
