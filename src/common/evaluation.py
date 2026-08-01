"""
Common Evaluation Metrics for Volatility Prediction

This module contains shared evaluation metrics used across
different baseline models (HAR-R, LSTM, etc.).

Author: Stock Volatility Prediction Team
Date: 2026-06-17
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict


def qlike_loss(y_true, y_pred, epsilon=1e-8):
    """
    QLIKE Loss - Primary metric for volatility forecasting.

    Formula: L = (1/n) * Σ(y_true/y_pred - log(y_true/y_pred) - 1)

    Args:
        y_true: Actual volatility values
        y_pred: Predicted volatility values
        epsilon: Small value to prevent division by zero

    Returns:
        float: QLIKE loss (lower is better)
    """
    y_pred = np.maximum(y_pred, epsilon)
    y_true = np.maximum(y_true, epsilon)

    ratio = y_true / y_pred
    qlike = ratio - np.log(ratio) - 1
    return np.mean(qlike)


def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy.

    Returns percentage of times the model correctly predicts
    the direction of volatility change.

    Args:
        y_true: Actual volatility values
        y_pred: Predicted volatility values

    Returns:
        float: Directional accuracy (0-100)
    """
    # DEBUG: Check input statistics
    print("\n[DEBUG directional_accuracy]")
    print(f"  y_true shape: {y_true.shape}")
    print(f"  y_pred shape: {y_pred.shape}")
    print(f"  y_true range: [{y_true.min():.6f}, {y_true.max():.6f}]")
    print(f"  y_pred range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
    print(f"  y_true mean: {y_true.mean():.6f}, std: {y_true.std():.6f}")
    print(f"  y_pred mean: {y_pred.mean():.6f}, std: {y_pred.std():.6f}")
    print(f"  Unique y_true values: {len(np.unique(y_true))}")
    print(f"  Unique y_pred values: {len(np.unique(y_pred))}")

    # Check if all predictions are identical (CRITICAL BUG)
    pred_variance = np.var(y_pred)
    print(f"  Prediction variance: {pred_variance:.10f}")
    if pred_variance < 1e-10:
        print(f"  [X] ERROR: All predictions are identical! variance = {pred_variance}")
        print("  [X] This will cause Dir Acc to be 0% or undefined!")

    # Calculate actual changes
    actual_changes = np.sign(np.diff(y_true))
    pred_changes = np.sign(np.diff(y_pred))

    print(f"  Actual changes (first 10): {actual_changes[:10]}")
    print(f"  Pred changes (first 10): {pred_changes[:10]}")
    print(f"  Change agreement: {np.sum(actual_changes == pred_changes)}/{len(actual_changes)}")

    # Calculate accuracy
    accuracy = np.mean(actual_changes == pred_changes)

    print(f"  Calculated Dir Acc: {accuracy * 100:.2f}%")

    return accuracy * 100


def directional_accuracy_grouped(y_true_groups, y_pred_groups):
    """
    Directional accuracy computed WITHIN each group separately, then
    micro-averaged — avoids a spurious "change" at the boundary between two
    unrelated groups (e.g. two different tickers' pooled volatility values)
    that a naive np.diff() over the concatenated array would include.

    Args:
        y_true_groups: list of 1-D arrays, one per group (e.g. per ticker),
            each already in its own chronological order.
        y_pred_groups: list of 1-D arrays, same shapes as y_true_groups.

    Returns:
        float: directional accuracy (0-100), weighted by each group's number
        of within-group diffs (groups with < 2 samples contribute 0 diffs).
    """
    total_correct = 0
    total_count = 0
    for true_g, pred_g in zip(y_true_groups, y_pred_groups):
        true_g = np.asarray(true_g)
        pred_g = np.asarray(pred_g)
        if len(true_g) < 2:
            continue
        actual_changes = np.sign(np.diff(true_g))
        pred_changes = np.sign(np.diff(pred_g))
        total_correct += int(np.sum(actual_changes == pred_changes))
        total_count += len(actual_changes)

    if total_count == 0:
        return float("nan")
    return (total_correct / total_count) * 100


def evaluate_predictions_grouped(y_true_groups, y_pred_groups) -> Dict[str, float]:
    """
    Like evaluate_predictions(), but for predictions pooled from multiple
    independent groups (e.g. per-ticker test sets). MSE/RMSE/MAE/R²/QLIKE are
    order-independent, so they are computed on the pooled (concatenated)
    arrays, same as evaluate_predictions(). directional_accuracy is NOT
    order-independent — it is computed via directional_accuracy_grouped()
    instead, to avoid spurious cross-group boundary diffs.

    Args:
        y_true_groups: list of 1-D arrays, one per group.
        y_pred_groups: list of 1-D arrays, same shapes as y_true_groups.

    Returns:
        Dict[str, float]: same keys as evaluate_predictions().
    """
    y_true_pooled = np.concatenate([np.asarray(g) for g in y_true_groups])
    y_pred_pooled = np.concatenate([np.asarray(g) for g in y_pred_groups])

    metrics = evaluate_predictions(y_true_pooled, y_pred_pooled)
    metrics["directional_accuracy"] = directional_accuracy_grouped(y_true_groups, y_pred_groups)
    return metrics


def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.

    Args:
        y_true: Actual volatility values
        y_pred: Predicted volatility values

    Returns:
        Dict[str, float]: Dictionary of metric names and values
    """
    # Calculate R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2,
        'qlike': qlike_loss(y_true, y_pred),
        'directional_accuracy': directional_accuracy(y_true, y_pred)
    }

    return metrics
