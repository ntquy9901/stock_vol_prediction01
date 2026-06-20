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
    print(f"\n[DEBUG directional_accuracy]")
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
        print(f"  [X] This will cause Dir Acc to be 0% or undefined!")

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
