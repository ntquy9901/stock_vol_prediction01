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


def assert_finite_metrics(metrics: Dict[str, float]) -> None:
    """
    Raise if any metric value is NaN or Infinity.

    Rationale (AUD-016): QLIKE (and, for extreme-magnitude inputs, MSE/RMSE/R^2)
    can become non-finite even past the epsilon guard in qlike_loss(). Python's
    json.dump()/json.dumps() default to allow_nan=True, so a non-finite metric
    would otherwise be silently written into a results.json as invalid JSON
    (NaN/Infinity tokens) instead of surfacing the underlying bug. This check
    fails loudly at the source instead of silently clipping/replacing values.

    Args:
        metrics: Dict of metric name -> value, as returned by evaluate_predictions().

    Raises:
        ValueError: naming the first non-finite metric key found.
    """
    for key, value in metrics.items():
        if not np.isfinite(value):
            raise ValueError(
                f"Metric '{key}' is non-finite ({value!r}). Refusing to return "
                "metrics that would silently produce NaN/Infinity in a results.json "
                "(json.dump's default allow_nan=True does not reject this)."
            )


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
    Calculate directional accuracy on a single chronological sequence.

    Returns percentage of times the model correctly predicts
    the direction of volatility change.

    CAUTION: y_true/y_pred must already be a single ticker's values ordered by
    time. Passing a flattened multi-ticker array (e.g. day-then-all-stocks
    order) silently mixes different tickers' same-day values into np.diff and
    inflates the result — use directional_accuracy_per_ticker() for that case
    (see docs/report_2026-08-01/DIRACC_ISSUE_NOTE.md).

    Args:
        y_true: Actual volatility values
        y_pred: Predicted volatility values

    Returns:
        float: Directional accuracy (0-100)
    """
    actual_changes = np.sign(np.diff(y_true))
    pred_changes = np.sign(np.diff(y_pred))
    accuracy = np.mean(actual_changes == pred_changes)
    return accuracy * 100


def directional_accuracy_per_ticker(y_true, y_pred, n_stocks):
    """
    Calculate directional accuracy correctly for multi-ticker flattened arrays.

    Args:
        y_true, y_pred: flattened arrays in (num_windows, n_stocks) day-major
            order, i.e. index i belongs to ticker (i % n_stocks) on window (i // n_stocks) —
            the layout produced by every multi-stock DataLoader in this project.
        n_stocks: number of tickers interleaved in the flattened array.

    Returns:
        float or None: mean per-ticker directional accuracy (0-100), or None
        if there are fewer than 2 windows per ticker to diff.
    """
    if len(y_true) < n_stocks * 2 or len(y_true) % n_stocks != 0:
        return None
    num_windows = len(y_true) // n_stocks
    t2 = np.asarray(y_true).reshape(num_windows, n_stocks)
    p2 = np.asarray(y_pred).reshape(num_windows, n_stocks)
    per_ticker = [
        np.mean(np.sign(np.diff(t2[:, s])) == np.sign(np.diff(p2[:, s]))) * 100
        for s in range(n_stocks)
    ]
    return float(np.mean(per_ticker))


def evaluate_predictions(y_true, y_pred, n_stocks=None) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.

    Args:
        y_true: Actual volatility values
        y_pred: Predicted volatility values
        n_stocks: if given, y_true/y_pred are treated as a flattened
            multi-ticker array (day-major order, see directional_accuracy_per_ticker),
            and 'directional_accuracy' is computed per-ticker (the correct,
            headline value). The old flattened calculation is still reported
            under 'directional_accuracy_flat_biased' for audit-trail comparison.
            If omitted, behavior is unchanged (single-ticker sequence).

    Returns:
        Dict[str, float]: Dictionary of metric names and values
    """
    # Calculate R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    mse = mean_squared_error(y_true, y_pred)
    flat_dir_acc = directional_accuracy(y_true, y_pred)
    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2,
        'qlike': qlike_loss(y_true, y_pred),
        'directional_accuracy': flat_dir_acc
    }

    if n_stocks is not None:
        per_ticker = directional_accuracy_per_ticker(y_true, y_pred, n_stocks)
        metrics['directional_accuracy_flat_biased'] = flat_dir_acc
        if per_ticker is not None:
            metrics['directional_accuracy'] = per_ticker

    assert_finite_metrics(metrics)
    return metrics
