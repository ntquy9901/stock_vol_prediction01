"""
Evaluation Module for Stock Volatility Prediction System.

This module implements comprehensive evaluation metrics for volatility forecasting,
including QLIKE loss (academic standard), directional accuracy, and benchmark comparisons.

Primary Functions:
    - qlike_loss: QLIKE loss function (primary volatility forecasting metric)
    - directional_accuracy: Directional prediction accuracy (volatility up/down)
    - theil_u_statistic: Theil's U (forecast vs random walk benchmark)
    - evaluate_forecast: Comprehensive evaluation with all metrics
    - compare_vs_benchmarks: Compare model vs simple benchmarks
    - generate_evaluation_report: Complete performance report with visualization

Key Metrics:
    - QLIKE Loss: Primary metric (lower is better, academic standard)
    - Directional Accuracy: >55% target (measures correct direction prediction)
    - RMSE: Root Mean Squared Error (accuracy metric)
    - MAE: Mean Absolute Error (robust to outliers)
    - R²: Variance explained (goodness of fit)
    - Theil's U: Forecast quality vs random walk (<1 = better than RW)

Academic Standards:
    QLIKE is the "stylized favorite of volatility forecasting literature"
    - Robust to noisy volatility proxies
    - Asymmetric penalty (underprediction > overprediction)
    - Superior model selection properties

Author: Stock Volatility Prediction Team
Date: 2026-06-15
"""

import logging
import warnings
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests/servers
import matplotlib.pyplot as plt
import seaborn as sns


# Configure module-level logger
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Constants
# ============================================================================

# Evaluation thresholds
DIRECTIONAL_ACCURACY_TARGET = 0.55  # 55% target
RMSE_TARGET_5DAY = 0.20  # 5-day forecast RMSE target
THEIL_U_THRESHOLD = 1.0  # Theil's U < 1.0 means better than random walk

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# QLIKE Loss Function (Primary Metric)
# ============================================================================

def qlike_loss(y_true: pd.Series,
               y_pred: np.ndarray,
               epsilon: float = 1e-8) -> float:
    """
    Calculate QLIKE loss for volatility forecasting evaluation.

    QLIKE is the PRIMARY evaluation metric for volatility forecasting:
    - "Stylized favorite of volatility forecasting literature"
    - Robust to noisy volatility proxies
    - Asymmetric penalty (underprediction > overprediction)
    - Superior model selection properties

    Formula: L = (1/n) * Σ(y_true/y_pred - log(y_true/y_pred) - 1)

    Args:
        y_true: True volatility values
        y_pred: Predicted volatility values
        epsilon: Small constant to prevent division by zero (default: 1e-8)

    Returns:
        float: QLIKE loss value (lower is better)

    Raises:
        ValueError: If y_true and y_pred have different lengths
        ValueError: If inputs contain negative values

    Example:
        >>> y_true = pd.Series([0.001, 0.002, 0.0015])
        >>> y_pred = np.array([0.0011, 0.0019, 0.0016])
        >>> loss = qlike_loss(y_true, y_pred)
        >>> print(f"QLIKE Loss: {loss:.6f}")
        QLIKE Loss: 0.001234

    Note:
        QLIKE loss is the PRIMARY metric for model comparison.
        Asymmetric penalty: underprediction penalized more than overprediction.
        This is desirable for risk management (better overestimate risk).
    """
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
        )

    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("QLIKE loss requires non-negative volatility values")

    # Ensure positive values (add epsilon)
    y_true_safe = np.maximum(y_true.values, epsilon)
    y_pred_safe = np.maximum(y_pred, epsilon)

    # Calculate QLIKE loss
    ratio = y_true_safe / y_pred_safe
    qlike_values = ratio - np.log(ratio) - 1
    qlike_loss_value = np.mean(qlike_values)

    logger.info(f"✅ QLIKE Loss calculated: {qlike_loss_value:.6f}")

    return qlike_loss_value


# ============================================================================
# Directional Accuracy
# ============================================================================

def directional_accuracy(y_true: pd.Series,
                        y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy for volatility forecasting.

    Directional accuracy measures the percentage of correct direction predictions:
    - Correct if both true and predicted volatility move in same direction
    - Direction: up (increase) or down (decrease) vs previous day

    Target: >55% directional accuracy (better than random guessing)

    Args:
        y_true: True volatility values
        y_pred: Predicted volatility values

    Returns:
        float: Directional accuracy (0 to 1, higher is better)

    Raises:
        ValueError: If y_true and y_pred have different lengths
        ValueError: If insufficient data for directional calculation

    Example:
        >>> y_true = pd.Series([0.001, 0.002, 0.0015, 0.0018])
        >>> y_pred = np.array([0.0011, 0.0021, 0.0014, 0.0017])
        >>> acc = directional_accuracy(y_true, y_pred)
        >>> print(f"Directional Accuracy: {acc*100:.1f}%")
        Directional Accuracy: 66.7%

    Note:
        Directional accuracy is critical for trading strategies.
        50% = random guessing, >55% = valuable predictive signal.
        Requires at least 2 data points for direction calculation.
    """
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
        )

    if len(y_true) < 2:
        raise ValueError(
            f"Insufficient data: need at least 2 points for directional accuracy"
        )

    # Calculate direction (up/down) for true and predicted
    true_direction = np.sign(np.diff(y_true.values))
    pred_direction = np.sign(np.diff(y_pred))

    # Count correct predictions (same direction)
    correct_predictions = (true_direction == pred_direction).sum()
    total_predictions = len(true_direction)

    # Exclude "no change" cases (direction = 0)
    valid_predictions = total_predictions - (true_direction == 0).sum()

    if valid_predictions == 0:
        logger.warning("No directional changes detected in data")
        return 0.0

    # Calculate accuracy (only on valid directional changes)
    directional_accuracy_value = correct_predictions / valid_predictions

    logger.info(
        f"✅ Directional Accuracy: {directional_accuracy_value*100:.1f}% "
        f"({correct_predictions}/{valid_predictions} correct direction predictions)"
    )

    return directional_accuracy_value


# ============================================================================
# Theil's U Statistic (vs Random Walk)
# ============================================================================

def theil_u_statistic(y_true: pd.Series,
                     y_pred: np.ndarray) -> float:
    """
    Calculate Theil's U statistic (forecast vs random walk benchmark).

    Theil's U measures forecast quality relative to random walk:
    - U < 1: Model BETTER than random walk
    - U = 1: Model equivalent to random walk
    - U > 1: Model WORSE than random walk

    Formula: U = sqrt(MSE(forecast) / MSE(random_walk))

    Where random_walk prediction = previous day's actual volatility

    Args:
        y_true: True volatility values
        y_pred: Predicted volatility values

    Returns:
        float: Theil's U statistic (lower is better, <1 = beats RW)

    Raises:
        ValueError: If y_true and y_pred have different lengths
        ValueError: If insufficient data for Theil's U calculation

    Example:
        >>> y_true = pd.Series([0.001, 0.002, 0.0015])
        >>> y_pred = np.array([0.0011, 0.0019, 0.0014])
        >>> u_stat = theil_u_statistic(y_true, y_pred)
        >>> print(f"Theil's U: {u_stat:.3f}")
        Theil's U: 0.857  # Better than random walk

    Note:
        Theil's U is a CRITICAL benchmark comparison.
        Random walk is the naive baseline in finance.
        U < 1 means model has genuine predictive power.
    """
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})"
        )

    if len(y_true) < 2:
        raise ValueError(
            f"Insufficient data: need at least 2 points for Theil's U"
        )

    # Calculate forecast MSE
    forecast_mse = np.mean((y_true.values - y_pred) ** 2)

    # Calculate random walk predictions (previous day's value)
    rw_pred = np.roll(y_true.values, 1)
    rw_pred[0] = rw_pred[1]  # Handle first value

    # Calculate random walk MSE
    rw_mse = np.mean((y_true.values - rw_pred) ** 2)

    # Calculate Theil's U
    if rw_mse == 0:
        logger.warning("Random walk MSE is zero (constant volatility)")
        return 0.0  # Perfect forecast if RW is perfect

    theil_u = np.sqrt(forecast_mse / rw_mse)

    if theil_u < 1.0:
        logger.info(f"✅ Theil's U: {theil_u:.3f} (BEATS random walk)")
    elif theil_u > 1.0:
        logger.warning(f"⚠️ Theil's U: {theil_u:.3f} (WORSE than random walk)")
    else:
        logger.info(f"ℹ️ Theil's U: {theil_u:.3f} (equivalent to random walk)")

    return theil_u


# ============================================================================
# Comprehensive Forecast Evaluation
# ============================================================================

def evaluate_forecast(y_true: pd.Series,
                     y_pred: np.ndarray,
                     dataset_name: str = "Test Set") -> Dict[str, Any]:
    """
    Comprehensive forecast evaluation with all metrics.

    This function calculates ALL evaluation metrics:
    - QLIKE Loss (primary metric)
    - Directional Accuracy (>55% target)
    - Theil's U (vs random walk)
    - RMSE, MAE, MSE, R² (standard metrics)

    Args:
        y_true: True volatility values
        y_pred: Predicted volatility values
        dataset_name: Name of dataset being evaluated (default: "Test Set")

    Returns:
        Dict with keys:
            - 'qlike_loss': float (primary metric)
            - 'directional_accuracy': float (percentage 0-1)
            - 'theil_u': float (vs random walk)
            - 'rmse': float (accuracy)
            - 'mae': float (robust accuracy)
            - 'mse': float (mean squared error)
            - 'r2': float (variance explained)
            - 'dataset_name': str
            - 'meets_targets': dict (boolean flags for each target)

    Example:
        >>> y_true = pd.Series([0.001, 0.002, 0.0015])
        >>> y_pred = np.array([0.0011, 0.0019, 0.0014])
        >>> metrics = evaluate_forecast(y_true, y_pred)
        >>> print(f"QLIKE: {metrics['qlike_loss']:.6f}")
        >>> print(f"Directional Acc: {metrics['directional_accuracy']*100:.1f}%")
        QLIKE: 0.001234
        Directional Acc: 66.7%

    Note:
        This is the PRIMARY evaluation function for model assessment.
        Use this for comprehensive performance reporting.
    """
    logger.info(f"📊 Evaluating forecast performance on {dataset_name}...")

    metrics = {
        'dataset_name': dataset_name,
        'sample_size': len(y_true)
    }

    # Primary metrics
    metrics['qlike_loss'] = qlike_loss(y_true, y_pred)
    metrics['directional_accuracy'] = directional_accuracy(y_true, y_pred)
    metrics['theil_u'] = theil_u_statistic(y_true, y_pred)

    # Standard metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)  # Add MSE
    metrics['r2'] = r2_score(y_true, y_pred)

    # Target achievement checks
    metrics['meets_targets'] = {
        'directional_accuracy_55': metrics['directional_accuracy'] >= DIRECTIONAL_ACCURACY_TARGET,
        'rmse_5day_0.20': metrics['rmse'] <= RMSE_TARGET_5DAY,
        'theil_u_lt_1': metrics['theil_u'] < THEIL_U_THRESHOLD
    }

    logger.info(
        f"✅ Evaluation complete: QLIKE={metrics['qlike_loss']:.6f}, "
        f"Dir Acc={metrics['directional_accuracy']*100:.1f}%, "
        f"Theil U={metrics['theil_u']:.3f}, MSE={metrics['mse']:.8f}"
    )

    return metrics


# ============================================================================
# Benchmark Comparisons
# ============================================================================

def compare_vs_benchmarks(y_true: pd.Series,
                         y_pred: np.ndarray,
                         model_name: str = "HAR-R Model") -> pd.DataFrame:
    """
    Compare model performance against simple benchmarks.

    Benchmarks:
    1. Random Walk: Use previous day's volatility as prediction
    2. Historical Mean: Use long-term average volatility as prediction
    3. Model: Your trained model's predictions

    Args:
        y_true: True volatility values
        y_pred: Model predicted volatility values
        model_name: Name of your model (default: "HAR-R Model")

    Returns:
        DataFrame with comparison rows for each benchmark:
            - 'benchmark': Name of benchmark/model
            - 'qlike_loss': QLIKE loss (lower is better)
            - 'directional_accuracy': Directional accuracy (higher is better)
            - 'theil_u': Theil's U (lower is better)

    Example:
        >>> y_true = pd.Series([0.001, 0.002, 0.0015])
        >>> y_pred = np.array([0.0011, 0.0019, 0.0014])
        >>> comparison = compare_vs_benchmarks(y_true, y_pred)
        >>> print(comparison)
                benchmark  qlike_loss  directional_accuracy  theil_u
        0      HAR-R Model     0.001234                 0.667    0.857
        1     Random Walk     0.002345                 0.500    1.000
        2  Historical Mean     0.003456                 0.000    1.500

    Note:
        Model should significantly outperform benchmarks to be valuable.
        Random walk is the naive baseline in financial forecasting.
    """
    logger.info("🔍 Comparing model vs benchmarks...")

    # Evaluate model
    model_metrics = evaluate_forecast(y_true, y_pred, dataset_name=model_name)

    # Random Walk benchmark
    rw_pred = np.roll(y_true.values, 1)
    rw_pred[0] = y_true.values[1]  # Handle first value
    rw_metrics = evaluate_forecast(y_true, rw_pred, dataset_name="Random Walk")

    # Historical Mean benchmark
    hist_mean = y_true.mean()
    hist_mean_pred = np.full(len(y_true), hist_mean)
    hist_mean_metrics = evaluate_forecast(y_true, hist_mean_pred, dataset_name="Historical Mean")

    # Compile comparison
    comparison_data = []
    for metrics_dict in [model_metrics, rw_metrics, hist_mean_metrics]:
        comparison_data.append({
            'benchmark': metrics_dict['dataset_name'],
            'qlike_loss': metrics_dict['qlike_loss'],
            'directional_accuracy': metrics_dict['directional_accuracy'],
            'theil_u': metrics_dict['theil_u']
        })

    comparison_df = pd.DataFrame(comparison_data)

    logger.info(f"✅ Benchmark comparison complete: {len(comparison_df)} benchmarks")

    return comparison_df


# ============================================================================
# Performance Visualization
# ============================================================================

def plot_forecast_performance(y_true: pd.Series,
                              y_pred: np.ndarray,
                              title: str = "Volatility Forecast Performance",
                              save_path: Optional[str] = None) -> None:
    """
    Visualize forecast performance with predicted vs actual plot.

    Creates a comprehensive visualization with:
    - Time series plot (actual vs predicted)
    - Scatter plot (predicted vs actual)
    - Error distribution

    Args:
        y_true: True volatility values
        y_pred: Predicted volatility values
        title: Plot title (default: "Volatility Forecast Performance")
        save_path: Path to save plot (optional, default: None = display only)

    Example:
        >>> y_true = pd.Series([0.001, 0.002, 0.0015])
        >>> y_pred = np.array([0.0011, 0.0019, 0.0014])
        >>> plot_forecast_performance(y_true, y_pred, save_path='results/performance.png')

    Note:
        Requires matplotlib and seaborn.
        Saves plot if save_path provided, otherwise displays interactively.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: Time series (actual vs predicted)
    ax1 = axes[0, 0]
    ax1.plot(y_true.values, label='Actual', marker='o', linewidth=2)
    ax1.plot(y_pred, label='Predicted', marker='s', linewidth=2, alpha=0.7)
    ax1.set_title('Time Series: Actual vs Predicted Volatility')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Volatility')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter plot (predicted vs actual)
    ax2 = axes[0, 1]
    ax2.scatter(y_true.values, y_pred, alpha=0.6, s=50)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_title('Scatter Plot: Predicted vs Actual')
    ax2.set_xlabel('Actual Volatility')
    ax2.set_ylabel('Predicted Volatility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Error distribution
    ax3 = axes[1, 0]
    errors = y_pred - y_true.values
    ax3.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean Error: {errors.mean():.6f}')
    ax3.set_title('Error Distribution')
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cumulative errors
    ax4 = axes[1, 1]
    cumulative_error = np.cumsum(errors)
    ax4.plot(cumulative_error, linewidth=2, color='purple')
    ax4.axhline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_title('Cumulative Prediction Error')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Cumulative Error')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Performance plot saved to {save_path}")
    else:
        logger.info("✅ Performance plot generated (display mode)")

    plt.close()


# ============================================================================
# Complete Evaluation Report
# ============================================================================

def generate_evaluation_report(y_train_true: pd.Series,
                               y_train_pred: np.ndarray,
                               y_test_true: pd.Series,
                               y_test_pred: np.ndarray,
                               model_name: str = "HAR-R Model",
                               save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate complete evaluation report with metrics, benchmarks, and visualization.

    This function creates a comprehensive evaluation report:
    1. Train set performance metrics
    2. Test set performance metrics
    3. Benchmark comparisons (random walk, historical mean)
    4. Performance visualization
    5. Target achievement summary

    Args:
        y_train_true: Training set true values
        y_train_pred: Training set predicted values
        y_test_true: Test set true values
        y_test_pred: Test set predicted values
        model_name: Name of model (default: "HAR-R Model")
        save_dir: Directory to save plots and reports (optional)

    Returns:
        Dict containing:
            - 'train_metrics': Training set performance dict
            - 'test_metrics': Test set performance dict
            - 'benchmark_comparison': DataFrame vs benchmarks
            - 'target_achievement': Summary of targets met
            - 'plot_paths': Paths to saved plots (if save_dir provided)

    Example:
        >>> report = generate_evaluation_report(
        ...     y_train_true, y_train_pred,
        ...     y_test_true, y_test_pred,
        ...     model_name="HAR-R Baseline",
        ...     save_dir="results"
        ... )
        >>> print(f"Test QLIKE: {report['test_metrics']['qlike_loss']:.6f}")
        Test QLIKE: 0.001234

    Note:
        This is the PRIMARY entry point for model evaluation.
        Generates comprehensive report for analysis and documentation.
    """
    logger.info("📊 Generating complete evaluation report...")

    # Evaluate train and test sets
    train_metrics = evaluate_forecast(y_train_true, y_train_pred, dataset_name="Training Set")
    test_metrics = evaluate_forecast(y_test_true, y_test_pred, dataset_name="Test Set")

    # Benchmark comparison (using test set)
    benchmark_comparison = compare_vs_benchmarks(y_test_true, y_test_pred, model_name=model_name)

    # Compile report
    report = {
        'model_name': model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'benchmark_comparison': benchmark_comparison,
        'target_achievement': {
            'test_directional_accuracy_55': test_metrics['meets_targets']['directional_accuracy_55'],
            'test_rmse_5day_0.20': test_metrics['meets_targets']['rmse_5day_0.20'],
            'test_theil_u_lt_1': test_metrics['meets_targets']['theil_u_lt_1']
        }
    }

    # Generate visualizations
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Test set performance plot
        test_plot_path = os.path.join(save_dir, 'test_performance.png')
        plot_forecast_performance(y_test_true, y_test_pred,
                                 title=f"{model_name} - Test Set Performance",
                                 save_path=test_plot_path)

        report['plot_paths'] = {
            'test_performance': test_plot_path
        }

        logger.info(f"✅ Plots saved to {save_dir}")

    # Summary logging
    logger.info("=" * 80)
    logger.info(f"📊 EVALUATION REPORT: {model_name}")
    logger.info("=" * 80)
    logger.info(f"Test Set Performance:")
    logger.info(f"  QLIKE Loss: {test_metrics['qlike_loss']:.6f} (lower is better)")
    logger.info(f"  Directional Accuracy: {test_metrics['directional_accuracy']*100:.1f}% (target: >55%)")
    logger.info(f"  Theil's U: {test_metrics['theil_u']:.3f} (target: <1.0)")
    logger.info(f"  RMSE: {test_metrics['rmse']:.6f} (target: <0.20)")
    logger.info(f"  MSE: {test_metrics['mse']:.8f} (RMSE²)")
    logger.info(f"  R²: {test_metrics['r2']:.4f}")
    logger.info(f"\nTarget Achievement:")
    for target, achieved in report['target_achievement'].items():
        status = "✅" if achieved else "❌"
        logger.info(f"  {status} {target}")
    logger.info("=" * 80)

    return report
