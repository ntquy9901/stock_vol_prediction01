"""
Model Training Module for Stock Volatility Prediction System.

This module implements HAR-R (Heterogeneous Autoregressive) baseline model
with linear regression for 5-day ahead volatility forecasting.

Primary Functions:
    - set_random_seeds: Set all random seeds for reproducibility
    - temporal_train_test_split: Chronological train/test split (prevents leakage)
    - train_har_r_baseline: Train HAR-R linear regression model
    - detect_overfitting: Check overfitting via train/test performance gap
    - get_model_predictions: Generate predictions from trained model
    - complete_training_pipeline: End-to-end training workflow

Model Specification (HAR-R):
    target_5d = β₀ + β₁*har_daily_vol + β₂*har_weekly_vol + β₃*har_monthly_vol

Key Design Principles:
    - Temporal integrity: NO random splitting (prevents data leakage)
    - Reproducibility: Fixed random seeds (42)
    - Quality monitoring: Overfitting detection, learning curves
    - Transparency: Feature importance, model coefficients

Author: Stock Volatility Prediction Team
Date: 2026-06-15
"""

import logging
import warnings
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


# Configure module-level logger
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Constants
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Default train/test split ratio
DEFAULT_TEST_SIZE = 0.2  # 20% test, 80% train

# Overfitting detection threshold
OVERFITTING_THRESHOLD = 1.2  # 20% degradation allowed

# Feature columns for HAR-R model
HAR_FEATURE_COLUMNS = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']

# Target column (5-day horizon)
TARGET_COLUMN = 'target_5d'


# ============================================================================
# Reproducibility Setup
# ============================================================================

def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Set all random seeds for reproducible experiments.

    This function ensures:
    - NumPy random operations are deterministic
    - Scikit-learn operations are reproducible
    - Any random initialization is controlled

    Args:
        seed: Random seed value (default: 42)

    Example:
        >>> set_random_seeds(42)
        >>> np.random.randint(0, 100, size=3)
        array([51, 92, 14])  # Will always produce same values
        >>> set_random_seeds(42)
        >>> np.random.randint(0, 100, size=3)
        array([51, 92, 14])  # Identical result

    Note:
        Must be called BEFORE any model training or random operations.
        Call once at the start of training pipeline.
    """
    np.random.seed(seed)

    # Note: sklearn's seed is set via numpy seed
    # For future PyTorch usage:
    # try:
    #     import torch
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed_all(seed)
    # except ImportError:
    #     pass

    logger.info(f"✅ Random seeds set to {seed} for reproducibility")


# ============================================================================
# Temporal Train/Test Split
# ============================================================================

def temporal_train_test_split(featureset: pd.DataFrame,
                               test_size: float = DEFAULT_TEST_SIZE,
                               target_column: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split featureset into train and test sets CHRONOLOGICALLY.

    This function preserves temporal ordering to prevent data leakage:
    - Train data: First (1-test_size) portion of data
    - Test data: Last test_size portion of data

    CRITICAL: Do NOT use random train_test_split for time series!

    Args:
        featureset: DataFrame with HAR features and target column
        test_size: Fraction of data for testing (default: 0.2 → 20% test)
        target_column: Name of target column (default: 'target_5d')

    Returns:
        Tuple of (train_data, test_data) DataFrames

    Raises:
        ValueError: If featureset is empty
        ValueError: If test_size not in (0, 1)
        ValueError: If insufficient data after NaN removal

    Example:
        >>> data = pd.DataFrame({
        ...     'har_daily_vol': [0.001, 0.002, 0.0015, 0.0018, 0.002],
        ...     'target_5d': [0.0025, np.nan, 0.002, 0.0022, np.nan]
        ... })
        >>> train, test = temporal_train_test_split(data, test_size=0.4)
        >>> print(len(train), len(test))
        2 1  # Only rows with non-NaN targets are used

    Note:
        Rows with NaN targets (trailing rows) are automatically excluded.
        Split is performed on valid (non-NaN target) rows only.
    """
    # Validate inputs
    if len(featureset) == 0:
        raise ValueError("Featureset is empty")

    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    # Remove rows with NaN targets OR NaN features (warm-up + trailing rows)
    feature_columns = [col for col in featureset.columns if col != target_column]
    valid_data = featureset.dropna(subset=[target_column] + feature_columns).copy()

    if len(valid_data) < 10:
        raise ValueError(
            f"Insufficient valid data: {len(valid_data)} samples. "
            f"Need at least 10 samples for training."
        )

    # Calculate split point (chronological)
    split_idx = int(len(valid_data) * (1 - test_size))

    # Split chronologically
    train_data = valid_data.iloc[:split_idx].copy()
    test_data = valid_data.iloc[split_idx:].copy()

    # Verify temporal order
    assert train_data.index[-1] < test_data.index[0], "Train/test split not chronological!"

    logger.info(
        f"✅ Temporal split: {len(train_data)} train samples, "
        f"{len(test_data)} test samples ({test_size*100:.0f}% test)"
    )

    return train_data, test_data


# ============================================================================
# HAR-R Baseline Model Training
# ============================================================================

def train_har_r_baseline(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         feature_columns: list = None) -> LinearRegression:
    """
    Train HAR-R (Heterogeneous Autoregressive) baseline model.

    Model Specification:
        target_5d = β₀ + β₁*har_daily_vol + β₂*har_weekly_vol + β₃*har_monthly_vol

    This is the BASELINE model for volatility forecasting:
    - Linear regression with HAR features
    - Interpretability: Coefficients show feature importance
    - Academically established: HAR-R is industry standard

    Args:
        X_train: Training features DataFrame (HAR features)
        y_train: Training target Series (5-day ahead volatility)
        feature_columns: List of feature column names (default: HAR_FEATURE_COLUMNS)

    Returns:
        LinearRegression: Trained HAR-R model

    Raises:
        ValueError: If X_train and y_train have different lengths
        ValueError: If required feature columns missing

    Example:
        >>> X_train = pd.DataFrame({
        ...     'har_daily_vol': [0.001, 0.002, 0.0015],
        ...     'har_weekly_vol': [0.0012, 0.0018, 0.0016],
        ...     'har_monthly_vol': [0.0013, 0.0017, 0.0015]
        ... })
        >>> y_train = pd.Series([0.0025, 0.0022, 0.002])
        >>> model = train_har_r_baseline(X_train, y_train)
        >>> print(f"Intercept: {model.intercept_:.6f}")
        Intercept: 0.000856

    Note:
        HAR-R is the BASELINE - compare all advanced models against this.
        Model coefficients indicate feature importance (β₃ > β₂ > β₁ typically).
    """
    if feature_columns is None:
        feature_columns = HAR_FEATURE_COLUMNS

    # Validate inputs
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train ({len(X_train)}) and y_train ({len(y_train)}) have different lengths"
        )

    # Check required features
    missing_features = [col for col in feature_columns if col not in X_train.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Extract features
    X_features = X_train[feature_columns]

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_features, y_train)

    logger.info(
        f"✅ HAR-R baseline trained: {len(feature_columns)} features, "
        f"{len(X_train)} training samples"
    )

    # Log model coefficients
    coef_str = ", ".join([f"{col}: {coef:.4f}" for col, coef in zip(feature_columns, model.coef_)])
    logger.info(f"Model coefficients: {coef_str}")
    logger.info(f"Model intercept: {model.intercept_:.6f}")

    return model


# ============================================================================
# Overfitting Detection
# ============================================================================

def detect_overfitting(model: LinearRegression,
                      X_train: pd.DataFrame,
                      X_test: pd.DataFrame,
                      y_train: pd.Series,
                      y_test: pd.Series,
                      threshold: float = OVERFITTING_THRESHOLD) -> Dict[str, Any]:
    """
    Detect overfitting by comparing train vs test performance.

    Overfitting Detection Logic:
    - Calculate MSE on training and testing sets
    - Compute overfitting_ratio = test_MSE / train_MSE
    - If ratio > threshold, model may be overfitting

    Args:
        model: Trained HAR-R model
        X_train: Training features
        X_test: Testing features
        y_train: Training targets
        y_test: Testing targets
        threshold: Overfitting threshold (default: 1.2 → 20% degradation allowed)

    Returns:
        Dict with keys:
            - 'is_overfitting': bool (True if overfitting detected)
            - 'train_mse': float (MSE on training set)
            - 'test_mse': float (MSE on testing set)
            - 'overfitting_ratio': float (test_MSE / train_MSE)

    Example:
        >>> model = train_har_r_baseline(X_train, y_train)
        >>> overfitting_results = detect_overfitting(model, X_train, X_test, y_train, y_test)
        >>> print(f"Overfitting: {overfitting_results['is_overfitting']}")
        Overfitting: False
        >>> print(f"Ratio: {overfitting_results['overfitting_ratio']:.2f}")
        Ratio: 1.05

    Note:
        Ratio > 1.2 indicates potential overfitting (more than 20% test degradation).
        Linear regression typically has low overfitting risk.
    """
    feature_columns = HAR_FEATURE_COLUMNS

    # Generate predictions
    train_pred = model.predict(X_train[feature_columns])
    test_pred = model.predict(X_test[feature_columns])

    # Calculate MSE
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    # Calculate overfitting ratio
    overfitting_ratio = test_mse / train_mse if train_mse > 0 else float('inf')

    # Detect overfitting
    is_overfitting = overfitting_ratio > threshold

    results = {
        'is_overfitting': is_overfitting,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'overfitting_ratio': overfitting_ratio
    }

    if is_overfitting:
        warning_msg = (
            f"⚠️ POTENTIAL OVERFITTING: Test MSE is {overfitting_ratio:.2f}x train MSE. "
            f"Model performs significantly worse on test data."
        )
        logger.warning(warning_msg)
        warnings.warn(warning_msg, UserWarning)
    else:
        logger.info(
            f"✅ No overfitting detected (ratio: {overfitting_ratio:.2f} < {threshold})"
        )

    return results


# ============================================================================
# Model Predictions
# ============================================================================

def get_model_predictions(model: LinearRegression,
                         X: pd.DataFrame,
                         feature_columns: list = None) -> np.ndarray:
    """
    Generate predictions from trained HAR-R model.

    Args:
        model: Trained HAR-R model
        X: Feature DataFrame
        feature_columns: List of feature column names

    Returns:
        np.ndarray: Model predictions

    Example:
        >>> model = train_har_r_baseline(X_train, y_train)
        >>> predictions = get_model_predictions(model, X_test)
        >>> print(f"Mean prediction: {predictions.mean():.6f}")
        Mean prediction: 0.002145

    Note:
        Returns predictions for all rows (including NaN features).
        Filter NaN predictions if needed.
    """
    if feature_columns is None:
        feature_columns = HAR_FEATURE_COLUMNS

    predictions = model.predict(X[feature_columns])

    logger.info(f"✅ Predictions generated for {len(predictions)} samples")

    return predictions


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model_performance(y_true: pd.Series,
                               y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive model performance metrics.

    Metrics calculated:
    - RMSE: Root Mean Squared Error (primary accuracy metric)
    - MAE: Mean Absolute Error (robust to outliers)
    - R²: Coefficient of determination (variance explained)

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Dict with metric names as keys and values as values

    Example:
        >>> y_true = pd.Series([0.002, 0.0025, 0.003])
        >>> y_pred = np.array([0.0021, 0.0024, 0.0029])
        >>> metrics = evaluate_model_performance(y_true, y_pred)
        >>> print(f"RMSE: {metrics['rmse']:.6f}")
        RMSE: 0.000058

    Note:
        QLIKE loss is calculated separately (not here).
        QLIKE requires specialized implementation in evaluation module.
    """
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

    logger.info(
        f"✅ Model performance: RMSE={metrics['rmse']:.6f}, "
        f"MAE={metrics['mae']:.6f}, R²={metrics['r2']:.4f}"
    )

    return metrics


# ============================================================================
# Complete Training Pipeline
# ============================================================================

def complete_har_r_training_pipeline(featureset: pd.DataFrame,
                                     test_size: float = DEFAULT_TEST_SIZE,
                                     save_model: bool = False,
                                     model_path: str = None) -> Dict[str, Any]:
    """
    Complete HAR-R training pipeline: split → train → evaluate → validate.

    This function orchestrates the complete model training workflow:
    1. Set random seeds for reproducibility
    2. Temporal train/test split
    3. Train HAR-R baseline model
    4. Generate predictions
    5. Evaluate performance (train + test)
    6. Detect overfitting
    7. (Optional) Save model to disk

    Args:
        featureset: Complete featureset (HAR features + target_5d)
        test_size: Test set fraction (default: 0.2)
        save_model: If True, save trained model (default: False)
        model_path: Path to save model (default: 'models/har_r_baseline.pkl')

    Returns:
        Dict containing:
            - 'model': Trained LinearRegression model
            - 'X_train': Training features
            - 'X_test': Testing features
            - 'y_train': Training targets
            - 'y_test': Testing targets
            - 'train_predictions': Training set predictions
            - 'test_predictions': Testing set predictions
            - 'train_metrics': Training performance metrics
            - 'test_metrics': Testing performance metrics
            - 'overfitting_results': Overfitting detection results

    Raises:
        ValueError: If featureset is missing required columns
        ValueError: If insufficient data for training

    Example:
        >>> from src.feature_engineering import create_featureset
        >>> vol_data = load_volatility_data()
        >>> featureset = create_featureset(vol_data)
        >>> results = complete_har_r_training_pipeline(featureset)
        >>> print(f"Test RMSE: {results['test_metrics']['rmse']:.6f}")
        Test RMSE: 0.000847

    Note:
        This is the PRIMARY entry point for model training.
        All intermediate results are returned for analysis and debugging.
        Model is NOT saved by default - set save_model=True to persist.
    """
    logger.info("🚀 Starting complete HAR-R training pipeline")

    # Step 1: Set random seeds
    logger.info("Step 1/7: Setting random seeds...")
    set_random_seeds(RANDOM_SEED)

    # Step 2: Validate featureset
    logger.info("Step 2/7: Validating featureset...")
    required_columns = HAR_FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in featureset.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Step 3: Temporal train/test split
    logger.info("Step 3/7: Performing temporal train/test split...")
    train_data, test_data = temporal_train_test_split(featureset, test_size=test_size)

    # Extract features and targets
    X_train = train_data[HAR_FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]
    X_test = test_data[HAR_FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    # Step 4: Train HAR-R model
    logger.info("Step 4/7: Training HAR-R baseline model...")
    model = train_har_r_baseline(X_train, y_train)

    # Step 5: Generate predictions
    logger.info("Step 5/7: Generating predictions...")
    train_predictions = get_model_predictions(model, X_train)
    test_predictions = get_model_predictions(model, X_test)

    # Step 6: Evaluate performance
    logger.info("Step 6/7: Evaluating model performance...")
    train_metrics = evaluate_model_performance(y_train, train_predictions)
    test_metrics = evaluate_model_performance(y_test, test_predictions)

    # Step 7: Detect overfitting
    logger.info("Step 7/7: Detecting overfitting...")
    overfitting_results = detect_overfitting(
        model, X_train, X_test, y_train, y_test
    )

    # Compile results
    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'overfitting_results': overfitting_results
    }

    # Save model if requested
    if save_model:
        if model_path is None:
            os.makedirs('models', exist_ok=True)
            model_path = 'models/har_r_baseline.pkl'

        joblib.dump(model, model_path)
        logger.info(f"✅ Model saved to {model_path}")
        results['model_path'] = model_path

    logger.info(
        f"✅ Complete HAR-R training pipeline finished: "
        f"Test RMSE={test_metrics['rmse']:.6f}, R²={test_metrics['r2']:.4f}"
    )

    return results


# ============================================================================
# Feature Importance Analysis
# ============================================================================

def get_feature_coefficients(model: LinearRegression,
                             feature_columns: list = None) -> pd.DataFrame:
    """
    Extract and format model coefficients for feature importance analysis.

    Args:
        model: Trained HAR-R model
        feature_columns: List of feature names (default: HAR_FEATURE_COLUMNS)

    Returns:
        DataFrame with columns:
            - 'feature': Feature name
            - 'coefficient': Model coefficient
            - 'abs_coefficient': Absolute coefficient (for ranking)

    Example:
        >>> model = train_har_r_baseline(X_train, y_train)
        >>> coefficients = get_feature_coefficients(model)
        >>> print(coefficients)
                feature  coefficient  abs_coefficient
        0  har_daily_vol          0.52              0.52
        1  har_weekly_vol          0.31              0.31
        2  har_monthly_vol         0.18              0.18

    Note:
        Higher absolute coefficients indicate stronger influence on predictions.
        For HAR-R models, monthly feature typically has highest coefficient.
    """
    if feature_columns is None:
        feature_columns = HAR_FEATURE_COLUMNS

    coefficients = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    })

    # Sort by absolute coefficient (descending)
    coefficients = coefficients.sort_values('abs_coefficient', ascending=False)

    logger.info(f"✅ Feature coefficients extracted: {len(coefficients)} features")

    return coefficients
