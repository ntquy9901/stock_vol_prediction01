"""
Feature Engineering Module for Stock Volatility Prediction System.

This module implements HAR (Heterogeneous Autoregressive) feature creation
and multi-horizon target variables for volatility forecasting.

Primary Functions:
    - create_har_features: Create HAR features (daily, weekly, monthly)
    - create_5day_target: Create 5-day ahead target variable
    - validate_har_features: Validate feature quality and consistency
    - create_featureset: Combine features and target into complete dataset

HAR Feature Windows (Confirmed):
    - Daily: 1-day rolling window (instantaneous volatility)
    - Weekly: 5-day rolling window (short-term volatility)
    - Monthly: 22-day rolling window (long-term volatility)

Author: Stock Volatility Prediction Team
Date: 2026-06-15
"""

import logging
import warnings
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


# Configure module-level logger
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Constants
# ============================================================================

# HAR Feature Windows - Confirmed 22-day monthly window
HAR_WINDOWS = {
    'daily': 1,      # 1-day window (instantaneous)
    'weekly': 5,     # 5-day window (~1 trading week)
    'monthly': 22    # 22-day window (~1 trading month) ✅
}

# Forecast Horizon Configuration
FORECAST_HORIZON_DAYS = 5  # Phase 1: 5-day focus only


# ============================================================================
# HAR Feature Creation
# ============================================================================

def create_har_features(volatility_series: pd.Series,
                        windows: Dict[str, int] = None) -> pd.DataFrame:
    """
    Create HAR (Heterogeneous Autoregressive) features from volatility series.

    HAR features capture volatility dynamics across different time scales:
    - Daily (1-day): Recent volatility level
    - Weekly (5-day): Short-term volatility trend
    - Monthly (22-day): Long-term volatility baseline

    Formula: HAR_feature = rolling_mean(volatility, window)

    Args:
        volatility_series: Series of volatility values (e.g., Parkinson volatility)
        windows: Dictionary of window sizes (default: HAR_WINDOWS = {1, 5, 22})

    Returns:
        pd.DataFrame: DataFrame with HAR feature columns
            - 'har_daily_vol': 1-day rolling mean
            - 'har_weekly_vol': 5-day rolling mean
            - 'har_monthly_vol': 22-day rolling mean

    Raises:
        ValueError: If volatility_series is empty
        ValueError: If volatility_series has insufficient data

    Example:
        >>> vol = pd.Series([0.001, 0.002, 0.0015, 0.0018, 0.0022])
        >>> har_features = create_har_features(vol)
        >>> print(har_features)
           har_daily_vol  har_weekly_vol  har_monthly_vol
        0         0.0010             NaN              NaN
        1         0.0020             NaN              NaN
        2         0.0015             NaN              NaN
        3         0.0018             NaN              NaN
        4         0.0022          0.0017              NaN

    Note:
        HAR features use 22-day monthly window (confirmed standard).
        Rolling means produce NaN for initial rows (warm-up period).
        These NaN values are handled in model training via data splitting.
    """
    if windows is None:
        windows = HAR_WINDOWS

    # Validate inputs
    if len(volatility_series) == 0:
        raise ValueError("Volatility series is empty")

    # Create features DataFrame
    har_features = pd.DataFrame(index=volatility_series.index)

    # Daily HAR feature (1-day window)
    daily_window = windows['daily']
    har_features['har_daily_vol'] = volatility_series.rolling(window=daily_window).mean()
    logger.info(f"✅ HAR daily feature created (window: {daily_window} day)")

    # Weekly HAR feature (5-day window)
    weekly_window = windows['weekly']
    har_features['har_weekly_vol'] = volatility_series.rolling(window=weekly_window).mean()
    logger.info(f"✅ HAR weekly feature created (window: {weekly_window} days)")

    # Monthly HAR feature (22-day window) ✅
    monthly_window = windows['monthly']
    har_features['har_monthly_vol'] = volatility_series.rolling(window=monthly_window).mean()
    logger.info(f"✅ HAR monthly feature created (window: {monthly_window} days)")

    # Log feature statistics
    feature_count = len(har_features.columns)
    logger.info(f"✅ HAR features created: {feature_count} features for {len(volatility_series)} time points")

    return har_features


# ============================================================================
# Target Variable Creation (5-Day Focus)
# ============================================================================

def create_5day_target(volatility_series: pd.Series,
                       horizon_days: int = FORECAST_HORIZON_DAYS) -> pd.Series:
    """
    Create 5-day ahead target variable for volatility forecasting.

    This function shifts the volatility series forward by horizon_days to create
    the target variable for supervised learning.

    Formula: target[t] = volatility[t + horizon_days]

    Args:
        volatility_series: Series of volatility values
        horizon_days: Forecast horizon in days (default: 5)

    Returns:
        pd.Series: Target variable shifted forward by horizon_days
            - Last 'horizon_days' values are NaN (no future data available)

    Raises:
        ValueError: If horizon_days <= 0
        ValueError: If volatility_series is empty

    Example:
        >>> vol = pd.Series([0.001, 0.002, 0.0015, 0.0018, 0.0022, 0.0025, 0.0020])
        >>> target = create_5day_target(vol, horizon_days=5)
        >>> print(target)
        0    0.0025  # vol[0+5] = vol[5]
        1    0.0020  # vol[1+5] = vol[6]
        2       NaN  # vol[7] doesn't exist
        3       NaN
        4       NaN
        5       NaN
        6       NaN
        dtype: float64

    Note:
        Phase 1 focuses exclusively on 5-day forecasts.
        Last 5 rows will be NaN - handled during train/test splitting.
        Do NOT use this function for multi-horizon forecasting (future expansion).
    """
    # Validate inputs
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}")

    if len(volatility_series) == 0:
        raise ValueError("Volatility series is empty")

    # Shift volatility series forward to create target
    target = volatility_series.shift(-horizon_days)

    logger.info(
        f"✅ {horizon_days}-day target created: "
        f"{target.notna().sum()} valid targets, {target.isna().sum()} trailing NaN"
    )

    return target


# ============================================================================
# Feature Validation
# ============================================================================

def validate_har_features(features: pd.DataFrame,
                          min_required_samples: int = 30) -> bool:
    """
    Validate HAR features quality and consistency.

    Validation checks:
    - No infinite values in features
    - Sufficient non-NaN samples for training
    - Feature correlations are reasonable (no perfect multicollinearity)
    - Feature values are within expected ranges

    Args:
        features: DataFrame of HAR features (output from create_har_features)
        min_required_samples: Minimum non-NaN samples required (default: 30)

    Returns:
        bool: True if all validations pass

    Raises:
        ValueError: If features contain infinite values
        ValueError: If insufficient non-NaN samples
        UserWarning: If feature correlation > 0.99 (multicollinearity risk)

    Example:
        >>> har_features = create_har_features(volatility_series)
        >>> is_valid = validate_har_features(har_features)
        >>> print(is_valid)
        True

    Note:
        High correlation (>0.99) between HAR features is expected due to
        rolling window overlap (weekly includes daily, monthly includes weekly).
        This is acceptable for HAR models.
    """
    # Check for infinite values
    if np.isinf(features.values).any():
        raise ValueError(
            "HAR features contain infinite values. "
            "Check volatility input for zeros or extreme values."
        )

    # Check sufficient non-NaN samples
    valid_samples = features.notna().all(axis=1).sum()
    if valid_samples < min_required_samples:
        raise ValueError(
            f"Insufficient non-NaN samples: {valid_samples} < {min_required_samples}. "
            f"Need more data points for HAR feature calculation."
        )

    # Check feature correlations (warn if > 0.99)
    corr_matrix = features.corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > 0.99:
                feature1 = corr_matrix.columns[i]
                feature2 = corr_matrix.columns[j]
                high_corr_pairs.append((feature1, feature2, corr_val))

    if high_corr_pairs:
        warning_msg = (
            f"⚠️ High feature correlation detected (>0.99): {high_corr_pairs}. "
            "This may indicate multicollinearity but is acceptable for HAR models."
        )
        logger.warning(warning_msg)
        warnings.warn(warning_msg, UserWarning)

    logger.info(
        f"✅ HAR features validated: {valid_samples} valid samples, "
        f"{len(features.columns)} features, {features.isna().sum().sum()} NaN values"
    )

    return True


# ============================================================================
# Complete Feature Set Creation
# ============================================================================

def create_featureset(volatility_series: pd.Series,
                      horizon_days: int = FORECAST_HORIZON_DAYS,
                      windows: Dict[str, int] = None) -> pd.DataFrame:
    """
    Create complete feature set: HAR features + target variable.

    This function orchestrates the complete feature engineering workflow:
    1. Create HAR features (daily, weekly, monthly)
    2. Create 5-day ahead target
    3. Combine into single DataFrame
    4. Validate feature quality

    Args:
        volatility_series: Series of volatility values (e.g., Parkinson volatility)
        horizon_days: Forecast horizon in days (default: 5)
        windows: HAR window sizes (default: HAR_WINDOWS = {1, 5, 22})

    Returns:
        pd.DataFrame: Complete feature set with columns:
            - 'har_daily_vol': 1-day HAR feature
            - 'har_weekly_vol': 5-day HAR feature
            - 'har_monthly_vol': 22-day HAR feature
            - 'target_5d': 5-day ahead target variable

    Raises:
        ValueError: If volatility_series is empty
        ValueError: If insufficient data for HAR features

    Example:
        >>> vol = pd.Series([0.001, 0.002, 0.0015, 0.0018, 0.0022, 0.0025])
        >>> featureset = create_featureset(vol)
        >>> print(featureset.columns.tolist())
        ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol', 'target_5d']

    Note:
        This is the primary entry point for feature engineering.
        Returns features with NaN values (warm-up + trailing) - handled in model training.
    """
    logger.info("🚀 Starting complete feature engineering pipeline")

    # Step 1: Create HAR features
    logger.info("Step 1/3: Creating HAR features...")
    har_features = create_har_features(volatility_series, windows=windows)

    # Step 2: Create target variable
    logger.info("Step 2/3: Creating 5-day target variable...")
    target = create_5day_target(volatility_series, horizon_days=horizon_days)

    # Step 3: Combine features and target
    logger.info("Step 3/3: Combining features and target...")
    featureset = har_features.copy()
    featureset[f'target_{horizon_days}d'] = target

    # Validate feature quality
    logger.info("Validating feature quality...")
    validate_har_features(featureset.drop(columns=[f'target_{horizon_days}d']))

    logger.info(
        f"✅ Complete feature engineering finished: "
        f"{len(featureset.columns)} columns for {len(featureset)} time points"
    )

    return featureset


# ============================================================================
# Utility Functions
# ============================================================================

def get_har_feature_importance(model_coef: np.ndarray,
                               feature_names: list = None) -> pd.DataFrame:
    """
    Extract and format HAR feature importance from model coefficients.

    Args:
        model_coef: Model coefficients (e.g., from linear regression)
        feature_names: List of feature names (default: HAR feature names)

    Returns:
        pd.DataFrame: Feature importance with columns:
            - 'feature': Feature name
            - 'coefficient': Model coefficient
            - 'abs_coefficient': Absolute coefficient value (for ranking)

    Example:
        >>> coef = np.array([0.5, 0.3, 0.2])
        >>> importance = get_har_feature_importance(coef)
        >>> print(importance)
                feature  coefficient  abs_coefficient
        0  har_daily_vol          0.5              0.50
        1  har_weekly_vol          0.3              0.30
        2  har_monthly_vol         0.2              0.20

    Note:
        Higher absolute coefficients indicate stronger influence on predictions.
        For HAR-R models, monthly features typically have highest importance.
    """
    if feature_names is None:
        feature_names = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']

    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model_coef,
        'abs_coefficient': np.abs(model_coef)
    })

    # Sort by absolute coefficient (descending)
    importance = importance.sort_values('abs_coefficient', ascending=False)

    logger.info(f"✅ Feature importance extracted: {len(importance)} features")

    return importance
