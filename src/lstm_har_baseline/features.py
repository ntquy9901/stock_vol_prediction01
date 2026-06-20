"""
HAR Feature Engineering for LSTM-HAR Baseline

This module creates Heterogeneous Autoregressive (HAR) features
for LSTM model input instead of raw Parkinson volatility.

Features:
    - har_daily_vol: 1-day average (daily)
    - har_weekly_vol: 5-day average (weekly)
    - har_monthly_vol: 22-day average (monthly)

Author: Stock Volatility Prediction Team
Date: 2026-06-18
"""

import pandas as pd
import numpy as np


def create_har_features(parkinson_volatility: pd.Series,
                       windows: dict = None) -> pd.DataFrame:
    """
    Create HAR features from Parkinson volatility time series.

    HAR (Heterogeneous Autoregressive) features capture multi-scale
    temporal patterns using different rolling windows.

    Args:
        parkinson_volatility: Parkinson volatility time series
        windows: Dictionary of window sizes (default: daily=1, weekly=5, monthly=22)

    Returns:
        pd.DataFrame: DataFrame with HAR features

    Example:
        >>> vol = pd.Series([0.01, 0.02, 0.015, ...])
        >>> har_features = create_har_features(vol)
        >>> print(har_features.head())
    """
    if windows is None:
        windows = {
            'daily': 1,
            'weekly': 5,
            'monthly': 22
        }

    features = pd.DataFrame()

    # Create HAR features using rolling windows
    for name, window in windows.items():
        feature_name = f'har_{name}_vol'
        features[feature_name] = parkinson_volatility.rolling(window=window).mean()

    # Remove NaN values created by rolling windows
    features = features.dropna()

    return features


def validate_har_features(har_df: pd.DataFrame) -> bool:
    """
    Validate HAR features DataFrame.

    Args:
        har_df: DataFrame with HAR features

    Returns:
        bool: True if valid

    Raises:
        ValueError: If HAR features are invalid
    """
    required_cols = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']
    missing_cols = [col for col in required_cols if col not in har_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required HAR features: {missing_cols}")

    # Check for NaN values
    if har_df[required_cols].isnull().any().any():
        raise ValueError("HAR features contain NaN values")

    # Check for extreme values (Parkinson vol should be reasonable)
    if (har_df[required_cols] > 0.1).any().any():
        raise ValueError("HAR features contain extreme values > 0.1")

    if (har_df[required_cols] < 0).any().any():
        raise ValueError("HAR features contain negative values")

    return True


def create_har_dataset_with_targets(har_df: pd.DataFrame,
                                    forecast_horizon: int = 5) -> pd.DataFrame:
    """
    Create HAR dataset with future target variable.

    Args:
        har_df: DataFrame with HAR features
        forecast_horizon: Days ahead to predict (default: 5)

    Returns:
        pd.DataFrame: HAR features + target column

    Example:
        >>> har_features = create_har_features(volatility)
        >>> dataset = create_har_dataset_with_targets(har_features, forecast_horizon=5)
        >>> print(dataset[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol', 'target']].head())
    """
    dataset = har_df.copy()

    # Create target: HAR daily volatility at t + forecast_horizon
    # We use har_daily_vol as the target (it's the same as raw Parkinson vol)
    dataset['target'] = dataset['har_daily_vol'].shift(-forecast_horizon)

    # Remove NaN values created by shifting
    dataset = dataset.dropna()

    return dataset