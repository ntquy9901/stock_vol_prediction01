"""
HAR Features Generator for Volatility Prediction

Generates Heterogeneous Autoregressive (HAR) features from Parkinson volatility
for crypto and stock volatility prediction.

Standard HAR methodology uses:
- Daily: 1-day rolling mean (recent volatility)
- Weekly: 5-day rolling mean (weekly pattern)
- Monthly: 22-day rolling mean (monthly trend)

This enables the CryptoMamba Enhanced V2 model to train by providing the required
input features: har_daily_vol, har_weekly_vol, har_monthly_vol.

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

import pandas as pd
import numpy as np
from typing import Union


def generate_har_features(
    data: Union[pd.DataFrame, str],
    volatility_col: str = 'parkinson_volatility',
    daily_window: int = 1,
    weekly_window: int = 5,
    monthly_window: int = 22
) -> pd.DataFrame:
    """
    Generate HAR features from volatility data.

    Args:
        data: DataFrame containing volatility data OR path to CSV file
        volatility_col: Name of volatility column (default: 'parkinson_volatility')
        daily_window: Window for daily HAR feature (default: 1)
        weekly_window: Window for weekly HAR feature (default: 5)
        monthly_window: Window for monthly HAR feature (default: 22)

    Returns:
        DataFrame with added HAR features: har_daily_vol, har_weekly_vol, har_monthly_vol

    Raises:
        ValueError: If volatility column not found
        ValueError: If insufficient data for windows
    """
    # Load data if path provided
    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()

    # Validate volatility column exists
    if volatility_col not in df.columns:
        raise ValueError(
            f"Volatility column '{volatility_col}' not found. "
            f"Available columns: {list(df.columns[:5])}..."
        )

    # Validate sufficient data
    min_required = monthly_window + 1
    if len(df) < min_required:
        raise ValueError(
            f"Insufficient data for HAR features: {len(df)} rows, "
            f"need at least {min_required} rows for {monthly_window}-day window"
        )

    # Generate HAR features
    # Daily: 1-day rolling mean (recent volatility)
    df['har_daily_vol'] = df[volatility_col].rolling(
        window=daily_window, min_periods=1
    ).mean()

    # Weekly: 5-day rolling mean (weekly pattern)
    df['har_weekly_vol'] = df[volatility_col].rolling(
        window=weekly_window, min_periods=1
    ).mean()

    # Monthly: 22-day rolling mean (monthly trend)
    df['har_monthly_vol'] = df[volatility_col].rolling(
        window=monthly_window, min_periods=1
    ).mean()

    # Forward fill NaN values from rolling windows
    df['har_daily_vol'] = df['har_daily_vol'].fillna(method='ffill').fillna(0)
    df['har_weekly_vol'] = df['har_weekly_vol'].fillna(method='ffill').fillna(0)
    df['har_monthly_vol'] = df['har_monthly_vol'].fillna(method='ffill').fillna(0)

    # Validate HAR features were generated
    required_cols = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"Failed to generate HAR feature: {col}")

    # Validate HAR features have valid values
    for col in required_cols:
        if df[col].isnull().all():
            raise ValueError(f"HAR feature '{col}' is all NaN after generation")
        if (df[col] == 0).all():
            raise ValueError(f"HAR feature '{col}' is all zeros after generation")

    print(f"[OK] HAR features generated: {required_cols}")
    print(f"   Daily range: [{df['har_daily_vol'].min():.6f}, {df['har_daily_vol'].max():.6f}]")
    print(f"   Weekly range: [{df['har_weekly_vol'].min():.6f}, {df['har_weekly_vol'].max():.6f}]")
    print(f"   Monthly range: [{df['har_monthly_vol'].min():.6f}, {df['har_monthly_vol'].max():.6f}]")

    return df


def validate_har_features(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame contains required HAR features.

    Args:
        df: DataFrame to validate

    Returns:
        True if all HAR features exist and have valid values

    Raises:
        ValueError: If HAR features are missing or invalid
    """
    required_cols = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']

    # Check columns exist
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required HAR columns: {missing}")

    # Check for valid values
    for col in required_cols:
        if df[col].isnull().all():
            raise ValueError(f"HAR feature '{col}' is all NaN")
        if (df[col] == 0).all():
            raise ValueError(f"HAR feature '{col}' is all zeros")
        if (df[col] < 0).any():
            raise ValueError(f"HAR feature '{col}' contains negative values (invalid for volatility)")

    return True


# CLI interface for standalone usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.common.har_features <data_file_or_dir>")
        print("Example: python -m src.common.har_features data/processed/stock_001.csv")
        sys.exit(1)

    data_path = sys.argv[1]

    try:
        df = generate_har_features(data_path)
        print(f"\n[OK] HAR features successfully generated for {len(df)} rows")
        print(f"Output columns: {list(df.columns)}")

        # Save back to same file
        if data_path.endswith('.csv'):
            df.to_csv(data_path, index=False)
            print(f"Updated: {data_path}")

    except Exception as e:
        print(f"\n[X] Error: {e}")
        sys.exit(1)
