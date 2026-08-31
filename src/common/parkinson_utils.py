"""
Common Data Processing Functions for Volatility Prediction

This module contains shared data processing utilities used across
different baseline models (HAR-R, LSTM, etc.).

Author: Stock Volatility Prediction Team
Date: 2026-06-17
"""

import pandas as pd
import numpy as np
from typing import Tuple

# Relative tolerance for OHLC-geometry validation: rows are rejected only when a bar
# violates high>=low / high>=max(open,close) / low<=min(open,close) by more than this
# fraction of the bar's price scale. Adjusted-price floating-point noise (~1e-17) passes;
# material violations (e.g. low well above open) are rejected. (data audit 2026-08-29)
_OHLC_GEOMETRY_RTOL = 1e-6


def calculate_parkinson_variance(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Parkinson volatility estimator from OHLCV data.

    Formula: σ² = (log(H/L)²) / (4*log(2))

    Args:
        df: DataFrame with high and low columns

    Returns:
        pd.Series: Parkinson volatility values
    """
    # Normalize column names
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]

    high = df['high']
    low = df['low']

    # Calculate Parkinson volatility
    parkinson = (np.log(high / low) ** 2) / (4 * np.log(2))

    return parkinson


def validate_ohlc_data(df: pd.DataFrame) -> bool:
    """
    Validate OHLCV data integrity.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        bool: True if valid

    Raises:
        ValueError: If OHLCV relationships are invalid
    """
    # Normalize column names
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]

    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Relaxed validation: only check for critical issues
    # 1. No negative prices or volume
    if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
        raise ValueError("Prices and volume must be non-negative")

    # 2. No NaN values in critical columns
    if df[['open', 'high', 'low', 'close']].isnull().any().any():
        raise ValueError("OHLC prices cannot be NaN")

    # 3. Valid OHLC ordering: high is the bar max, low is the bar min. Impossible
    #    geometry (e.g. high < low, or high below open/close) would otherwise pass
    #    validation and produce a finite-but-meaningless Parkinson variance. Use a
    #    RELATIVE tolerance so adjusted-price floating-point noise (e.g. Yahoo
    #    auto_adjust differences ~1e-17) is not flagged, while material violations are.
    row_max = df[['open', 'close']].max(axis=1)
    row_min = df[['open', 'close']].min(axis=1)
    scale = df[['open', 'high', 'low', 'close']].abs().max(axis=1).clip(lower=1e-12)
    tol = _OHLC_GEOMETRY_RTOL * scale
    bad_geometry = (
        (df['low'] - df['high'] > tol)
        | (row_max - df['high'] > tol)
        | (df['low'] - row_min > tol)
    )
    if bad_geometry.any():
        raise ValueError(
            "Invalid OHLC geometry: require high>=low, high>=max(open,close), "
            "low<=min(open,close)"
        )

    return True


def process_single_stock(raw_file: str, output_dir: str) -> Tuple[str, int]:
    """
    Process single stock OHLCV data to Parkinson volatility.

    Args:
        raw_file: Path to raw OHLCV CSV file
        output_dir: Directory to save processed CSV

    Returns:
        Tuple[str, int]: (ticker, num_records)
    """
    import os

    # Extract ticker from filename
    ticker = os.path.basename(raw_file).replace('_ohlcv.csv', '').replace('.csv', '')

    try:
        # Load raw data
        df = pd.read_csv(raw_file)

        # Validate OHLCV data
        validate_ohlc_data(df)

        # Calculate Parkinson volatility
        parkinson_vol = calculate_parkinson_variance(df)

        # Normalize date to plain YYYY-MM-DD: most raw sources already store this, but
        # VPB/VRE store "YYYY-MM-DD HH:MM:SS+07:00" (mixed format across raw sources) —
        # taking the token before the first space keeps the calendar date as originally
        # written, with no tz-conversion arithmetic that could shift it across midnight.
        date_str = df['date'].astype(str).str.split(' ').str[0]

        # Create processed DataFrame
        processed_df = pd.DataFrame({
            'date': date_str,
            'parkinson_variance': parkinson_vol
        })

        # Remove NaN and infinite values
        processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
        processed_df = processed_df.dropna()

        # Clip extreme values (Parkinson vol should typically be < 0.1)
        processed_df['parkinson_variance'] = processed_df['parkinson_variance'].clip(upper=0.1)

        # Save to processed directory
        output_file = os.path.join(output_dir, f'{ticker}_processed.csv')
        processed_df.to_csv(output_file, index=False)

        return ticker, len(processed_df)

    except Exception as e:
        print(f"[ERROR] Error processing {ticker}: {str(e)}")
        return ticker, 0
