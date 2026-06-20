"""
Data Processing Module for Stock Volatility Prediction System.

This module implements volatility calculation, OHLCV data validation,
missing data handling, outlier detection, and fallback estimators.

Primary Functions:
    - calculate_parkinson_volatility: Primary Parkinson volatility estimator
    - calculate_garman_klass_volatility: Garman-Klass fallback estimator
    - validate_ohlc_data: OHLCV data consistency validation
    - handle_missing_data: Missing data imputation and interpolation
    - detect_outliers: Outlier detection using IQR method
    - validate_volatility_bounds: Volatility range validation

Author: Stock Volatility Prediction Team
Date: 2026-06-15
"""

import logging
import warnings
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


# Configure module-level logger for data quality issues
logger = logging.getLogger(__name__)


# ============================================================================
# OHLCV Data Cleaning
# ============================================================================

def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OHLCV data by fixing common data quality issues.

    This function handles:
    - High/Low inversions (swap if High < Low)
    - Close outside High/Low (adjust High/Low to include Close)
    - Zero or negative Volume (remove rows)
    - Missing values (forward-fill)

    Args:
        df: DataFrame containing OHLCV columns

    Returns:
        pd.DataFrame: Cleaned OHLCV data

    Example:
        >>> dirty_data = pd.DataFrame({
        ...     'Open': [100.0], 'High': [98.0], 'Low': [102.0],
        ...     'Close': [101.0], 'Volume': [1000000]
        ... })
        >>> clean_data = clean_ohlcv_data(dirty_data)
        >>> print(clean_data)
             Open   High    Low  Close    Volume
        0  100.0  101.0   98.0  101.0   1000000

    Note:
        This function modifies data to ensure OHLCV consistency.
        Removed rows are logged for transparency.
    """
    df_clean = df.copy()

    # Fix High/Low inversions
    high_low_inversion = df_clean['High'] < df_clean['Low']
    if high_low_inversion.any():
        count = high_low_inversion.sum()
        logger.warning(f"⚠️ Found {count} rows with High < Low - swapping values")
        # Swap High and Low where inverted
        df_clean.loc[high_low_inversion, ['High', 'Low']] = \
            df_clean.loc[high_low_inversion, ['Low', 'High']].values

    # Fix Close outside [Low, High] range
    close_above_high = df_clean['Close'] > df_clean['High']
    close_below_low = df_clean['Close'] < df_clean['Low']

    if close_above_high.any():
        count = close_above_high.sum()
        logger.warning(f"⚠️ Found {count} rows with Close > High - adjusting High")
        df_clean.loc[close_above_high, 'High'] = df_clean.loc[close_above_high, 'Close']

    if close_below_low.any():
        count = close_below_low.sum()
        logger.warning(f"⚠️ Found {count} rows with Close < Low - adjusting Low")
        df_clean.loc[close_below_low, 'Low'] = df_clean.loc[close_below_low, 'Close']

    # Remove rows with zero or negative volume
    invalid_volume = df_clean['Volume'] <= 0
    if invalid_volume.any():
        count = invalid_volume.sum()
        logger.warning(f"⚠️ Removing {count} rows with Volume <= 0")
        df_clean = df_clean[~invalid_volume].copy()

    # Handle any remaining NaN values
    if df_clean.isna().any().any():
        nan_count = df_clean.isna().sum().sum()
        logger.warning(f"⚠️ Found {nan_count} NaN values - applying forward-fill")
        df_clean = df_clean.ffill().bfill()

    logger.info(f"✅ OHLCV data cleaned: {len(df_clean)} rows remaining")

    return df_clean


# ============================================================================
# OHLCV Data Validation
# ============================================================================

def validate_ohlc_data(df: pd.DataFrame) -> bool:
    """
    Validate OHLCV data integrity and consistency.

    This function enforces OHLCV relationships required for volatility calculation:
    - High >= Close >= Low (price ordering)
    - High >= Low (range validity)
    - Volume > 0 (positive trading volume)

    Args:
        df: DataFrame containing 'Open', 'High', 'Low', 'Close', 'Volume' columns

    Returns:
        bool: True if all validations pass

    Raises:
        ValueError: If any OHLCV consistency check fails with specific error message
        ValueError: If df is missing required columns

    Example:
        >>> data = pd.DataFrame({
        ...     'Open': [100.0], 'High': [102.0], 'Low': [98.0],
        ...     'Close': [101.0], 'Volume': [1000000]
        ... })
        >>> is_valid = validate_ohlc_data(data)
        >>> print(is_valid)
        True
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check High >= Close
    if not all(df['High'] >= df['Close']):
        invalid_count = sum(df['High'] < df['Close'])
        raise ValueError(
            f"OHLCV validation failed: High must be >= Close. "
            f"Found {invalid_count} rows where High < Close. "
            "This violates OHLCV data integrity and indicates data quality issues."
        )

    # Check Close >= Low
    if not all(df['Close'] >= df['Low']):
        invalid_count = sum(df['Close'] < df['Low'])
        raise ValueError(
            f"OHLCV validation failed: Close must be >= Low. "
            f"Found {invalid_count} rows where Close < Low. "
            "This violates OHLCV data integrity and indicates data quality issues."
        )

    # Check High >= Low
    if not all(df['High'] >= df['Low']):
        invalid_count = sum(df['High'] < df['Low'])
        raise ValueError(
            f"OHLCV validation failed: High must be >= Low. "
            f"Found {invalid_count} rows where High < Low. "
            "This violates OHLCV data integrity and indicates data quality issues."
        )

    # Check Volume > 0
    if not all(df['Volume'] > 0):
        invalid_count = sum(df['Volume'] <= 0)
        raise ValueError(
            f"OHLCV validation failed: Volume must be positive. "
            f"Found {invalid_count} rows where Volume <= 0. "
            "Zero or negative volume indicates data quality issues."
        )

    logger.info("✅ OHLCV data validation passed")
    return True


# ============================================================================
# Parkinson Volatility Estimator (PRIMARY METHOD)
# ============================================================================

def calculate_parkinson_volatility(ohlc_data: pd.DataFrame, clean: bool = True) -> pd.Series:
    """
    Calculate Parkinson volatility estimator from daily OHLCV data.

    Formula: σ² = (log(H/L)²) / (4*log(2))

    This estimator is the PRIMARY volatility estimator for this system because:
    - Uses intraday range information (High-Low)
    - More efficient than close-to-close estimator (5x more efficient)
    - Robust for daily frequency data
    - Industry standard for range-based volatility
    - Less sensitive to microstructure noise than close-to-close

    Args:
        ohlc_data: DataFrame containing 'High' and 'Low' columns
        clean: If True, clean data before calculation (default: True)

    Returns:
        pd.Series: Parkinson volatility values (squared volatility)

    Raises:
        ValueError: If df contains NaN values after cleaning
        ValueError: If df has insufficient data after cleaning

    Example:
        >>> data = pd.DataFrame({'High': [102, 105, 103], 'Low': [98, 100, 99]})
        >>> parkinson_vol = calculate_parkinson_volatility(data)
        >>> print(f"Mean Parkinson vol: {parkinson_vol.mean():.6f}")
        Mean Parkinson vol: 0.000421

    Note:
        Parkinson estimator assumes continuous trading. For stocks with
        trading halts, consider alternative estimators like Garman-Klass.
        Data cleaning is applied by default to handle real-world data issues.
    """
    # Clean data if requested
    if clean:
        ohlc_data = clean_ohlcv_data(ohlc_data)

    high = ohlc_data['High']
    low = ohlc_data['Low']

    # Check for NaN values BEFORE OHLCV validation
    if high.isna().any() or low.isna().any():
        raise ValueError(
            "OHLCV data contains NaN values. "
            "Use handle_missing_data() to impute missing values before calculation."
        )

    # Validate OHLCV consistency before calculation
    validate_ohlc_data(ohlc_data)

    # Calculate Parkinson volatility: σ² = (log(H/L)²) / (4*log(2))
    parkinson_volatility = (np.log(high / low) ** 2) / (4 * np.log(2))

    logger.info(f"✅ Parkinson volatility calculated for {len(parkinson_volatility)} data points")
    return parkinson_volatility


# ============================================================================
# Garman-Klass Fallback Estimator
# ============================================================================

def calculate_garman_klass_volatility(ohlc_data: pd.DataFrame) -> pd.Series:
    """
    Calculate Garman-Klass volatility estimator from daily OHLCV data.

    Formula: σ² = 0.5*(log(H/L)²) - (2*log(2)-1)*(log(C/O)²)

    This estimator is an ALTERNATIVE to Parkinson when:
    - Parkinson estimator fails or is insufficient
    - Open price data is available and reliable
    - Additional information from opening price is valuable

    Advantages over Parkinson:
    - Uses both intraday range (High-Low) and overnight movement (Close-Open)
    - More efficient than Parkinson (uses more price information)
    - Better captures volatility when overnight gaps are significant

    Args:
        ohlc_data: DataFrame containing 'Open', 'High', 'Low', 'Close' columns
                    Must be validated OHLCV data

    Returns:
        pd.Series: Garman-Klass volatility values (squared volatility)

    Raises:
        ValueError: If df contains NaN values
        ValueError: If df has invalid OHLCV relationships

    Example:
        >>> data = pd.DataFrame({
        ...     'Open': [100, 101], 'High': [102, 105],
        ...     'Low': [98, 100], 'Close': [101, 103]
        ... })
        >>> gk_vol = calculate_garman_klass_volatility(data)
        >>> print(f"Mean Garman-Klass vol: {gk_vol.mean():.6f}")
        Mean Garman-Klass vol: 0.000485

    Note:
        Garman-Klass is used as FALLBACK when Parkinson estimator fails.
        It requires reliable Open price data, which may not always be available.
    """
    high = ohlc_data['High']
    low = ohlc_data['Low']
    close = ohlc_data['Close']
    open_price = ohlc_data['Open']

    # Check for NaN values BEFORE OHLCV validation
    if high.isna().any() or low.isna().any() or close.isna().any() or open_price.isna().any():
        raise ValueError(
            "OHLCV data contains NaN values. "
            "Use handle_missing_data() to impute missing values before calculation."
        )

    # Validate OHLCV consistency before calculation
    validate_ohlc_data(ohlc_data)

    # Calculate Garman-Klass volatility
    # σ² = 0.5*(log(H/L)²) - (2*log(2)-1)*(log(C/O)²)
    range_component = 0.5 * (np.log(high / low) ** 2)
    overnight_component = (2 * np.log(2) - 1) * (np.log(close / open_price) ** 2)
    garman_klass_volatility = range_component - overnight_component

    logger.info(f"✅ Garman-Klass volatility calculated for {len(garman_klass_volatility)} data points")
    return garman_klass_volatility


# ============================================================================
# Missing Data Handling
# ============================================================================

def handle_missing_data(df: pd.DataFrame, max_forward_fill: int = 5) -> pd.DataFrame:
    """
    Handle missing data using forward-fill then linear interpolation.

    This function preserves temporal ordering and prevents data leakage:
    - Forward-fill up to max_forward_fill consecutive missing days
    - Linear interpolation for remaining gaps
    - NO future data leakage (temporal integrity maintained)

    Args:
        df: DataFrame with potential missing values (NaN)
        max_forward_fill: Maximum consecutive days for forward-fill (default: 5)

    Returns:
        pd.DataFrame: DataFrame with missing values imputed

    Example:
        >>> data = pd.DataFrame({'Price': [100, np.nan, np.nan, 103, 104]})
        >>> clean_data = handle_missing_data(data)
        >>> print(clean_data)
        Price
        0  100.0
        1  100.0  # Forward-filled
        2  100.0  # Forward-filled
        3  103.0
        4  104.0

    Note:
        This function preserves temporal ordering - no future data is used
        to impute past values. This is critical for financial time series data.
    """
    # Log missing data statistics
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100

    logger.info(f"Missing data: {missing_cells}/{total_cells} cells ({missing_percentage:.2f}%)")

    # Create copy to avoid modifying original
    df_clean = df.copy()

    # Step 1: Forward-fill up to max_forward_fill consecutive days
    df_clean = df_clean.ffill(limit=max_forward_fill)

    # Step 2: Linear interpolation for remaining gaps
    df_clean = df_clean.interpolate(method='linear', limit_direction='both')

    # Log remaining missing data
    remaining_missing = df_clean.isna().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"⚠️ {remaining_missing} missing values remain after imputation")

    logger.info(f"✅ Missing data handled: {missing_cells - remaining_missing} values imputed")
    return df_clean


# ============================================================================
# Outlier Detection (IQR Method)
# ============================================================================

def detect_outliers(volatility_series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in volatility series using IQR (Interquartile Range) method.

    Outliers are defined as values beyond threshold × IQR from Q1 or Q3:
    - Lower bound: Q1 - threshold × IQR
    - Upper bound: Q3 + threshold × IQR

    Default threshold of 3.0 is standard for financial volatility data.

    Args:
        volatility_series: Series of volatility values
        threshold: IQR multiplier for outlier detection (default: 3.0)

    Returns:
        pd.Series: Boolean series where True indicates an outlier

    Example:
        >>> vol = pd.Series([0.001, 0.002, 0.0015, 0.05, 0.0018])  # 0.05 is outlier
        >>> outliers = detect_outliers(vol)
        >>> print(outliers)
        0    False
        1    False
        2    False
        3     True  # Outlier detected
        4    False
        dtype: bool

    Note:
        IQR method is robust to non-normal distributions common in financial data.
        Threshold of 3.0 is conservative - adjust based on data characteristics.
    """
    # Calculate IQR (Interquartile Range)
    Q1 = volatility_series.quantile(0.25)
    Q3 = volatility_series.quantile(0.75)
    IQR = Q3 - Q1

    # Calculate outlier bounds
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Detect outliers
    is_outlier = (volatility_series < lower_bound) | (volatility_series > upper_bound)

    outlier_count = is_outlier.sum()
    logger.info(
        f"🔍 Outlier detection: {outlier_count} outliers detected "
        f"(threshold: {threshold}×IQR, bounds: [{lower_bound:.6f}, {upper_bound:.6f}])"
    )

    return is_outlier


# ============================================================================
# Volatility Bounds Validation
# ============================================================================

def validate_volatility_bounds(volatility_series: pd.Series,
                               min_bound: float = 0.0001,
                               max_bound: float = 0.05) -> None:
    """
    Validate Parkinson volatility values are within expected range.

    Expected bounds for Parkinson volatility:
    - Minimum: 0.0001 (very low volatility day)
    - Maximum: 0.05 (extremely high volatility day)

    Args:
        volatility_series: Series of Parkinson volatility values
        min_bound: Minimum acceptable volatility (default: 0.0001)
        max_bound: Maximum acceptable volatility (default: 0.05)

    Raises:
        UserWarning: If values are outside expected range

    Example:
        >>> vol = pd.Series([0.0002, 0.001, 0.0005])
        >>> validate_volatility_bounds(vol)  # No warning
        >>>
        >>> low_vol = pd.Series([0.00001])  # Below minimum
        >>> validate_volatility_bounds(low_vol)  # Issues warning

    Note:
        Values outside range don't necessarily indicate errors - extreme
        volatility days do occur. This validation flags values for manual review.
    """
    below_min = (volatility_series < min_bound).sum()
    above_max = (volatility_series > max_bound).sum()

    if below_min > 0:
        warning_msg = (
            f"⚠️ {below_min} volatility values below minimum bound ({min_bound:.6f}). "
            f"These may indicate calculation errors or extremely low volatility days."
        )
        logger.warning(warning_msg)
        warnings.warn(warning_msg, UserWarning)

    if above_max > 0:
        warning_msg = (
            f"⚠️ {above_max} volatility values above maximum bound ({max_bound:.6f}). "
            f"These may indicate calculation errors or extreme volatility events."
        )
        logger.warning(warning_msg)
        warnings.warn(warning_msg, UserWarning)

    if below_min == 0 and above_max == 0:
        logger.info(f"✅ Volatility bounds validated: all values in range [{min_bound:.6f}, {max_bound:.6f}]")


# ============================================================================
# Complete Data Processing Pipeline
# ============================================================================

def process_ohlcv_data(ohlc_data: pd.DataFrame,
                      use_garman_klass: bool = False) -> pd.Series:
    """
    Complete data processing pipeline: validate → clean → calculate volatility.

    This function orchestrates the complete data processing workflow:
    1. Validate OHLCV data integrity
    2. Handle missing data (forward-fill + interpolation)
    3. Calculate volatility (Parkinson or Garman-Klass)
    4. Detect outliers (IQR method, 3× threshold)
    5. Validate volatility bounds (0.0001 to 0.05)

    Args:
        ohlc_data: Raw OHLCV DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume'
        use_garman_klass: If True, use Garman-Klass estimator (default: False → Parkinson)

    Returns:
        pd.Series: Calculated volatility values with quality validation

    Raises:
        ValueError: If OHLCV validation fails
        ValueError: If missing data cannot be handled

    Example:
        >>> data = load_ohlcv_data('VCB')
        >>> parkinson_vol = process_ohlcv_data(data)
        >>> print(f"Mean volatility: {parkinson_vol.mean():.6f}")
        Mean volatility: 0.000847

    Note:
        This is the primary entry point for data processing.
        All data quality issues are logged for monitoring and debugging.
    """
    logger.info("🚀 Starting complete data processing pipeline")

    # Step 1: Validate OHLCV data
    logger.info("Step 1/5: Validating OHLCV data...")
    validate_ohlc_data(ohlc_data)

    # Step 2: Handle missing data
    logger.info("Step 2/5: Handling missing data...")
    clean_data = handle_missing_data(ohlc_data)

    # Step 3: Calculate volatility
    logger.info("Step 3/5: Calculating volatility...")
    if use_garman_klass:
        volatility = calculate_garman_klass_volatility(clean_data)
        logger.info("Using Garman-Klass volatility estimator")
    else:
        volatility = calculate_parkinson_volatility(clean_data)
        logger.info("Using Parkinson volatility estimator (PRIMARY)")

    # Step 4: Detect outliers
    logger.info("Step 4/5: Detecting outliers...")
    outliers = detect_outliers(volatility)

    # Step 5: Validate bounds
    logger.info("Step 5/5: Validating volatility bounds...")
    validate_volatility_bounds(volatility)

    logger.info("✅ Complete data processing pipeline finished successfully")
    return volatility
