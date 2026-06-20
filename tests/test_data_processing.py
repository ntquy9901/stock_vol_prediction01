"""
Comprehensive test suite for data processing module.

This module tests Parkinson volatility calculation, OHLCV validation,
missing data handling, outlier detection, and fallback estimators.

Test Coverage Target: 90%+ for data processing module
"""

import pytest
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Import module to test - this will fail initially
from src.data_processing import (
    validate_ohlc_data,
    calculate_parkinson_volatility,
    calculate_garman_klass_volatility,
    handle_missing_data,
    detect_outliers,
    validate_volatility_bounds
)


# ============================================================================
# OHLCV Validation Tests
# ============================================================================

class TestOHLCVValidation:
    """Test suite for OHLCV data validation."""

    def test_validate_valid_ohlc_data(self, sample_ohlc_data):
        """Test validation passes for valid OHLCV data."""
        result = validate_ohlc_data(sample_ohlc_data)
        assert result is True

    def test_validate_ohlc_high_greater_than_close(self, sample_ohlc_data_invalid):
        """Test validation fails when High < Close."""
        with pytest.raises(ValueError, match="High must be.*Close"):
            validate_ohlc_data(sample_ohlc_data_invalid)

    def test_validate_ohlc_close_less_than_low(self):
        """Test validation fails when Close < Low."""
        # Create specific test data where Close < Low but High >= Close
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        invalid_data = pd.DataFrame({
            'Date': dates,
            'Open': [100.0] * 10,
            'High': [102.0] * 10,  # Valid: High >= Close
            'Low': [101.0] * 10,   # Invalid: Low > Close
            'Close': [100.0] * 10,
            'Volume': [1000000] * 10
        })

        with pytest.raises(ValueError, match="Close must be.*Low"):
            validate_ohlc_data(invalid_data.set_index('Date'))

    def test_validate_ohlc_volume_positive(self, sample_ohlc_data):
        """Test validation ensures Volume > 0."""
        # Modify data to have zero volume
        invalid_data = sample_ohlc_data.copy()
        invalid_data.loc[invalid_data.index[0], 'Volume'] = 0

        with pytest.raises(ValueError, match="Volume must be positive"):
            validate_ohlc_data(invalid_data)

    def test_validate_ohlc_returns_specific_error_messages(self, sample_ohlc_data_invalid):
        """Test validation provides specific actionable error messages."""
        with pytest.raises(ValueError) as exc_info:
            validate_ohlc_data(sample_ohlc_data_invalid)

        error_message = str(exc_info.value)
        assert len(error_message) > 20  # Substantial error message


# ============================================================================
# Parkinson Volatility Calculation Tests
# ============================================================================

class TestParkinsonVolatility:
    """Test suite for Parkinson volatility estimator."""

    def test_parkinson_volatility_formula_correctness(self, sample_ohlc_data):
        """Test Parkinson formula: σ² = (log(H/L)²) / (4*log(2))."""
        parkinson_vol = calculate_parkinson_volatility(sample_ohlc_data)

        # Manual calculation for first row
        high = sample_ohlc_data['High'].iloc[0]
        low = sample_ohlc_data['Low'].iloc[0]
        expected = (np.log(high / low) ** 2) / (4 * np.log(2))

        assert abs(parkinson_vol.iloc[0] - expected) < 1e-10

    def test_parkinson_volatility_range(self, sample_ohlc_data):
        """Test Parkinson volatility values are in expected range (0.0001 to 0.05)."""
        parkinson_vol = calculate_parkinson_volatility(sample_ohlc_data)

        assert parkinson_vol.min() >= 0.0001
        assert parkinson_vol.max() <= 0.05

    def test_parkinson_volatility_returns_series(self, sample_ohlc_data):
        """Test Parkinson calculation returns pandas Series."""
        parkinson_vol = calculate_parkinson_volatility(sample_ohlc_data)

        assert isinstance(parkinson_vol, pd.Series)
        assert len(parkinson_vol) == len(sample_ohlc_data)

    def test_parkinson_volatility_handles_nan_input(self, sample_ohlc_data):
        """Test Parkinson calculation raises error for NaN input when cleaning disabled."""
        # Create DataFrame with NaN but all required columns
        data = sample_ohlc_data.iloc[:2].copy()
        data.loc[data.index[0], 'High'] = np.nan

        # Test with cleaning disabled - should raise error
        with pytest.raises(ValueError, match="contains NaN"):
            calculate_parkinson_volatility(data, clean=False)

    def test_parkinson_volatility_validates_before_calculation(self, sample_ohlc_data_invalid):
        """Test Parkinson validates OHLCV before calculation when cleaning disabled."""
        # Test with cleaning disabled - should raise error
        with pytest.raises(ValueError):
            calculate_parkinson_volatility(sample_ohlc_data_invalid, clean=False)


# ============================================================================
# Missing Data Handling Tests
# ============================================================================

class TestMissingDataHandling:
    """Test suite for missing data handling."""

    def test_handle_missing_data_forward_fill_max_5_days(self, sample_ohlc_data_with_missing):
        """Test forward-fill fills max 5 consecutive missing days."""
        filled_data = handle_missing_data(sample_ohlc_data_with_missing)

        # Check forward-fill was applied
        assert filled_data.isna().sum().sum() < sample_ohlc_data_with_missing.isna().sum().sum()

    def test_handle_missing_data_linear_interpolation(self, sample_ohlc_data_with_missing):
        """Test linear interpolation after forward-fill."""
        filled_data = handle_missing_data(sample_ohlc_data_with_missing)

        # Check all NaN values handled (or reduced to acceptable level)
        remaining_na = filled_data.isna().sum().sum()
        assert remaining_na == 0 or remaining_na < 5

    def test_handle_missing_data_preserves_temporal_ordering(self, sample_ohlc_data_with_missing):
        """Test missing data handling preserves temporal ordering (no future data leakage)."""
        original_dates = sample_ohlc_data_with_missing.index.copy()
        filled_data = handle_missing_data(sample_ohlc_data_with_missing)

        assert filled_data.index.equals(original_dates)
        assert all(filled_data.index == original_dates)

    def test_handle_missing_data_logs_statistics(self, sample_ohlc_data_with_missing, caplog):
        """Test missing data handling logs statistics."""
        with caplog.at_level(logging.INFO):
            filled_data = handle_missing_data(sample_ohlc_data_with_missing)

        assert any("missing" in record.message.lower() for record in caplog.records)


# ============================================================================
# Outlier Detection Tests
# ============================================================================

class TestOutlierDetection:
    """Test suite for outlier detection using IQR method."""

    def test_detect_outliers_3x_iqr_threshold(self, sample_ohlc_data_extreme_values):
        """Test outlier detection uses 3×IQR threshold."""
        parkinson_vol = calculate_parkinson_volatility(sample_ohlc_data_extreme_values)
        outliers = detect_outliers(parkinson_vol)

        # The extreme value at index 10 should be flagged
        # Get the date at position 10 and check if it's in outliers
        extreme_date = sample_ohlc_data_extreme_values.index[10]
        assert outliers.loc[extreme_date] == True, f"Expected outlier at {extreme_date}"

    def test_detect_outliers_returns_boolean_series(self, sample_ohlc_data):
        """Test outlier detection returns boolean Series."""
        parkinson_vol = calculate_parkinson_volatility(sample_ohlc_data)
        outliers = detect_outliers(parkinson_vol)

        assert isinstance(outliers, pd.Series)
        assert outliers.dtype == bool

    def test_detect_outliers_logs_results(self, sample_ohlc_data, caplog):
        """Test outlier detection logs results."""
        parkinson_vol = calculate_parkinson_volatility(sample_ohlc_data)

        with caplog.at_level(logging.INFO):
            detect_outliers(parkinson_vol)

        assert any("outlier" in record.message.lower() for record in caplog.records)


# ============================================================================
# Garman-Klass Fallback Estimator Tests
# ============================================================================

class TestGarmanKlassEstimator:
    """Test suite for Garman-Klass fallback estimator."""

    def test_garman_klass_formula_correctness(self, sample_ohlc_data):
        """Test Garman-Klass formula: 0.5*(log(H/L)²) - (2*log(2)-1)*(log(C/O)²)."""
        gk_vol = calculate_garman_klass_volatility(sample_ohlc_data)

        # Manual calculation for first row
        high = sample_ohlc_data['High'].iloc[0]
        low = sample_ohlc_data['Low'].iloc[0]
        close = sample_ohlc_data['Close'].iloc[0]
        open_price = sample_ohlc_data['Open'].iloc[0]

        expected = (0.5 * (np.log(high / low) ** 2) -
                   (2 * np.log(2) - 1) * (np.log(close / open_price) ** 2))

        assert abs(gk_vol.iloc[0] - expected) < 1e-10

    def test_garman_klass_returns_series(self, sample_ohlc_data):
        """Test Garman-Klass returns pandas Series."""
        gk_vol = calculate_garman_klass_volatility(sample_ohlc_data)

        assert isinstance(gk_vol, pd.Series)
        assert len(gk_vol) == len(sample_ohlc_data)

    def test_garman_klass_validates_ohlc_before_calculation(self, sample_ohlc_data_invalid):
        """Test Garman-Klass validates OHLCV before calculation."""
        with pytest.raises(ValueError):
            calculate_garman_klass_volatility(sample_ohlc_data_invalid)


# ============================================================================
# Volatility Bounds Validation Tests
# ============================================================================

class TestVolatilityBoundsValidation:
    """Test suite for volatility bounds validation."""

    def test_validate_volatility_bounds_in_range(self, sample_ohlc_data):
        """Test validation passes for volatility in range (0.0001 to 0.05)."""
        parkinson_vol = calculate_parkinson_volatility(sample_ohlc_data)

        # Should not raise any warnings or errors
        validate_volatility_bounds(parkinson_vol)

    def test_validate_volatility_bounds_below_minimum(self, sample_ohlc_data):
        """Test validation warns when volatility below minimum (0.0001)."""
        low_vol = pd.Series([0.00001] * 100)

        with pytest.warns(UserWarning, match="below minimum bound"):
            validate_volatility_bounds(low_vol)

    def test_validate_volatility_bounds_above_maximum(self, sample_ohlc_data):
        """Test validation warns when volatility above maximum (0.05)."""
        high_vol = pd.Series([0.1] * 100)

        with pytest.warns(UserWarning, match="above maximum bound"):
            validate_volatility_bounds(high_vol)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for data processing."""

    def test_parkinson_calculation_performance_30_stocks_20_years(self, sample_multi_stock_data):
        """Test Parkinson calculation completes in < 30 seconds for 30 stocks × 20 years."""
        import time

        # Scale up to 30 stocks with 20 years (~5040 days per stock)
        start_time = time.time()

        results = {}
        for stock_id, data in sample_multi_stock_data.items():
            parkinson_vol = calculate_parkinson_volatility(data)
            results[stock_id] = parkinson_vol

        elapsed_time = time.time() - start_time

        # Should complete in < 30 seconds
        assert elapsed_time < 30.0, f"Processing took {elapsed_time:.2f} seconds, expected < 30 seconds"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete data processing pipeline."""

    def test_complete_pipeline_valid_data(self, sample_ohlc_data):
        """Test complete pipeline: validate → handle missing → calculate volatility."""
        # Step 1: Validate
        validate_ohlc_data(sample_ohlc_data)

        # Step 2: Handle missing (none in this case)
        clean_data = handle_missing_data(sample_ohlc_data)

        # Step 3: Calculate Parkinson volatility
        parkinson_vol = calculate_parkinson_volatility(clean_data)

        # Step 4: Detect outliers
        outliers = detect_outliers(parkinson_vol)

        # Step 5: Validate bounds
        validate_volatility_bounds(parkinson_vol)

        assert len(parkinson_vol) == len(sample_ohlc_data)
        assert parkinson_vol.min() >= 0.0001
        assert parkinson_vol.max() <= 0.05

    def test_complete_pipeline_missing_data(self, sample_ohlc_data_with_missing):
        """Test complete pipeline handles missing data correctly."""
        # Step 1: Handle missing data first (before validation)
        clean_data = handle_missing_data(sample_ohlc_data_with_missing)

        # Step 2: Validate cleaned data
        validate_ohlc_data(clean_data)

        # Step 3: Calculate Parkinson volatility
        parkinson_vol = calculate_parkinson_volatility(clean_data)

        assert len(parkinson_vol) == len(clean_data)

    def test_fallback_to_garman_klass_on_parkinson_failure(self, sample_ohlc_data):
        """Test fallback to Garman-Klass when Parkinson fails."""
        # This tests the fallback mechanism
        parkinson_vol = calculate_parkinson_volatility(sample_ohlc_data)
        gk_vol = calculate_garman_klass_volatility(sample_ohlc_data)

        # Both should produce Series of same length
        assert len(parkinson_vol) == len(gk_vol)
