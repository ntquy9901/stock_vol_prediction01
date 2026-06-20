"""
Test Suite for Feature Engineering Module.

This test suite validates HAR feature creation, target variable generation,
and feature quality validation for volatility forecasting.

Test Coverage:
    - HAR feature creation (daily, weekly, monthly windows)
    - 5-day target variable creation
    - Feature quality validation (NaN, infinite, correlations)
    - Complete feature set integration
    - Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
import warnings

from src.feature_engineering import (
    create_har_features,
    create_5day_target,
    validate_har_features,
    create_featureset,
    get_har_feature_importance,
    HAR_WINDOWS,
    FORECAST_HORIZON_DAYS
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_volatility_series():
    """Create sample volatility series for testing (30 days)."""
    np.random.seed(42)
    return pd.Series(
        np.random.uniform(0.0005, 0.003, size=30),
        index=pd.date_range('2024-01-01', periods=30, freq='D')
    )


@pytest.fixture
def sample_volatility_long():
    """Create longer volatility series for testing (100 days)."""
    np.random.seed(42)
    return pd.Series(
        np.random.uniform(0.0005, 0.003, size=100),
        index=pd.date_range('2024-01-01', periods=100, freq='D')
    )


# ============================================================================
# HAR Feature Creation Tests
# ============================================================================

class TestHARFeatureCreation:
    """Test suite for HAR feature creation."""

    def test_har_features_creates_three_features(self, sample_volatility_long):
        """Test HAR feature creation produces exactly 3 features."""
        har_features = create_har_features(sample_volatility_long)

        assert len(har_features.columns) == 3
        assert 'har_daily_vol' in har_features.columns
        assert 'har_weekly_vol' in har_features.columns
        assert 'har_monthly_vol' in har_features.columns

    def test_har_daily_feature_uses_1day_window(self, sample_volatility_long):
        """Test HAR daily feature uses 1-day rolling window."""
        har_features = create_har_features(sample_volatility_long)

        # First value should equal original volatility (1-day window)
        assert har_features['har_daily_vol'].iloc[0] == sample_volatility_long.iloc[0]

    def test_har_weekly_feature_uses_5day_window(self, sample_volatility_long):
        """Test HAR weekly feature uses 5-day rolling window."""
        har_features = create_har_features(sample_volatility_long)

        # First 4 values should be NaN (5-day window warm-up)
        assert har_features['har_weekly_vol'].iloc[:4].isna().all()
        # 5th value should be mean of first 5 days
        expected_weekly = sample_volatility_long.iloc[:5].mean()
        assert har_features['har_weekly_vol'].iloc[4] == expected_weekly

    def test_har_monthly_feature_uses_22day_window(self, sample_volatility_long):
        """Test HAR monthly feature uses 22-day rolling window (confirmed)."""
        har_features = create_har_features(sample_volatility_long)

        # First 21 values should be NaN (22-day window warm-up)
        assert har_features['har_monthly_vol'].iloc[:21].isna().all()
        # 22nd value should be mean of first 22 days
        expected_monthly = sample_volatility_long.iloc[:22].mean()
        assert har_features['har_monthly_vol'].iloc[21] == expected_monthly

    def test_har_features_preserves_index(self, sample_volatility_long):
        """Test HAR features preserve original volatility series index."""
        har_features = create_har_features(sample_volatility_long)

        assert har_features.index.equals(sample_volatility_long.index)

    def test_har_features_returns_dataframe(self, sample_volatility_long):
        """Test HAR features returns pandas DataFrame."""
        har_features = create_har_features(sample_volatility_long)

        assert isinstance(har_features, pd.DataFrame)

    def test_har_features_custom_windows(self, sample_volatility_long):
        """Test HAR features with custom window sizes."""
        custom_windows = {'daily': 2, 'weekly': 7, 'monthly': 30}
        har_features = create_har_features(sample_volatility_long, windows=custom_windows)

        # Verify custom windows are used (2-day window means first 2 values are NaN)
        assert har_features['har_daily_vol'].iloc[:1].isna().all()
        assert har_features['har_weekly_vol'].iloc[:6].isna().all()

    def test_har_features_empty_series_raises_error(self):
        """Test HAR features raises error for empty series."""
        empty_series = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="empty"):
            create_har_features(empty_series)


# ============================================================================
# Target Variable Creation Tests
# ============================================================================

class TestTargetVariableCreation:
    """Test suite for 5-day target variable creation."""

    def test_5day_target_shifts_correctly(self, sample_volatility_long):
        """Test 5-day target shifts volatility series by 5 days."""
        target = create_5day_target(sample_volatility_long, horizon_days=5)

        # target[t] should equal volatility[t+5]
        assert target.iloc[0] == sample_volatility_long.iloc[5]
        assert target.iloc[10] == sample_volatility_long.iloc[15]

    def test_5day_target_last_5_values_are_nan(self, sample_volatility_long):
        """Test 5-day target has NaN for last 5 values (no future data)."""
        target = create_5day_target(sample_volatility_long, horizon_days=5)

        # Last 5 values should be NaN
        assert target.iloc[-5:].isna().all()

    def test_5day_target_preserves_index(self, sample_volatility_long):
        """Test 5-day target preserves original volatility series index."""
        target = create_5day_target(sample_volatility_long, horizon_days=5)

        assert target.index.equals(sample_volatility_long.index)

    def test_5day_target_default_horizon(self, sample_volatility_long):
        """Test 5-day target uses default 5-day horizon."""
        target = create_5day_target(sample_volatility_long)

        # Should use default horizon of 5 days
        assert target.iloc[0] == sample_volatility_long.iloc[FORECAST_HORIZON_DAYS]

    def test_5day_target_custom_horizon(self, sample_volatility_long):
        """Test 5-day target with custom horizon."""
        target = create_5day_target(sample_volatility_long, horizon_days=10)

        # target[t] should equal volatility[t+10]
        assert target.iloc[0] == sample_volatility_long.iloc[10]
        assert target.iloc[-10:].isna().all()

    def test_5day_target_negative_horizon_raises_error(self, sample_volatility_long):
        """Test 5-day target raises error for negative horizon."""
        with pytest.raises(ValueError, match="horizon_days must be positive"):
            create_5day_target(sample_volatility_long, horizon_days=-5)

    def test_5day_target_zero_horizon_raises_error(self, sample_volatility_long):
        """Test 5-day target raises error for zero horizon."""
        with pytest.raises(ValueError, match="horizon_days must be positive"):
            create_5day_target(sample_volatility_long, horizon_days=0)

    def test_5day_target_empty_series_raises_error(self):
        """Test 5-day target raises error for empty series."""
        empty_series = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="empty"):
            create_5day_target(empty_series)


# ============================================================================
# Feature Validation Tests
# ============================================================================

class TestFeatureValidation:
    """Test suite for HAR feature quality validation."""

    def test_validate_har_features_valid_data_passes(self, sample_volatility_long):
        """Test validation passes for valid HAR features."""
        har_features = create_har_features(sample_volatility_long)

        assert validate_har_features(har_features) is True

    def test_validate_har_features_infinite_values_raises_error(self, sample_volatility_long):
        """Test validation raises error for infinite values."""
        har_features = create_har_features(sample_volatility_long)
        har_features.iloc[0, 0] = np.inf

        with pytest.raises(ValueError, match="infinite"):
            validate_har_features(har_features)

    def test_validate_har_features_insufficient_samples_raises_error(self, sample_volatility_series):
        """Test validation raises error for insufficient samples."""
        har_features = create_har_features(sample_volatility_series)

        # Only 30 samples, but monthly window requires 22+ samples
        # This should pass with min_required_samples=30, but fail with higher
        with pytest.raises(ValueError, match="Insufficient"):
            validate_har_features(har_features, min_required_samples=100)

    def test_validate_har_features_warns_high_correlation(self, sample_volatility_long):
        """Test validation warns about high feature correlation (>0.99)."""
        # Create features with perfect correlation (50 samples to pass min_required_samples)
        har_features = pd.DataFrame({
            'feature1': range(1, 51),
            'feature2': [2*x for x in range(1, 51)],  # Perfect correlation (2x)
            'feature3': [x+0.5 for x in range(1, 51)]
        })

        with pytest.warns(UserWarning, match="High feature correlation"):
            validate_har_features(har_features)


# ============================================================================
# Complete Feature Set Tests
# ============================================================================

class TestCompleteFeatureSet:
    """Test suite for complete feature set creation."""

    def test_featureset_creates_four_columns(self, sample_volatility_long):
        """Test complete featureset creates 3 features + 1 target."""
        featureset = create_featureset(sample_volatility_long)

        assert len(featureset.columns) == 4
        assert 'har_daily_vol' in featureset.columns
        assert 'har_weekly_vol' in featureset.columns
        assert 'har_monthly_vol' in featureset.columns
        assert 'target_5d' in featureset.columns

    def test_featureset_target_column_name_uses_horizon(self, sample_volatility_long):
        """Test featureset target column name uses horizon parameter."""
        featureset = create_featureset(sample_volatility_long, horizon_days=10)

        assert 'target_10d' in featureset.columns

    def test_featureset_preserves_index(self, sample_volatility_long):
        """Test featureset preserves original volatility series index."""
        featureset = create_featureset(sample_volatility_long)

        assert featureset.index.equals(sample_volatility_long.index)

    def test_featureset_empty_series_raises_error(self):
        """Test featureset raises error for empty series."""
        empty_series = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="empty"):
            create_featureset(empty_series)


# ============================================================================
# Feature Importance Tests
# ============================================================================

class TestFeatureImportance:
    """Test suite for feature importance extraction."""

    def test_feature_importance_returns_dataframe(self):
        """Test feature importance returns pandas DataFrame."""
        coef = np.array([0.5, 0.3, 0.2])
        importance = get_har_feature_importance(coef)

        assert isinstance(importance, pd.DataFrame)

    def test_feature_importance_has_correct_columns(self):
        """Test feature importance has correct columns."""
        coef = np.array([0.5, 0.3, 0.2])
        importance = get_har_feature_importance(coef)

        assert 'feature' in importance.columns
        assert 'coefficient' in importance.columns
        assert 'abs_coefficient' in importance.columns

    def test_feature_importance_sorts_by_absolute_coefficient(self):
        """Test feature importance sorts by absolute coefficient (descending)."""
        coef = np.array([0.2, 0.5, 0.3])  # Not sorted
        importance = get_har_feature_importance(coef)

        # Should be sorted: 0.5, 0.3, 0.2
        assert importance['abs_coefficient'].is_monotonic_decreasing

    def test_feature_importance_custom_feature_names(self):
        """Test feature importance with custom feature names."""
        coef = np.array([0.5, 0.3])
        custom_names = ['feature_a', 'feature_b']
        importance = get_har_feature_importance(coef, feature_names=custom_names)

        assert importance['feature'].tolist() == custom_names


# ============================================================================
# Integration Tests
# ============================================================================

class TestFeatureEngineeringIntegration:
    """Integration tests for complete feature engineering workflow."""

    def test_end_to_end_feature_engineering(self, sample_volatility_long):
        """Test complete end-to-end feature engineering pipeline."""
        # Step 1: Create HAR features
        har_features = create_har_features(sample_volatility_long)
        assert len(har_features.columns) == 3

        # Step 2: Create target
        target = create_5day_target(sample_volatility_long)
        assert target.iloc[0] == sample_volatility_long.iloc[5]

        # Step 3: Validate features
        is_valid = validate_har_features(har_features)
        assert is_valid is True

        # Step 4: Combine into featureset
        featureset = create_featureset(sample_volatility_long)
        assert len(featureset.columns) == 4

    def test_feature_engineering_with_realistic_volatility(self):
        """Test feature engineering with realistic volatility range."""
        # Create realistic Parkinson volatility (0.0001 to 0.05 range)
        # Need 60 samples to have sufficient data after 22-day warm-up
        np.random.seed(42)
        realistic_vol = pd.Series(
            np.random.uniform(0.0002, 0.004, size=60),
            index=pd.date_range('2024-01-01', periods=60, freq='D')
        )

        featureset = create_featureset(realistic_vol)

        # Verify features are in reasonable range
        assert featureset['har_daily_vol'].min() >= 0
        assert featureset['har_daily_vol'].max() < 0.01
        assert featureset['target_5d'].notna().sum() > 0
