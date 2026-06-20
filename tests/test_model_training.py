"""
Test Suite for Model Training Module.

This test suite validates HAR-R baseline model training, temporal splitting,
reproducibility, overfitting detection, and complete training pipeline.

Test Coverage:
    - Random seed setting for reproducibility
    - Temporal train/test split (chronological integrity)
    - HAR-R baseline model training
    - Overfitting detection (train/test performance gap)
    - Model predictions and evaluation metrics
    - Complete training pipeline integration
    - Feature importance/coefficient extraction
    - Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression

from src.model_training import (
    set_random_seeds,
    temporal_train_test_split,
    train_har_r_baseline,
    detect_overfitting,
    get_model_predictions,
    evaluate_model_performance,
    complete_har_r_training_pipeline,
    get_feature_coefficients,
    RANDOM_SEED,
    DEFAULT_TEST_SIZE,
    HAR_FEATURE_COLUMNS,
    TARGET_COLUMN
)
from src.feature_engineering import create_featureset


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_featureset():
    """Create sample featureset for testing (100 days)."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # Create HAR features
    har_daily = np.random.uniform(0.0005, 0.003, 100)
    har_weekly = np.random.uniform(0.0006, 0.0025, 100)
    har_monthly = np.random.uniform(0.0007, 0.002, 100)

    # Create target (shift 5 days)
    target = np.random.uniform(0.0008, 0.0025, 100)
    target[-5:] = np.nan  # Last 5 are NaN (no future data)

    featureset = pd.DataFrame({
        'har_daily_vol': har_daily,
        'har_weekly_vol': har_weekly,
        'har_monthly_vol': har_monthly,
        'target_5d': target
    }, index=dates)

    return featureset


@pytest.fixture
def sample_X_train():
    """Create sample training features."""
    np.random.seed(42)
    return pd.DataFrame({
        'har_daily_vol': np.random.uniform(0.0005, 0.003, 50),
        'har_weekly_vol': np.random.uniform(0.0006, 0.0025, 50),
        'har_monthly_vol': np.random.uniform(0.0007, 0.002, 50)
    })


@pytest.fixture
def sample_y_train():
    """Create sample training targets."""
    np.random.seed(42)
    return pd.Series(np.random.uniform(0.0008, 0.0025, 50))


@pytest.fixture
def sample_X_test():
    """Create sample testing features."""
    np.random.seed(43)
    return pd.DataFrame({
        'har_daily_vol': np.random.uniform(0.0005, 0.003, 20),
        'har_weekly_vol': np.random.uniform(0.0006, 0.0025, 20),
        'har_monthly_vol': np.random.uniform(0.0007, 0.002, 20)
    })


@pytest.fixture
def sample_y_test():
    """Create sample testing targets."""
    np.random.seed(43)
    return pd.Series(np.random.uniform(0.0008, 0.0025, 20))


@pytest.fixture
def trained_model(sample_X_train, sample_y_train):
    """Create trained HAR-R model for testing."""
    return train_har_r_baseline(sample_X_train, sample_y_train)


# ============================================================================
# Random Seed Tests
# ============================================================================

class TestRandomSeeds:
    """Test suite for random seed setting."""

    def test_set_random_seeds_sets_numpy_seed(self):
        """Test setting random seeds makes numpy operations deterministic."""
        set_random_seeds(42)
        result1 = np.random.randint(0, 100, size=5)

        set_random_seeds(42)
        result2 = np.random.randint(0, 100, size=5)

        assert np.array_equal(result1, result2)

    def test_set_random_seeds_default_value(self):
        """Test setting random seeds with default value (42)."""
        set_random_seeds()  # Should use RANDOM_SEED (42)
        result1 = np.random.randint(0, 100, size=5)

        set_random_seeds(RANDOM_SEED)
        result2 = np.random.randint(0, 100, size=5)

        assert np.array_equal(result1, result2)


# ============================================================================
# Temporal Train/Test Split Tests
# ============================================================================

class TestTemporalTrainTestSplit:
    """Test suite for temporal train/test splitting."""

    def test_temporal_split_creates_train_and_test(self, sample_featureset):
        """Test temporal split creates both train and test sets."""
        train, test = temporal_train_test_split(sample_featureset, test_size=0.2)

        assert len(train) > 0
        assert len(test) > 0
        assert len(train) + len(test) <= len(sample_featureset)

    def test_temporal_split_maintains_chronological_order(self, sample_featureset):
        """Test temporal split preserves chronological order."""
        train, test = temporal_train_test_split(sample_featureset, test_size=0.2)

        # Verify: last train date < first test date
        assert train.index[-1] < test.index[0]

    def test_temporal_split_approximate_ratio(self, sample_featureset):
        """Test temporal split creates approximately correct ratio."""
        test_size = 0.2
        train, test = temporal_train_test_split(sample_featureset, test_size=test_size)

        # Calculate actual test ratio (excluding NaN targets)
        total_valid = len(train) + len(test)
        actual_test_ratio = len(test) / total_valid

        # Should be approximately 0.2 (±0.05 tolerance)
        assert abs(actual_test_ratio - test_size) < 0.05

    def test_temporal_split_excludes_nan_targets(self, sample_featureset):
        """Test temporal split excludes rows with NaN targets."""
        train, test = temporal_train_test_split(sample_featureset)

        # Neither train nor test should have NaN targets
        assert train['target_5d'].notna().all()
        assert test['target_5d'].notna().all()

    def test_temporal_split_empty_featureset_raises_error(self):
        """Test temporal split raises error for empty featureset."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            temporal_train_test_split(empty_df)

    def test_temporal_split_invalid_test_size_raises_error(self, sample_featureset):
        """Test temporal split raises error for invalid test_size."""
        with pytest.raises(ValueError, match="test_size must be in"):
            temporal_train_test_split(sample_featureset, test_size=1.5)

        with pytest.raises(ValueError, match="test_size must be in"):
            temporal_train_test_split(sample_featureset, test_size=0)


# ============================================================================
# HAR-R Model Training Tests
# ============================================================================

class TestHARModelTraining:
    """Test suite for HAR-R baseline model training."""

    def test_train_har_r_returns_model(self, sample_X_train, sample_y_train):
        """Test HAR-R training returns LinearRegression model."""
        model = train_har_r_baseline(sample_X_train, sample_y_train)

        assert isinstance(model, LinearRegression)

    def test_train_har_r_model_has_coefficients(self, sample_X_train, sample_y_train):
        """Test HAR-R model has learned coefficients."""
        model = train_har_r_baseline(sample_X_train, sample_y_train)

        assert model.coef_ is not None
        assert len(model.coef_) == len(HAR_FEATURE_COLUMNS)
        assert model.intercept_ is not None

    def test_train_har_r_mismatched_lengths_raises_error(self, sample_X_train, sample_y_train):
        """Test HAR-R training raises error for mismatched X and y lengths."""
        short_y = sample_y_train.iloc[:10]

        with pytest.raises(ValueError, match="different lengths"):
            train_har_r_baseline(sample_X_train, short_y)

    def test_train_har_r_missing_features_raises_error(self, sample_X_train, sample_y_train):
        """Test HAR-R training raises error for missing required features."""
        incomplete_features = sample_X_train[['har_daily_vol', 'har_weekly_vol']]

        with pytest.raises(ValueError, match="Missing required features"):
            train_har_r_baseline(incomplete_features, sample_y_train)


# ============================================================================
# Overfitting Detection Tests
# ============================================================================

class TestOverfittingDetection:
    """Test suite for overfitting detection."""

    def test_detect_overfitting_returns_dict(self, trained_model, sample_X_train, sample_y_train,
                                             sample_X_test, sample_y_test):
        """Test overfitting detection returns results dictionary."""
        results = detect_overfitting(
            trained_model, sample_X_train, sample_X_test, sample_y_train, sample_y_test
        )

        assert isinstance(results, dict)
        assert 'is_overfitting' in results
        assert 'train_mse' in results
        assert 'test_mse' in results
        assert 'overfitting_ratio' in results

    def test_detect_overfitting_calculates_correct_ratio(self, trained_model, sample_X_train, sample_y_train,
                                                        sample_X_test, sample_y_test):
        """Test overfitting detection calculates correct test/train MSE ratio."""
        results = detect_overfitting(
            trained_model, sample_X_train, sample_X_test, sample_y_train, sample_y_test
        )

        expected_ratio = results['test_mse'] / results['train_mse']
        assert results['overfitting_ratio'] == expected_ratio

    def test_detect_overfitting_detects_severe_overfitting(self):
        """Test overfitting detection identifies severe overfitting."""
        # Create perfect fit on train, poor on test
        X_train = pd.DataFrame({
            'har_daily_vol': [1, 2, 3],
            'har_weekly_vol': [1, 2, 3],
            'har_monthly_vol': [1, 2, 3]
        })
        y_train = pd.Series([1, 2, 3])  # Perfect linear relationship

        X_test = pd.DataFrame({
            'har_daily_vol': [4, 5],
            'har_weekly_vol': [4, 5],
            'har_monthly_vol': [4, 5]
        })
        y_test = pd.Series([100, 200])  # Very different (simulating overfitting)

        model = train_har_r_baseline(X_train, y_train)

        results = detect_overfitting(
            model, X_train, X_test, y_train, y_test, threshold=1.01
        )

        # Should detect overfitting (ratio >> 1.2)
        assert results['is_overfitting'] is True
        assert results['overfitting_ratio'] > 1.2


# ============================================================================
# Model Predictions Tests
# ============================================================================

class TestModelPredictions:
    """Test suite for model predictions."""

    def test_get_model_predictions_returns_array(self, trained_model, sample_X_test):
        """Test model predictions returns numpy array."""
        predictions = get_model_predictions(trained_model, sample_X_test)

        assert isinstance(predictions, np.ndarray)

    def test_get_model_predictions_correct_length(self, trained_model, sample_X_test):
        """Test model predictions has correct length."""
        predictions = get_model_predictions(trained_model, sample_X_test)

        assert len(predictions) == len(sample_X_test)

    def test_get_model_predictions_reasonable_values(self, trained_model, sample_X_test):
        """Test model predictions are in reasonable range."""
        predictions = get_model_predictions(trained_model, sample_X_test)

        # Predictions should be positive (volatility)
        assert (predictions >= 0).all()
        # Predictions should be in reasonable volatility range
        assert (predictions < 0.01).all()


# ============================================================================
# Model Evaluation Tests
# ============================================================================

class TestModelEvaluation:
    """Test suite for model performance evaluation."""

    def test_evaluate_performance_returns_dict(self):
        """Test performance evaluation returns metrics dictionary."""
        y_true = pd.Series([0.001, 0.002, 0.0015, 0.0018])
        y_pred = np.array([0.0011, 0.0019, 0.0016, 0.0017])

        metrics = evaluate_model_performance(y_true, y_pred)

        assert isinstance(metrics, dict)
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics

    def test_evaluate_performance_calculates_correct_rmse(self):
        """Test RMSE calculation is correct."""
        y_true = pd.Series([2.0, 4.0, 6.0])
        y_pred = np.array([2.5, 3.5, 6.5])

        metrics = evaluate_model_performance(y_true, y_pred)

        # Expected RMSE: sqrt(((0.5)^2 + (-0.5)^2 + (0.5)^2) / 3) = sqrt(0.75/3) = sqrt(0.25) = 0.5
        expected_rmse = 0.5
        assert abs(metrics['rmse'] - expected_rmse) < 1e-6

    def test_evaluate_performance_r2_perfect_fit(self):
        """Test R² = 1.0 for perfect fit."""
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        metrics = evaluate_model_performance(y_true, y_pred)

        assert abs(metrics['r2'] - 1.0) < 1e-6


# ============================================================================
# Feature Coefficients Tests
# ============================================================================

class TestFeatureCoefficients:
    """Test suite for feature coefficient extraction."""

    def test_get_coefficients_returns_dataframe(self, trained_model):
        """Test coefficient extraction returns DataFrame."""
        coefficients = get_feature_coefficients(trained_model)

        assert isinstance(coefficients, pd.DataFrame)

    def test_get_coefficients_has_correct_columns(self, trained_model):
        """Test coefficient DataFrame has correct columns."""
        coefficients = get_feature_coefficients(trained_model)

        assert 'feature' in coefficients.columns
        assert 'coefficient' in coefficients.columns
        assert 'abs_coefficient' in coefficients.columns

    def test_get_coefficients_sorted_by_absolute(self, trained_model):
        """Test coefficients are sorted by absolute value (descending)."""
        coefficients = get_feature_coefficients(trained_model)

        # Check monotonic decreasing
        assert coefficients['abs_coefficient'].is_monotonic_decreasing


# ============================================================================
# Complete Training Pipeline Tests
# ============================================================================

class TestCompleteTrainingPipeline:
    """Test suite for complete training pipeline integration."""

    def test_complete_pipeline_returns_dict(self, sample_featureset):
        """Test complete pipeline returns results dictionary."""
        results = complete_har_r_training_pipeline(sample_featureset)

        assert isinstance(results, dict)
        assert 'model' in results
        assert 'X_train' in results
        assert 'X_test' in results
        assert 'y_train' in results
        assert 'y_test' in results

    def test_complete_pipeline_includes_metrics(self, sample_featureset):
        """Test complete pipeline includes performance metrics."""
        results = complete_har_r_training_pipeline(sample_featureset)

        assert 'train_metrics' in results
        assert 'test_metrics' in results
        assert 'overfitting_results' in results

    def test_complete_pipeline_metrics_are_reasonable(self, sample_featureset):
        """Test complete pipeline produces reasonable metrics."""
        results = complete_har_r_training_pipeline(sample_featureset)

        # RMSE should be positive
        assert results['test_metrics']['rmse'] > 0
        # R² should be between -∞ and 1 (typically negative for noisy data)
        assert results['test_metrics']['r2'] <= 1.0

    def test_complete_pipeline_no_overfitting_warning(self, sample_featureset):
        """Test complete pipeline does not show overfitting for linear regression."""
        results = complete_har_r_training_pipeline(sample_featureset)

        # Linear regression typically has low overfitting
        # Check that overfitting is not detected
        assert results['overfitting_results']['is_overfitting'] is False

    def test_complete_pipeline_missing_columns_raises_error(self):
        """Test complete pipeline raises error for missing required columns."""
        incomplete_featureset = pd.DataFrame({
            'har_daily_vol': [0.001, 0.002],
            'target_5d': [0.0025, 0.0022]
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            complete_har_r_training_pipeline(incomplete_featureset)


# ============================================================================
# Integration Tests
# ============================================================================

class TestModelTrainingIntegration:
    """Integration tests for complete model training workflow."""

    def test_end_to_end_training_pipeline(self):
        """Test complete end-to-end training from featureset to trained model."""
        # Create sample volatility data
        np.random.seed(42)
        vol_data = pd.Series(
            np.random.uniform(0.0005, 0.003, size=60),
            index=pd.date_range('2024-01-01', periods=60, freq='D')
        )

        # Create featureset
        featureset = create_featureset(vol_data)

        # Train model
        results = complete_har_r_training_pipeline(featureset, test_size=0.2)

        # Verify results
        assert results['model'] is not None
        assert len(results['X_train']) > 0
        assert len(results['X_test']) > 0
        assert results['test_metrics']['rmse'] > 0

    def test_reproducibility_same_seed_same_model(self, sample_featureset):
        """Test same random seed produces identical results."""
        # First training
        set_random_seeds(42)
        results1 = complete_har_r_training_pipeline(sample_featureset)

        # Second training (same seed)
        set_random_seeds(42)
        results2 = complete_har_r_training_pipeline(sample_featureset)

        # Models should be identical
        assert np.array_equal(results1['model'].coef_, results2['model'].coef_)
        assert results1['model'].intercept_ == results2['model'].intercept_
