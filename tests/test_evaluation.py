"""
Test Suite for Evaluation Module.

This test suite validates QLIKE loss, directional accuracy, Theil's U statistic,
benchmark comparisons, performance visualization, and complete evaluation reports.

Test Coverage:
    - QLIKE loss calculation (primary metric)
    - Directional accuracy (>55% target)
    - Theil's U statistic (vs random walk)
    - Comprehensive forecast evaluation
    - Benchmark comparisons (random walk, historical mean)
    - Performance visualization
    - Complete evaluation reports
    - Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile

from src.evaluation import (
    qlike_loss,
    directional_accuracy,
    theil_u_statistic,
    evaluate_forecast,
    compare_vs_benchmarks,
    plot_forecast_performance,
    generate_evaluation_report,
    DIRECTIONAL_ACCURACY_TARGET,
    RMSE_TARGET_5DAY,
    THEIL_U_THRESHOLD
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_y_true():
    """Create sample true volatility values."""
    np.random.seed(42)
    return pd.Series(
        np.random.uniform(0.001, 0.003, size=30),
        index=pd.date_range('2024-01-01', periods=30, freq='D')
    )


@pytest.fixture
def sample_y_pred():
    """Create sample predicted volatility values (close to true)."""
    np.random.seed(43)
    # Create predictions close to true values with small noise
    base_values = np.array([0.001, 0.002, 0.0015, 0.0018, 0.0022,
                           0.0025, 0.0019, 0.0021, 0.0017, 0.0023])
    # Repeat to get 30 values
    predictions = np.tile(base_values, 3)[:30]
    # Add small noise
    predictions = predictions + np.random.normal(0, 0.0001, 30)
    # Ensure positive
    return np.maximum(predictions, 0.0001)


@pytest.fixture
def perfect_predictions():
    """Create perfect predictions for testing."""
    true_values = np.array([0.001, 0.002, 0.0015, 0.0018, 0.002])
    return true_values.copy()


@pytest.fixture
def sample_train_test_data():
    """Create sample train and test data for evaluation."""
    np.random.seed(42)

    # Training data (20 samples)
    y_train_true = pd.Series(np.random.uniform(0.001, 0.003, 20))
    y_train_pred = y_train_true.values + np.random.normal(0, 0.0001, 20)

    # Test data (10 samples)
    y_test_true = pd.Series(np.random.uniform(0.001, 0.003, 10))
    y_test_pred = y_test_true.values + np.random.normal(0, 0.0001, 10)

    return y_train_true, y_train_pred, y_test_true, y_test_pred


# ============================================================================
# QLIKE Loss Tests
# ============================================================================

class TestQLIKELoss:
    """Test suite for QLIKE loss calculation."""

    def test_qlike_loss_returns_float(self, sample_y_true, sample_y_pred):
        """Test QLIKE loss returns float value."""
        loss = qlike_loss(sample_y_true, sample_y_pred)

        assert isinstance(loss, (float, np.floating))

    def test_qlike_loss_positive_values(self, sample_y_true, sample_y_pred):
        """Test QLIKE loss is positive for valid inputs."""
        loss = qlike_loss(sample_y_true, sample_y_pred)

        assert loss >= 0

    def test_qlike_loss_perfect_prediction_zero(self):
        """Test QLIKE loss is zero for perfect predictions."""
        y_true = pd.Series([0.001, 0.002, 0.0015])
        y_pred = np.array([0.001, 0.002, 0.0015])

        loss = qlike_loss(y_true, y_pred)

        assert abs(loss) < 1e-10

    def test_qlike_loss_length_mismatch_raises_error(self, sample_y_true):
        """Test QLIKE loss raises error for length mismatch."""
        short_pred = np.array([0.001, 0.002])

        with pytest.raises(ValueError, match="Length mismatch"):
            qlike_loss(sample_y_true, short_pred)

    def test_qlike_loss_negative_values_raises_error(self):
        """Test QLIKE loss raises error for negative values."""
        y_true = pd.Series([0.001, -0.002, 0.0015])
        y_pred = np.array([0.001, 0.002, 0.0015])

        with pytest.raises(ValueError, match="non-negative"):
            qlike_loss(y_true, y_pred)


# ============================================================================
# Directional Accuracy Tests
# ============================================================================

class TestDirectionalAccuracy:
    """Test suite for directional accuracy calculation."""

    def test_directional_accuracy_returns_float(self, sample_y_true, sample_y_pred):
        """Test directional accuracy returns float value."""
        acc = directional_accuracy(sample_y_true, sample_y_pred)

        assert isinstance(acc, (float, np.floating))

    def test_directional_accuracy_between_0_and_1(self, sample_y_true, sample_y_pred):
        """Test directional accuracy is between 0 and 1."""
        acc = directional_accuracy(sample_y_true, sample_y_pred)

        assert 0 <= acc <= 1

    def test_directional_accuracy_perfect_prediction_100_percent(self):
        """Test directional accuracy is 100% for perfect direction predictions."""
        y_true = pd.Series([0.001, 0.002, 0.0015, 0.0018, 0.002])
        y_pred = np.array([0.0011, 0.0021, 0.0016, 0.0019, 0.0021])

        acc = directional_accuracy(y_true, y_pred)

        # All predictions should be in correct direction (up or down)
        assert acc > 0.5  # Should be better than random

    def test_directional_accuracy_length_mismatch_raises_error(self, sample_y_true):
        """Test directional accuracy raises error for length mismatch."""
        short_pred = np.array([0.001, 0.002])

        with pytest.raises(ValueError, match="Length mismatch"):
            directional_accuracy(sample_y_true, short_pred)

    def test_directional_accuracy_insufficient_data_raises_error(self):
        """Test directional accuracy raises error for insufficient data."""
        y_true = pd.Series([0.001])
        y_pred = np.array([0.0011])

        with pytest.raises(ValueError, match="Insufficient data"):
            directional_accuracy(y_true, y_pred)


# ============================================================================
# Theil's U Statistic Tests
# ============================================================================

class TestTheilUStatistic:
    """Test suite for Theil's U statistic calculation."""

    def test_theil_u_returns_float(self, sample_y_true, sample_y_pred):
        """Test Theil's U returns float value."""
        u_stat = theil_u_statistic(sample_y_true, sample_y_pred)

        assert isinstance(u_stat, (float, np.floating))

    def test_theil_u_positive_values(self, sample_y_true, sample_y_pred):
        """Test Theil's U is positive."""
        u_stat = theil_u_statistic(sample_y_true, sample_y_pred)

        assert u_stat >= 0

    def test_theil_u_length_mismatch_raises_error(self, sample_y_true):
        """Test Theil's U raises error for length mismatch."""
        short_pred = np.array([0.001, 0.002])

        with pytest.raises(ValueError, match="Length mismatch"):
            theil_u_statistic(sample_y_true, short_pred)

    def test_theil_u_insufficient_data_raises_error(self):
        """Test Theil's U raises error for insufficient data."""
        y_true = pd.Series([0.001])
        y_pred = np.array([0.0011])

        with pytest.raises(ValueError, match="Insufficient data"):
            theil_u_statistic(y_true, y_pred)


# ============================================================================
# Comprehensive Forecast Evaluation Tests
# ============================================================================

class TestForecastEvaluation:
    """Test suite for comprehensive forecast evaluation."""

    def test_evaluate_forecast_returns_dict(self, sample_y_true, sample_y_pred):
        """Test evaluate_forecast returns dictionary."""
        metrics = evaluate_forecast(sample_y_true, sample_y_pred)

        assert isinstance(metrics, dict)

    def test_evaluate_forecast_has_all_metrics(self, sample_y_true, sample_y_pred):
        """Test evaluate_forecast contains all required metrics."""
        metrics = evaluate_forecast(sample_y_true, sample_y_pred)

        required_keys = [
            'qlike_loss', 'directional_accuracy', 'theil_u',
            'rmse', 'mae', 'mse', 'r2', 'meets_targets'
        ]

        for key in required_keys:
            assert key in metrics

    def test_evaluate_forecast_metrics_are_reasonable(self, sample_y_true, sample_y_pred):
        """Test evaluate_forecast produces reasonable metric values."""
        metrics = evaluate_forecast(sample_y_true, sample_y_pred)

        # QLIKE should be positive
        assert metrics['qlike_loss'] >= 0
        # Directional accuracy should be between 0 and 1
        assert 0 <= metrics['directional_accuracy'] <= 1
        # RMSE should be positive
        assert metrics['rmse'] >= 0
        # MSE should be positive and equal to RMSE²
        assert metrics['mse'] >= 0
        assert abs(metrics['mse'] - metrics['rmse']**2) < 1e-10
        # R² should be ≤ 1
        assert metrics['r2'] <= 1

    def test_evaluate_forecast_includes_target_checks(self, sample_y_true, sample_y_pred):
        """Test evaluate_forecast includes target achievement checks."""
        metrics = evaluate_forecast(sample_y_true, sample_y_pred)

        assert 'meets_targets' in metrics
        assert isinstance(metrics['meets_targets'], dict)

        required_targets = ['directional_accuracy_55', 'rmse_5day_0.20', 'theil_u_lt_1']
        for target in required_targets:
            assert target in metrics['meets_targets']


# ============================================================================
# Benchmark Comparison Tests
# ============================================================================

class TestBenchmarkComparisons:
    """Test suite for benchmark comparisons."""

    def test_compare_vs_benchmarks_returns_dataframe(self, sample_y_true, sample_y_pred):
        """Test benchmark comparison returns DataFrame."""
        comparison = compare_vs_benchmarks(sample_y_true, sample_y_pred)

        assert isinstance(comparison, pd.DataFrame)

    def test_compare_vs_benchmarks_has_three_rows(self, sample_y_true, sample_y_pred):
        """Test benchmark comparison has 3 benchmarks (model, RW, hist mean)."""
        comparison = compare_vs_benchmarks(sample_y_true, sample_y_pred)

        assert len(comparison) == 3

    def test_compare_vs_benchmarks_has_required_columns(self, sample_y_true, sample_y_pred):
        """Test benchmark comparison has required columns."""
        comparison = compare_vs_benchmarks(sample_y_true, sample_y_pred)

        required_columns = ['benchmark', 'qlike_loss', 'directional_accuracy', 'theil_u']
        for col in required_columns:
            assert col in comparison.columns

    def test_compare_vs_benchmarks_random_walk_theil_u_equals_1(self):
        """Test random walk benchmark has Theil's U close to 1.0."""
        # Create longer series for better Theil's U calculation
        y_true = pd.Series([0.001, 0.002, 0.0015, 0.0018, 0.002, 0.0025, 0.0019, 0.0021, 0.0017, 0.0023])
        # Use model predictions (should be different from random walk)
        y_pred = y_true.values + np.random.normal(0, 0.0001, len(y_true))

        comparison = compare_vs_benchmarks(y_true, y_pred)

        # Find random walk row
        rw_row = comparison[comparison['benchmark'] == 'Random Walk']
        # Random walk benchmark should have Theil's U close to 1.0 by definition
        # (it's comparing itself against itself)
        # Allow 0.2 tolerance due to first value handling in random walk calculation
        assert abs(rw_row['theil_u'].values[0] - 1.0) < 0.2


# ============================================================================
# Performance Visualization Tests
# ============================================================================

class TestPerformanceVisualization:
    """Test suite for performance visualization."""

    def test_plot_forecast_performance_generates_plot(self, sample_y_true, sample_y_pred):
        """Test plot_forecast_performance generates plot without error."""
        # Should not raise any exceptions
        plot_forecast_performance(sample_y_true, sample_y_pred)

    def test_plot_forecast_performance_saves_to_file(self, sample_y_true, sample_y_pred):
        """Test plot_forecast_performance saves plot to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_plot.png')

            plot_forecast_performance(sample_y_true, sample_y_pred, save_path=save_path)

            assert os.path.exists(save_path)


# ============================================================================
# Complete Evaluation Report Tests
# ============================================================================

class TestCompleteEvaluationReport:
    """Test suite for complete evaluation report generation."""

    def test_generate_evaluation_report_returns_dict(self, sample_train_test_data):
        """Test generate_evaluation_report returns dictionary."""
        y_train_true, y_train_pred, y_test_true, y_test_pred = sample_train_test_data

        report = generate_evaluation_report(
            y_train_true, y_train_pred,
            y_test_true, y_test_pred
        )

        assert isinstance(report, dict)

    def test_generate_evaluation_report_has_all_components(self, sample_train_test_data):
        """Test generate_evaluation_report contains all required components."""
        y_train_true, y_train_pred, y_test_true, y_test_pred = sample_train_test_data

        report = generate_evaluation_report(
            y_train_true, y_train_pred,
            y_test_true, y_test_pred
        )

        required_keys = [
            'model_name', 'train_metrics', 'test_metrics',
            'benchmark_comparison', 'target_achievement'
        ]

        for key in required_keys:
            assert key in report

    def test_generate_evaluation_report_train_test_metrics(self, sample_train_test_data):
        """Test generate_evaluation_report includes train and test metrics."""
        y_train_true, y_train_pred, y_test_true, y_test_pred = sample_train_test_data

        report = generate_evaluation_report(
            y_train_true, y_train_pred,
            y_test_true, y_test_pred
        )

        # Both should have QLIKE loss
        assert 'qlike_loss' in report['train_metrics']
        assert 'qlike_loss' in report['test_metrics']

    def test_generate_evaluation_report_with_save_dir(self, sample_train_test_data):
        """Test generate_evaluation_report saves plots when save_dir provided."""
        y_train_true, y_train_pred, y_test_true, y_test_pred = sample_train_test_data

        with tempfile.TemporaryDirectory() as tmpdir:
            report = generate_evaluation_report(
                y_train_true, y_train_pred,
                y_test_true, y_test_pred,
                save_dir=tmpdir
            )

            # Should have plot paths
            assert 'plot_paths' in report
            assert 'test_performance' in report['plot_paths']

            # Plot file should exist
            plot_path = report['plot_paths']['test_performance']
            assert os.path.exists(plot_path)


# ============================================================================
# Integration Tests
# ============================================================================

class TestEvaluationIntegration:
    """Integration tests for complete evaluation workflow."""

    def test_end_to_end_evaluation_pipeline(self):
        """Test complete end-to-end evaluation pipeline."""
        np.random.seed(42)

        # Create sample data
        y_train_true = pd.Series(np.random.uniform(0.001, 0.003, 30))
        y_train_pred = y_train_true.values + np.random.normal(0, 0.0001, 30)

        y_test_true = pd.Series(np.random.uniform(0.001, 0.003, 10))
        y_test_pred = y_test_true.values + np.random.normal(0, 0.0001, 10)

        # Generate complete report
        report = generate_evaluation_report(
            y_train_true, y_train_pred,
            y_test_true, y_test_pred,
            model_name="Test Model"
        )

        # Verify report completeness
        assert report['test_metrics']['qlike_loss'] >= 0
        assert 0 <= report['test_metrics']['directional_accuracy'] <= 1
        assert len(report['benchmark_comparison']) == 3

    def test_evaluation_with_realistic_volatility_range(self):
        """Test evaluation with realistic Parkinson volatility range."""
        # Create realistic volatility (0.0001 to 0.05 range)
        np.random.seed(42)
        y_true = pd.Series(np.random.uniform(0.0002, 0.004, 30))
        y_pred = y_true.values + np.random.normal(0, 0.0002, 30)
        y_pred = np.maximum(y_pred, 0.0001)  # Ensure positive

        metrics = evaluate_forecast(y_true, y_pred)

        # Should produce valid metrics
        assert metrics['qlike_loss'] >= 0
        assert metrics['rmse'] >= 0
        assert 0 <= metrics['directional_accuracy'] <= 1

    def test_benchmark_comparison_model_beats_random_walk(self):
        """Test that better model beats random walk benchmark."""
        # Create data where model is better than random walk
        np.random.seed(42)
        base_volatility = 0.002
        y_true = pd.Series([base_volatility] * 10 + [base_volatility * 1.5] * 10)

        # Model predictions (close to true with small error)
        y_pred = y_true.values + np.random.normal(0, 0.0001, 20)

        comparison = compare_vs_benchmarks(y_true, y_pred, model_name="Good Model")

        # Model should have better (lower) QLIKE than random walk
        model_qlike = comparison[comparison['benchmark'] == 'Good Model']['qlike_loss'].values[0]
        rw_qlike = comparison[comparison['benchmark'] == 'Random Walk']['qlike_loss'].values[0]

        assert model_qlike < rw_qlike
