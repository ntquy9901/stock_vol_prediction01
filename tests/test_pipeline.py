"""
Test Suite for Integration Pipeline Module.

This test suite validates end-to-end pipeline orchestration, data loading,
single/multi-stock processing, and summary report generation.

Test Coverage:
    - OHLCV data loading from CSV files
    - Single stock complete pipeline
    - Multiple stocks batch processing
    - Summary report generation
    - Utility functions (list data, validate directory)
    - Integration tests with mock data
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json
import os

from src.pipeline import (
    load_ohlcv_data,
    process_single_stock,
    process_multiple_stocks,
    generate_summary_report,
    run_complete_pipeline,
    list_available_data,
    validate_data_directory,
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    RANDOM_SEED
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv_csv():
    """Create sample OHLCV CSV file for testing."""
    np.random.seed(42)

    # Generate sample OHLCV data (100 days)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # Generate realistic OHLCV data
    close_prices = np.cumsum(np.random.normal(0.001, 0.02, 100)) + 100
    data = {
        'Date': dates.strftime('%Y-%m-%d'),
        'Open': close_prices * (1 + np.random.uniform(-0.02, 0.02, 100)),
        'High': close_prices * (1 + np.random.uniform(0.01, 0.05, 100)),
        'Low': close_prices * (1 - np.random.uniform(0.01, 0.05, 100)),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }

    df = pd.DataFrame(data)

    # Create temporary directory and file
    temp_dir = tempfile.mkdtemp()
    csv_path = Path(temp_dir) / 'VCB.csv'
    df.to_csv(csv_path, index=False)

    yield temp_dir, csv_path, df

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_multiple_stocks_csv():
    """Create sample OHLCV CSV files for multiple stocks."""
    np.random.seed(42)

    tickers = ['VCB', 'VIC', 'VHM']
    temp_dir = tempfile.mkdtemp()
    csv_files = {}

    for ticker in tickers:
        # Generate sample data for each stock
        dates = pd.date_range('2024-01-01', periods=80, freq='D')
        close_prices = np.cumsum(np.random.normal(0.001, 0.02, 80)) + 100

        data = {
            'Date': dates.strftime('%Y-%m-%d'),
            'Open': close_prices * (1 + np.random.uniform(-0.02, 0.02, 80)),
            'High': close_prices * (1 + np.random.uniform(0.01, 0.05, 80)),
            'Low': close_prices * (1 - np.random.uniform(0.01, 0.05, 80)),
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 10000000, 80)
        }

        df = pd.DataFrame(data)
        csv_path = Path(temp_dir) / f'{ticker}.csv'
        df.to_csv(csv_path, index=False)
        csv_files[ticker] = csv_path

    yield temp_dir, csv_files

    # Cleanup
    shutil.rmtree(temp_dir)


# ============================================================================
# Data Loading Tests
# ============================================================================

class TestDataLoading:
    """Test suite for OHLCV data loading."""

    def test_load_ohlcv_data_loads_file(self, sample_ohlcv_csv):
        """Test load_ohlcv_data successfully loads CSV file."""
        temp_dir, csv_path, original_df = sample_ohlcv_csv

        loaded_df = load_ohlcv_data('VCB', data_dir=Path(temp_dir))

        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) > 0

    def test_load_ohlcv_data_has_required_columns(self, sample_ohlcv_csv):
        """Test loaded data has all required OHLCV columns."""
        temp_dir, csv_path, original_df = sample_ohlcv_csv

        loaded_df = load_ohlcv_data('VCB', data_dir=Path(temp_dir))

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            assert col in loaded_df.columns

    def test_load_ohlcv_data_parses_date_index(self, sample_ohlcv_csv):
        """Test loaded data has date as index."""
        temp_dir, csv_path, original_df = sample_ohlcv_csv

        loaded_df = load_ohlcv_data('VCB', data_dir=Path(temp_dir))

        assert isinstance(loaded_df.index, pd.DatetimeIndex)

    def test_load_ohlcv_file_not_found_raises_error(self):
        """Test load_ohlcv_data raises error for missing file."""
        temp_dir = tempfile.mkdtemp()

        with pytest.raises(FileNotFoundError, match="OHLCV data file not found"):
            load_ohlcv_data('NONEXISTENT', data_dir=Path(temp_dir))

        shutil.rmtree(temp_dir)


# ============================================================================
# Single Stock Processing Tests
# ============================================================================

class TestSingleStockProcessing:
    """Test suite for single stock complete pipeline."""

    def test_process_single_stock_returns_dict(self, sample_ohlcv_csv):
        """Test process_single_stock returns results dictionary."""
        temp_dir, csv_path, original_df = sample_ohlcv_csv
        results_dir = tempfile.mkdtemp()

        results = process_single_stock(
            'VCB',
            data_dir=Path(temp_dir),
            results_dir=Path(results_dir),
            save_model=False,
            save_plots=False
        )

        assert isinstance(results, dict)
        assert 'ticker' in results
        assert 'status' in results

        shutil.rmtree(results_dir)

    def test_process_single_stock_success_status(self, sample_ohlcv_csv):
        """Test process_single_stock has SUCCESS status."""
        temp_dir, csv_path, original_df = sample_ohlcv_csv
        results_dir = tempfile.mkdtemp()

        results = process_single_stock(
            'VCB',
            data_dir=Path(temp_dir),
            results_dir=Path(results_dir),
            save_model=False,
            save_plots=False
        )

        assert results['status'] == 'SUCCESS'

        shutil.rmtree(results_dir)

    def test_process_single_stock_has_test_metrics(self, sample_ohlcv_csv):
        """Test process_single_stock includes test metrics."""
        temp_dir, csv_path, original_df = sample_ohlcv_csv
        results_dir = tempfile.mkdtemp()

        results = process_single_stock(
            'VCB',
            data_dir=Path(temp_dir),
            results_dir=Path(results_dir),
            save_model=False,
            save_plots=False
        )

        assert 'test_results' in results
        assert 'qlike_loss' in results['test_results']
        assert 'directional_accuracy' in results['test_results']
        assert 'theil_u' in results['test_results']

        shutil.rmtree(results_dir)

    def test_process_single_stock_saves_results(self, sample_ohlcv_csv):
        """Test process_single_stock saves results when save_plots=True."""
        temp_dir, csv_path, original_df = sample_ohlcv_csv
        results_dir = tempfile.mkdtemp()

        results = process_single_stock(
            'VCB',
            data_dir=Path(temp_dir),
            results_dir=Path(results_dir),
            save_model=False,
            save_plots=True
        )

        # Check that results directory was created
        stock_results_dir = Path(results_dir) / 'VCB'
        assert stock_results_dir.exists()

        # Check that summary JSON was created
        summary_file = stock_results_dir / 'VCB_summary.json'
        assert summary_file.exists()

        shutil.rmtree(results_dir)


# ============================================================================
# Multiple Stocks Processing Tests
# ============================================================================

class TestMultipleStocksProcessing:
    """Test suite for batch processing multiple stocks."""

    def test_process_multiple_stocks_returns_dict(self, sample_multiple_stocks_csv):
        """Test process_multiple_stocks returns results dictionary."""
        temp_dir, csv_files = sample_multiple_stocks_csv
        results_dir = tempfile.mkdtemp()
        tickers = list(csv_files.keys())

        results = process_multiple_stocks(
            tickers,
            data_dir=Path(temp_dir),
            results_dir=Path(results_dir),
            save_models=False,
            save_plots=False
        )

        assert isinstance(results, dict)
        assert 'total_stocks' in results
        assert 'successful' in results
        assert 'failed' in results

        shutil.rmtree(results_dir)

    def test_process_multiple_stocks_processes_all_stocks(self, sample_multiple_stocks_csv):
        """Test process_multiple_stocks processes all provided stocks."""
        temp_dir, csv_files = sample_multiple_stocks_csv
        results_dir = tempfile.mkdtemp()
        tickers = list(csv_files.keys())

        results = process_multiple_stocks(
            tickers,
            data_dir=Path(temp_dir),
            results_dir=Path(results_dir),
            save_models=False,
            save_plots=False
        )

        assert results['total_stocks'] == len(tickers)
        assert results['successful'] + results['failed'] == len(tickers)

        shutil.rmtree(results_dir)

    def test_process_multiple_stocks_includes_summary(self, sample_multiple_stocks_csv):
        """Test process_multiple_stocks includes aggregate summary."""
        temp_dir, csv_files = sample_multiple_stocks_csv
        results_dir = tempfile.mkdtemp()
        tickers = list(csv_files.keys())

        results = process_multiple_stocks(
            tickers,
            data_dir=Path(temp_dir),
            results_dir=Path(results_dir),
            save_models=False,
            save_plots=False
        )

        assert 'summary' in results
        assert 'mean_qlike_loss' in results['summary']
        assert 'mean_directional_accuracy' in results['summary']

        shutil.rmtree(results_dir)


# ============================================================================
# Summary Report Tests
# ============================================================================

class TestSummaryReport:
    """Test suite for summary report generation."""

    def test_generate_summary_report_returns_dataframe(self, sample_multiple_stocks_csv):
        """Test generate_summary_report returns DataFrame."""
        temp_dir, csv_files = sample_multiple_stocks_csv
        results_dir = tempfile.mkdtemp()
        tickers = list(csv_files.keys())

        batch_results = process_multiple_stocks(
            tickers,
            data_dir=Path(temp_dir),
            results_dir=Path(results_dir),
            save_models=False,
            save_plots=False
        )

        summary_df = generate_summary_report(batch_results)

        assert isinstance(summary_df, pd.DataFrame)

        shutil.rmtree(results_dir)

    def test_generate_summary_report_has_correct_columns(self, sample_multiple_stocks_csv):
        """Test summary report has required columns."""
        temp_dir, csv_files = sample_multiple_stocks_csv
        results_dir = tempfile.mkdtemp()
        tickers = list(csv_files.keys())

        batch_results = process_multiple_stocks(
            tickers,
            data_dir=Path(temp_dir),
            results_dir=Path(results_dir),
            save_models=False,
            save_plots=False
        )

        summary_df = generate_summary_report(batch_results)

        required_columns = ['ticker', 'status', 'test_qlike_loss', 'test_directional_accuracy', 'test_theil_u']
        for col in required_columns:
            assert col in summary_df.columns

        shutil.rmtree(results_dir)

    def test_generate_summary_report_saves_to_csv(self, sample_multiple_stocks_csv):
        """Test summary report saves to CSV when output_path provided."""
        temp_dir, csv_files = sample_multiple_stocks_csv
        results_dir = tempfile.mkdtemp()
        tickers = list(csv_files.keys())

        batch_results = process_multiple_stocks(
            tickers,
            data_dir=Path(temp_dir),
            results_dir=Path(results_dir),
            save_models=False,
            save_plots=False
        )

        output_path = Path(results_dir) / 'test_summary.csv'
        summary_df = generate_summary_report(batch_results, output_path=output_path)

        assert output_path.exists()

        shutil.rmtree(results_dir)


# ============================================================================
# Utility Functions Tests
# ============================================================================

class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_list_available_data_returns_list(self, sample_multiple_stocks_csv):
        """Test list_available_data returns list of tickers."""
        temp_dir, csv_files = sample_multiple_stocks_csv

        tickers = list_available_data(data_dir=Path(temp_dir))

        assert isinstance(tickers, list)

    def test_list_available_data_finds_all_stocks(self, sample_multiple_stocks_csv):
        """Test list_available_data finds all stock files."""
        temp_dir, csv_files = sample_multiple_stocks_csv

        tickers = list_available_data(data_dir=Path(temp_dir))

        # Should find at least the tickers we created
        for ticker in csv_files.keys():
            assert ticker in tickers

    def test_validate_data_directory_true_when_exists(self, sample_multiple_stocks_csv):
        """Test validate_data_directory returns True for valid directory."""
        temp_dir, csv_files = sample_multiple_stocks_csv

        is_valid = validate_data_directory(data_dir=Path(temp_dir))

        assert is_valid is True

    def test_validate_data_directory_false_when_not_exists(self):
        """Test validate_data_directory returns False for non-existent directory."""
        is_valid = validate_data_directory(data_dir=Path('nonexistent_dir'))

        assert is_valid is False


# ============================================================================
# Integration Tests
# ============================================================================

class TestPipelineIntegration:
    """Integration tests for complete pipeline workflow."""

    def test_end_to_end_single_stock_pipeline(self, sample_ohlcv_csv):
        """Test complete end-to-end pipeline for single stock."""
        temp_dir, csv_path, original_df = sample_ohlcv_csv
        results_dir = tempfile.mkdtemp()

        results = process_single_stock(
            'VCB',
            data_dir=Path(temp_dir),
            results_dir=Path(results_dir),
            save_model=True,
            save_plots=True
        )

        # Verify complete results
        assert results['status'] == 'SUCCESS'
        assert results['test_results']['qlike_loss'] >= 0
        assert 0 <= results['test_results']['directional_accuracy'] <= 1
        assert results['feature_coefficients'] is not None

        # Verify saved files
        stock_dir = Path(results_dir) / 'VCB'
        assert stock_dir.exists()
        assert (stock_dir / 'VCB_summary.json').exists()

        shutil.rmtree(results_dir)

    def test_end_to_end_batch_pipeline(self, sample_multiple_stocks_csv):
        """Test complete end-to-end batch pipeline."""
        temp_dir, csv_files = sample_multiple_stocks_csv
        results_dir = tempfile.mkdtemp()
        tickers = list(csv_files.keys())

        # Run complete pipeline
        final_results = run_complete_pipeline(
            tickers,
            data_dir=Path(temp_dir),
            results_dir=Path(results_dir),
            save_models=True,
            save_plots=True
        )

        # Verify structure
        assert 'batch_results' in final_results
        assert 'summary_df' in final_results
        assert 'summary_stats' in final_results

        # Verify saved files
        assert (Path(results_dir) / 'summary_report.csv').exists()
        assert (Path(results_dir) / 'aggregate_results.json').exists()

        # Verify individual stock results
        for ticker in tickers:
            stock_dir = Path(results_dir) / ticker
            assert stock_dir.exists()

        shutil.rmtree(results_dir)

    def test_pipeline_with_realistic_data_volume(self):
        """Test pipeline with realistic data volume (2006-2024)."""
        # Generate longer time series (simulating 20 years)
        np.random.seed(42)
        temp_dir = tempfile.mkdtemp()
        results_dir = tempfile.mkdtemp()

        # Create realistic data (5000 data points ~ 20 years)
        dates = pd.date_range('2006-01-01', periods=5000, freq='D')
        close_prices = np.cumsum(np.random.normal(0.0005, 0.015, 5000)) + 100

        data = {
            'Date': dates.strftime('%Y-%m-%d'),
            'Open': close_prices * (1 + np.random.uniform(-0.01, 0.01, 5000)),
            'High': close_prices * (1 + np.random.uniform(0.005, 0.02, 5000)),
            'Low': close_prices * (1 - np.random.uniform(0.005, 0.02, 5000)),
            'Close': close_prices,
            'Volume': np.random.randint(500000, 5000000, 5000)
        }

        df = pd.DataFrame(data)
        csv_path = Path(temp_dir) / 'VCB.csv'
        df.to_csv(csv_path, index=False)

        # Process stock
        results = process_single_stock(
            'VCB',
            data_dir=Path(temp_dir),
            results_dir=Path(results_dir),
            save_model=False,
            save_plots=False
        )

        # Should handle large dataset successfully
        assert results['status'] == 'SUCCESS'
        assert results['data_info']['rows'] == 5000

        shutil.rmtree(temp_dir)
        shutil.rmtree(results_dir)
