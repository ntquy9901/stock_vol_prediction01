"""Tests for S&P 500 download script."""
import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Bootstrap path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)


@pytest.fixture
def mock_hf_dataset():
    """Mock Hugging Face dataset with 3 stocks."""
    return pd.DataFrame({
        'date': ['2011-01-03', '2011-01-04', '2011-01-03', '2011-01-04', '2011-01-03', '2011-01-04'],
        'act_symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT', 'GOOGL', 'GOOGL'],
        'open': [325.9, 330.0, 28.0, 28.5, 600.0, 610.0],
        'high': [330.26, 333.5, 28.5, 29.0, 610.0, 620.0],
        'low': [324.84, 328.0, 27.5, 28.0, 595.0, 605.0],
        'close': [329.57, 332.5, 28.2, 28.8, 605.0, 615.0],
        'volume': [15897201, 14500000, 25000000, 24000000, 3000000, 2800000],
    })


class TestDownloadAndSave:
    """Test download_and_save function."""

    @patch('datasets.load_dataset')
    def test_creates_output_directory(self, mock_load, mock_hf_dataset, tmp_path):
        """Should create output directory if it doesn't exist."""
        from datasets import Dataset
        mock_ds = MagicMock(spec=Dataset)
        mock_ds.to_pandas.return_value = mock_hf_dataset
        mock_load.return_value = mock_ds

        from src.experiments.sp500.download_sp500 import download_and_save
        output_dir = str(tmp_path / "prices_sp500")
        download_and_save(output_dir=output_dir, tickers=["AAPL", "MSFT", "GOOGL"])

        assert os.path.isdir(output_dir)

    @patch('datasets.load_dataset')
    def test_creates_per_stock_csv(self, mock_load, mock_hf_dataset, tmp_path):
        """Should create one CSV per ticker."""
        from datasets import Dataset
        mock_ds = MagicMock(spec=Dataset)
        mock_ds.to_pandas.return_value = mock_hf_dataset
        mock_load.return_value = mock_ds

        from src.experiments.sp500.download_sp500 import download_and_save
        output_dir = str(tmp_path / "prices_sp500")
        download_and_save(output_dir=output_dir, tickers=["AAPL", "MSFT"])

        assert os.path.isfile(os.path.join(output_dir, "AAPL.csv"))
        assert os.path.isfile(os.path.join(output_dir, "MSFT.csv"))

    @patch('datasets.load_dataset')
    def test_csv_has_vn30_columns(self, mock_load, mock_hf_dataset, tmp_path):
        """CSV files should have VN30-compatible columns."""
        from datasets import Dataset
        mock_ds = MagicMock(spec=Dataset)
        mock_ds.to_pandas.return_value = mock_hf_dataset
        mock_load.return_value = mock_ds

        from src.experiments.sp500.download_sp500 import download_and_save
        output_dir = str(tmp_path / "prices_sp500")
        download_and_save(output_dir=output_dir, tickers=["AAPL"])

        df = pd.read_csv(os.path.join(output_dir, "AAPL.csv"))
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    @patch('datasets.load_dataset')
    def test_csv_row_count(self, mock_load, mock_hf_dataset, tmp_path):
        """Each CSV should have correct number of rows."""
        from datasets import Dataset
        mock_ds = MagicMock(spec=Dataset)
        mock_ds.to_pandas.return_value = mock_hf_dataset
        mock_load.return_value = mock_ds

        from src.experiments.sp500.download_sp500 import download_and_save
        output_dir = str(tmp_path / "prices_sp500")
        download_and_save(output_dir=output_dir, tickers=["AAPL", "MSFT", "GOOGL"])

        aapl_df = pd.read_csv(os.path.join(output_dir, "AAPL.csv"))
        assert len(aapl_df) == 2  # 2 rows for AAPL in mock data

    @patch('datasets.load_dataset')
    def test_filters_to_requested_tickers(self, mock_load, mock_hf_dataset, tmp_path):
        """Should only create CSVs for requested tickers."""
        from datasets import Dataset
        mock_ds = MagicMock(spec=Dataset)
        mock_ds.to_pandas.return_value = mock_hf_dataset
        mock_load.return_value = mock_ds

        from src.experiments.sp500.download_sp500 import download_and_save
        output_dir = str(tmp_path / "prices_sp500")
        download_and_save(output_dir=output_dir, tickers=["AAPL"])

        # Only AAPL.csv should exist, not MSFT.csv or GOOGL.csv
        assert os.path.isfile(os.path.join(output_dir, "AAPL.csv"))
        assert not os.path.isfile(os.path.join(output_dir, "MSFT.csv"))
        assert not os.path.isfile(os.path.join(output_dir, "GOOGL.csv"))
