"""Tests for S&P 500 data adapter."""
import os
import sys
import pandas as pd
import pytest

# Bootstrap path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)


@pytest.fixture
def sample_hf_ohlcv():
    """Sample Hugging Face stocks-ohlcv format."""
    return pd.DataFrame({
        'date': ['2011-01-03 00:00:00', '2011-01-04 00:00:00', '2011-01-05 00:00:00'],
        'act_symbol': ['AAPL', 'AAPL', 'AAPL'],
        'open': [325.9, 330.0, 332.0],
        'high': [330.26, 333.5, 335.0],
        'low': [324.84, 328.0, 330.0],
        'close': [329.57, 332.5, 334.0],
        'volume': [15897201, 14500000, 13200000],
    })


@pytest.fixture
def sample_multi_stock():
    """Sample with multiple stocks."""
    return pd.DataFrame({
        'date': ['2011-01-03', '2011-01-03', '2011-01-04', '2011-01-04'],
        'act_symbol': ['AAPL', 'MSFT', 'AAPL', 'MSFT'],
        'open': [325.9, 28.0, 330.0, 28.5],
        'high': [330.26, 28.5, 333.5, 29.0],
        'low': [324.84, 27.5, 328.0, 28.0],
        'close': [329.57, 28.2, 332.5, 28.8],
        'volume': [15897201, 25000000, 14500000, 24000000],
    })


class TestAdaptToVN30Format:
    """Test adapt_to_vn30_format function."""

    def test_adapter_returns_dataframe(self, sample_hf_ohlcv):
        """Adapter should return a pandas DataFrame."""
        from src.common.data_adapters import adapt_to_vn30_format
        result = adapt_to_vn30_format(sample_hf_ohlcv, source="stocks_ohlcv")
        assert isinstance(result, pd.DataFrame)

    def test_adapter_has_vn30_columns(self, sample_hf_ohlcv):
        """Output should have VN30-compatible column names."""
        from src.common.data_adapters import adapt_to_vn30_format
        result = adapt_to_vn30_format(sample_hf_ohlcv, source="stocks_ohlcv")
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_adapter_date_format(self, sample_hf_ohlcv):
        """Date should be YYYY-MM-DD format (no time component)."""
        from src.common.data_adapters import adapt_to_vn30_format
        result = adapt_to_vn30_format(sample_hf_ohlcv, source="stocks_ohlcv")
        # Should be string in YYYY-MM-DD format
        assert result['Date'].iloc[0] == '2011-01-03'

    def test_adapter_row_count_preserved(self, sample_hf_ohlcv):
        """Row count should be preserved after adaptation."""
        from src.common.data_adapters import adapt_to_vn30_format
        result = adapt_to_vn30_format(sample_hf_ohlcv, source="stocks_ohlcv")
        assert len(result) == len(sample_hf_ohlcv)

    def test_adapter_values_preserved(self, sample_hf_ohlcv):
        """OHLCV values should be preserved exactly."""
        from src.common.data_adapters import adapt_to_vn30_format
        result = adapt_to_vn30_format(sample_hf_ohlcv, source="stocks_ohlcv")
        assert result['Open'].iloc[0] == 325.9
        assert result['High'].iloc[0] == 330.26
        assert result['Low'].iloc[0] == 324.84
        assert result['Close'].iloc[0] == 329.57
        assert result['Volume'].iloc[0] == 15897201

    def test_adapter_invalid_source_raises(self, sample_hf_ohlcv):
        """Invalid source should raise ValueError."""
        from src.common.data_adapters import adapt_to_vn30_format
        with pytest.raises(ValueError, match="Unknown source"):
            adapt_to_vn30_format(sample_hf_ohlcv, source="unknown_format")


class TestSplitByTicker:
    """Test split_by_ticker function."""

    def test_split_creates_dict(self, sample_multi_stock):
        """Should return dict of DataFrames keyed by ticker."""
        from src.common.data_adapters import split_by_ticker
        result = split_by_ticker(sample_multi_stock)
        assert isinstance(result, dict)
        assert 'AAPL' in result
        assert 'MSFT' in result

    def test_split_row_counts(self, sample_multi_stock):
        """Each ticker DataFrame should have correct row count."""
        from src.common.data_adapters import split_by_ticker
        result = split_by_ticker(sample_multi_stock)
        assert len(result['AAPL']) == 2
        assert len(result['MSFT']) == 2

    def test_split_vn30_format(self, sample_multi_stock):
        """Split DataFrames should have VN30 column names."""
        from src.common.data_adapters import split_by_ticker
        result = split_by_ticker(sample_multi_stock)
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_cols:
            assert col in result['AAPL'].columns
            assert col in result['MSFT'].columns
