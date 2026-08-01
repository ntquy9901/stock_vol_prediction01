"""Tests for market data loader."""
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
def sample_market_data():
    """Sample market data CSV content."""
    return pd.DataFrame({
        "Date": ["2011-01-03", "2011-01-04", "2011-01-05"],
        "Close": [20.5, 21.0, 20.8],
    })


class TestMarketDataLoader:
    """Test market data loader functions."""

    def test_load_market_data_returns_dataframe(self, sample_market_data, tmp_path):
        """load_market_data should return a DataFrame."""
        from src.common.market_data_loader import load_market_data

        # Write sample data
        data_dir = str(tmp_path / "market_data")
        os.makedirs(data_dir)
        sample_market_data.to_csv(os.path.join(data_dir, "vix.csv"), index=False)

        with patch("src.common.market_data_loader.MARKET_INDICATORS", {"test": {"vix": "^VIX"}}):
            result = load_market_data(market="test", data_dir=data_dir)

        assert isinstance(result, pd.DataFrame)
        assert "vix" in result.columns

    def test_merge_with_stock_data(self, sample_market_data):
        """merge_with_stock_data should add market columns to stock data."""
        from src.common.market_data_loader import merge_with_stock_data

        stock_df = pd.DataFrame({
            "Date": ["2011-01-03", "2011-01-04", "2011-01-05"],
            "Open": [100.0, 101.0, 102.0],
            "Close": [100.5, 101.5, 102.5],
        })

        market_df = sample_market_data.set_index("Date")
        market_df.index = pd.to_datetime(market_df.index)
        market_df = market_df.rename(columns={"Close": "vix"})

        result = merge_with_stock_data(stock_df, market_df)

        assert "vix" in result.columns
        assert len(result) == 3
        assert result["vix"].iloc[0] == 20.5

    def test_market_indicators_defined(self):
        """MARKET_INDICATORS should have sp500 and vn30."""
        from src.common.market_data_loader import MARKET_INDICATORS

        assert "sp500" in MARKET_INDICATORS
        assert "vn30" in MARKET_INDICATORS
        assert "vix" in MARKET_INDICATORS["sp500"]
        assert "treasury_10y" in MARKET_INDICATORS["sp500"]
