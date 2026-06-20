"""
Shared test fixtures for stock volatility prediction tests.

This module provides pytest fixtures for sample OHLCV data, test data generation,
and common test utilities used across the test suite.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlc_data():
    """
    Fixture providing valid OHLCV data for testing.

    Returns:
        pd.DataFrame: Sample OHLCV data with 100 days of valid data
                      High ≥ Close ≥ Low, Volume > 0
    """
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')

    # Generate realistic price movements
    base_price = 100.0
    returns = np.random.normal(0, 0.02, 100)
    prices = base_price * (1 + returns).cumprod()

    # Create OHLCV data with proper relationships
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    })

    # Ensure OHLCV consistency
    data['High'] = data[['Open', 'Close']].max(axis=1) * 1.01
    data['Low'] = data[['Open', 'Close']].min(axis=1) * 0.99

    return data.set_index('Date')


@pytest.fixture
def sample_ohlc_data_invalid():
    """
    Fixture providing invalid OHLCV data for testing validation.

    Returns:
        pd.DataFrame: OHLCV data with invalid relationships (High < Low)
    """
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Open': [100.0] * 10,
        'High': [99.0] * 10,  # Invalid: High < Open
        'Low': [101.0] * 10,  # Invalid: Low > Open
        'Close': [100.0] * 10,
        'Volume': [1000000] * 10
    })
    return data.set_index('Date')


@pytest.fixture
def sample_ohlc_data_with_missing():
    """
    Fixture providing OHLCV data with missing values for testing.

    Returns:
        pd.DataFrame: OHLCV data with some missing values
    """
    dates = pd.date_range(start='2020-01-01', periods=20, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Open': [100.0] * 20,
        'High': [102.0] * 20,
        'Low': [98.0] * 20,
        'Close': [100.0] * 20,
        'Volume': [1000000] * 20
    })

    # Introduce missing values
    data.loc[5:7, 'High'] = np.nan
    data.loc[10:12, 'Close'] = np.nan

    return data.set_index('Date')


@pytest.fixture
def sample_ohlc_data_extreme_values():
    """
    Fixture providing OHLCV data with extreme volatility values.

    Returns:
        pd.DataFrame: OHLCV data with extreme price movements
    """
    dates = pd.date_range(start='2020-01-01', periods=20, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Open': [100.0] * 20,
        'High': [105.0] * 20,
        'Low': [95.0] * 20,
        'Close': [100.0] * 20,
        'Volume': [1000000] * 20
    })

    # Introduce extreme volatility
    data.loc[10, 'High'] = 150.0
    data.loc[10, 'Low'] = 50.0
    data.loc[10, 'Close'] = 100.0

    return data.set_index('Date')


@pytest.fixture
def sample_multi_stock_data():
    """
    Fixture providing OHLCV data for multiple stocks.

    Returns:
        dict: Dictionary mapping stock IDs to OHLCV DataFrames
    """
    stocks = {}
    for stock_id in ['VCB', 'VIC', 'HPG']:
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
        base_price = np.random.uniform(50, 150)
        returns = np.random.normal(0, 0.02, 252)
        prices = base_price * (1 + returns).cumprod()

        data = pd.DataFrame({
            'Date': dates,
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 252)
        })

        # Ensure OHLCV consistency
        data['High'] = data[['Open', 'Close']].max(axis=1) * 1.01
        data['Low'] = data[['Open', 'Close']].min(axis=1) * 0.99

        stocks[stock_id] = data.set_index('Date')

    return stocks
