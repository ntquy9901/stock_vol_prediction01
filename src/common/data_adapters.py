"""
Data Adapters for Global Benchmark Datasets

Converts various global dataset formats to VN30-compatible format
for use with existing processing pipeline.

Author: Stock Volatility Prediction Team
Date: 2026-08-01
"""

import pandas as pd
from typing import Dict


ADAPTERS = {
    "stocks_ohlcv": {
        "Date": lambda df: pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
        "Open": lambda df: df["open"],
        "High": lambda df: df["high"],
        "Low": lambda df: df["low"],
        "Close": lambda df: df["close"],
        "Volume": lambda df: df["volume"],
    },
    "fnspid": {
        "Date": lambda df: pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
        "Open": lambda df: df["open"],
        "High": lambda df: df["high"],
        "Low": lambda df: df["low"],
        "Close": lambda df: df["close"],
        "Volume": lambda df: df["volume"],
    },
}


def adapt_to_vn30_format(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Convert global dataset to VN30-compatible format.

    Args:
        df: Input DataFrame from global dataset
        source: Dataset source identifier (e.g., "stocks_ohlcv", "fnspid")

    Returns:
        DataFrame with VN30 columns: Date, Open, High, Low, Close, Volume

    Raises:
        ValueError: If source is not recognized
    """
    if source not in ADAPTERS:
        raise ValueError(
            f"Unknown source: '{source}'. "
            f"Available sources: {list(ADAPTERS.keys())}"
        )

    adapter = ADAPTERS[source]
    result = pd.DataFrame()
    for col, transform in adapter.items():
        result[col] = transform(df)

    return result


def split_by_ticker(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split multi-stock DataFrame into per-ticker VN30-format DataFrames.

    Args:
        df: Input DataFrame with 'act_symbol' column and HF format

    Returns:
        Dict mapping ticker symbol to VN30-format DataFrame
    """
    adapted = adapt_to_vn30_format(df, source="stocks_ohlcv")

    tickers = df["act_symbol"].unique()
    result = {}
    for ticker in tickers:
        mask = df["act_symbol"] == ticker
        result[ticker] = adapted[mask].reset_index(drop=True)

    return result
