"""
Market Data Loader for Global Benchmark Datasets

Downloads and caches market indicators (VIX, treasury rates, indices)
for use with stock volatility forecasting.

Author: Stock Volatility Prediction Team
Date: 2026-08-01
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Bootstrap path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


MARKET_INDICATORS = {
    "sp500": {
        "vix": "^VIX",
        "treasury_10y": "^TNX",
        "sp500_index": "^GSPC",
    },
    "vn30": {
        "vn_index": "^VNINDEX",
        "usd_vnd": "VND=X",
    },
}


def download_market_data(
    market: str = "sp500",
    start_date: str = "2011-01-01",
    end_date: str = "2026-12-31",
    output_dir: str = None,
):
    """
    Download market indicators via yfinance and save to CSV.

    Args:
        market: Market identifier ("sp500" or "vn30")
        start_date: Start date for data download
        end_date: End date for data download
        output_dir: Directory to save CSV files
    """
    import yfinance as yf

    if output_dir is None:
        output_dir = os.path.join(project_root, "data", "raw", "market_data", market)

    os.makedirs(output_dir, exist_ok=True)

    indicators = MARKET_INDICATORS.get(market, {})
    if not indicators:
        raise ValueError(f"Unknown market: '{market}'. Available: {list(MARKET_INDICATORS.keys())}")

    print(f"[INFO] Downloading market data for '{market}' ({start_date} to {end_date})")

    for name, ticker in indicators.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                print(f"[SKIP] {name} ({ticker}): No data returned")
                continue

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()

            # Ensure Date column is string
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            elif "date" in df.columns:
                df["Date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df = df.drop(columns=["date"])

            # Keep only Date and Close
            if "Close" in df.columns:
                save_df = df[["Date", "Close"]].copy()
            elif "close" in df.columns:
                save_df = df[["Date", "close"]].copy()
                save_df = save_df.rename(columns={"close": "Close"})
            else:
                save_df = df

            output_file = os.path.join(output_dir, f"{name}.csv")
            save_df.to_csv(output_file, index=False)
            print(f"[OK] {name}: {len(save_df):,} rows -> {output_file}")

        except Exception as e:
            print(f"[ERROR] {name} ({ticker}): {e}")

    print(f"\n[SUCCESS] Market data saved to {output_dir}")


def load_market_data(market: str = "sp500", data_dir: str = None) -> pd.DataFrame:
    """
    Load all market indicators for a market into a single merged DataFrame.

    Args:
        market: Market identifier
        data_dir: Directory containing market data CSVs

    Returns:
        DataFrame with Date index and columns for each indicator
    """
    if data_dir is None:
        data_dir = os.path.join(project_root, "data", "raw", "market_data", market)

    indicators = MARKET_INDICATORS.get(market, {})
    frames = []

    for name in indicators.keys():
        csv_path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.isfile(csv_path):
            print(f"[WARN] {name}.csv not found, skipping")
            continue

        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.set_index("Date")
        df = df.rename(columns={"Close": name})
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No market data files found in {data_dir}")

    # Merge all indicators on Date index
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.join(frame, how="outer")

    merged = merged.sort_index()
    merged = merged.ffill()  # Forward fill missing dates

    return merged


def merge_with_stock_data(stock_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge market indicators with stock data.

    Args:
        stock_df: Stock DataFrame with 'Date' column
        market_df: Market indicators DataFrame with DatetimeIndex

    Returns:
        Stock DataFrame with added market indicator columns
    """
    stock_df = stock_df.copy()
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])

    merged = stock_df.merge(
        market_df.reset_index(),
        on="Date",
        how="left",
    )

    # Forward fill any missing market data
    market_cols = [c for c in merged.columns if c not in stock_df.columns]
    merged[market_cols] = merged[market_cols].ffill()

    return merged
