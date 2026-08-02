"""
Main Parkinson Data Processing Pipeline

This script processes raw OHLCV data to Parkinson volatility
and saves it to the processed directory for all 30 stocks.

Usage from project root:
    python -m src.common.process_parkinson_pipeline

Or from src/common directory:
    python process_parkinson_pipeline.py

Author: Stock Volatility Prediction Team
Date: 2026-06-17
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.common.parkinson_utils import process_single_stock


def process_all_stocks(raw_dir: str, output_dir: str, tickers: list = None):
    """
    Process stocks from raw OHLCV to Parkinson volatility.

    Args:
        raw_dir: Directory with raw per-ticker OHLCV CSVs.
        output_dir: Directory to save processed CSVs.
        tickers: Optional list of ticker symbols to restrict processing to
            (matched against the raw filename stem, e.g. "AAPL.csv" -> "AAPL").
            Default: process every CSV found in raw_dir.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all raw OHLCV files
    raw_files = [f for f in os.listdir(raw_dir)
                 if f.endswith('_ohlcv.csv') or f.endswith('.csv')]

    if tickers:
        ticker_set = set(tickers)
        raw_files = [f for f in raw_files if os.path.splitext(f)[0] in ticker_set]

    print(f"Found {len(raw_files)} raw files in {raw_dir}")
    print("=" * 80)

    # Process each stock
    results = []
    for raw_file in raw_files:
        raw_path = os.path.join(raw_dir, raw_file)
        ticker, num_records = process_single_stock(raw_path, output_dir)

        if num_records > 0:
            results.append({
                'ticker': ticker,
                'num_records': num_records
            })
            print(f"[OK] {ticker}: {num_records} records")
        else:
            print(f"[SKIP] {ticker}: Skipped (error or no data)")

    # Create summary DataFrame
    summary_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Processed {len(summary_df)} stocks successfully")
    print(f"[TOTAL] Total records: {summary_df['num_records'].sum()}")
    print(f"[DIR] Output directory: {output_dir}")

    if len(summary_df) > 0:
        print("\n[STATS] Summary Statistics:")
        print(summary_df.describe())

        # Save summary
        summary_file = os.path.join(output_dir, 'processing_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\n[SAVE] Summary saved to: {summary_file}")

    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


import argparse

MARKET_PATHS = {
    "vn30": {
        "raw": os.path.join(project_root, 'data/raw/prices'),
        "processed": os.path.join(project_root, 'data/processed'),
    },
    "sp500": {
        "raw": os.path.join(project_root, 'data/raw/prices_sp500'),
        "processed": os.path.join(project_root, 'data/processed_sp500'),
    },
}


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Process OHLCV data to Parkinson volatility")
    parser.add_argument(
        "--market", default="vn30", choices=["vn30", "sp500"],
        help="Market to process (default: vn30)"
    )
    parser.add_argument("--raw_dir", default=None, help="Override raw data directory")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--tickers", nargs="+", default=None,
                         help="Restrict processing to these tickers (default: all files in raw_dir)")
    args = parser.parse_args()

    paths = MARKET_PATHS[args.market]
    raw_dir = args.raw_dir or paths["raw"]
    output_dir = args.output_dir or paths["processed"]

    print("=" * 80)
    print(f"PROCESS RAW OHLCV DATA TO PARKINSON VOLATILITY (market={args.market})")
    print("=" * 80)

    # Check if raw directory exists
    if not os.path.exists(raw_dir):
        print(f"[ERROR] Raw data directory not found: {raw_dir}")
        print(f"Please ensure raw OHLCV files are in {raw_dir}/")
        return

    # Process stocks (all, or restricted to --tickers)
    process_all_stocks(raw_dir, output_dir, tickers=args.tickers)


if __name__ == "__main__":
    main()
