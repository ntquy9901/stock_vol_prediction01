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


def process_all_stocks(raw_dir: str, output_dir: str):
    """
    Process all stocks from raw OHLCV to Parkinson volatility.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all raw OHLCV files
    raw_files = [f for f in os.listdir(raw_dir)
                 if f.endswith('_ohlcv.csv') or f.endswith('.csv')]

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


def main():
    """Main execution function."""
    print("=" * 80)
    print("PROCESS RAW OHLCV DATA TO PARKINSON VOLATILITY")
    print("=" * 80)

    # Define directories (relative to project root)
    raw_dir = os.path.join(project_root, 'data/raw/prices')
    output_dir = os.path.join(project_root, 'data/processed')

    # Check if raw directory exists
    if not os.path.exists(raw_dir):
        print(f"[ERROR] Raw data directory not found: {raw_dir}")
        print("Please ensure raw OHLCV files are in data/raw/prices/")
        return

    # Process all stocks
    process_all_stocks(raw_dir, output_dir)


if __name__ == "__main__":
    main()
