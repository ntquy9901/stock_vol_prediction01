"""
Process All Available Data to Parkinson Volatility
Process 210 stocks from data/raw/all_available to data/processed_all

Usage:
    python process_all_available.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from common.process_parkinson_pipeline import process_all_stocks


def main():
    """Process all available stocks."""
    print("=" * 80)
    "PROCESS ALL AVAILABLE DATA (210 STOCKS)"
    print("=" * 80)

    # Define directories
    project_root = Path(__file__).parent
    raw_dir = project_root / 'data' / 'raw' / 'all_available'
    output_dir = project_root / 'data' / 'processed_all'

    print(f"\n[CONFIG]")
    print(f"Raw directory: {raw_dir}")
    print(f"Output directory: {output_dir}")

    # Check raw directory exists
    if not raw_dir.exists():
        print(f"[ERROR] Raw data directory not found: {raw_dir}")
        print("Please run data combination first:")
        print("  python src/data/combine_datasets.py --sources vn30 vn100 hnx --output all_available")
        return

    # Count raw files
    raw_files = list(raw_dir.glob('*_ohlcv.csv'))
    print(f"[INFO] Found {len(raw_files)} raw OHLCV files")

    if len(raw_files) == 0:
        print("[ERROR] No OHLCV files found in raw directory")
        return

    # Process all stocks
    print("\n[START] Processing...")
    process_all_stocks(str(raw_dir), str(output_dir))

    # Count processed files
    processed_files = list(output_dir.glob('*_processed.csv'))
    print(f"\n[FINAL RESULT]")
    print(f"Processed: {len(processed_files)} stocks")
    print(f"Output: {output_dir}")
    print("\n[READY] You can now train with:")
    print(f"  python src/lstm_har_enhanced/train_with_validation.py --data_dir data/processed_all")


if __name__ == "__main__":
    main()
