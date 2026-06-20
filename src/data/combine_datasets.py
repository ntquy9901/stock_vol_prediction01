"""
Combine multiple stock datasets
Combine VN30, VN100, HNX into single datasets

Usage:
    python src/data/combine_datasets.py --sources vn30 vn100 --output combined
    python src/data/combine_datasets.py --sources vn30 vn100 hnx --output all
"""

import argparse
from pathlib import Path
import pandas as pd
import shutil
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def combine_datasets(
    source_dirs: List[str],
    output_dir: str,
    base_path: str = "data/raw"
):
    """
    Combine stock data from multiple directories

    Args:
        source_dirs: List of source directory names
        output_dir: Output directory name
        base_path: Base path for all directories
    """
    base_path = Path(base_path)
    output_path = base_path / output_dir

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Combining data to: {output_path}")
    print("=" * 60)

    seen_symbols = set()
    total_files = 0
    duplicate_count = 0

    # Process each source directory
    for source_dir in source_dirs:
        source_path = base_path / source_dir

        if not source_path.exists():
            print(f"⚠️  Source directory not found: {source_path}")
            continue

        # Find all CSV files
        csv_files = list(source_path.glob("*_ohlcv.csv"))
        print(f"\n[DIR] Processing: {source_dir} ({len(csv_files)} files)")

        # Copy files to output directory
        for csv_file in csv_files:
            symbol = csv_file.stem.replace("_ohlcv", "")
            output_file = output_path / csv_file.name

            # Check for duplicates
            if output_file.exists():
                duplicate_count += 1
                logger.debug(f"Duplicate skipped: {symbol}")
                continue

            # Copy file
            shutil.copy2(csv_file, output_file)
            seen_symbols.add(symbol)
            total_files += 1

        print(f"   [OK] Copied {len(csv_files)} files")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Source directories: {len(source_dirs)}")
    print(f"Total files copied: {total_files}")
    print(f"Unique stocks: {len(seen_symbols)}")
    print(f"Duplicates skipped: {duplicate_count}")
    print(f"Output directory: {output_path}")

    # Create summary file
    create_summary(output_path)

    print("\n[SUCCESS] Dataset combination complete!")


def create_summary(output_dir: Path):
    """Create summary file for combined dataset"""
    csv_files = list(output_dir.glob("*_ohlcv.csv"))

    summary_data = []
    for csv_file in csv_files:
        symbol = csv_file.stem.replace("_ohlcv", "")
        try:
            df = pd.read_csv(csv_file)
            summary_data.append({
                "symbol": symbol,
                "start_date": df['date'].min(),
                "end_date": df['date'].max(),
                "total_days": len(df),
                "file": str(csv_file)
            })
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / "stock_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n[SUMMARY] Summary saved: {summary_file}")

        # Print summary statistics
        print("\n[STATS] Dataset Statistics:")
        print(f"   Total stocks: {len(summary_df)}")
        print(f"   Average days per stock: {summary_df['total_days'].mean():.0f}")
        print(f"   Min start date: {summary_df['start_date'].min()}")
        print(f"   Max end date: {summary_df['end_date'].max()}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Combine stock datasets")
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        required=True,
        help="Source directories (e.g., vn30 vn100 hnx)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory name (e.g., combined, all)"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default="data/raw",
        help="Base path for data directories"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Stock Dataset Combiner")
    print("=" * 60)

    combine_datasets(
        source_dirs=args.sources,
        output_dir=args.output,
        base_path=args.base_path
    )


if __name__ == "__main__":
    main()
