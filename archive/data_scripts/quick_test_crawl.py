"""
Quick test crawler - Download sample data
This is a minimal script to test data crawling with a small sample

Usage:
    python src/data/quick_test_crawl.py
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample stocks to test (small subset)
TEST_STOCKS = [
    # VN30 stocks (5 stocks for quick test)
    "ACB", "BID", "VCB", "VNM", "FPT",

    # Additional VN100 candidates (5 stocks)
    "MWG", "HPG", "MSN", "VIC", "GAS",

    # HNX candidates (3 stocks)
    "VCC", "MPC", "KLS"
]

VIETNAM_SUFFIX = ".VN"


def crawl_single_stock(symbol: str, start_date: str = "2020-01-01") -> pd.DataFrame:
    """Crawl single stock data"""
    try:
        # Add .VN suffix
        symbol_with_suffix = f"{symbol}{VIETNAM_SUFFIX}"

        logger.info(f"Downloading {symbol}...")
        ticker = yf.Ticker(symbol_with_suffix)
        data = ticker.history(start=start_date, interval="1d")

        if data.empty:
            logger.warning(f"  [X] No data for {symbol}")
            return None

        # Format data
        data = data.reset_index()
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'splits']
        data = data[['date', 'open', 'high', 'low', 'close', 'volume']]

        logger.info(f"  [OK] Downloaded {len(data)} rows for {symbol}")
        return data

    except Exception as e:
        logger.error(f"  [ERROR] Error downloading {symbol}: {e}")
        return None


def main():
    """Main execution"""
    print("=" * 60)
    print("Quick Test Crawler - Vietnam Stocks")
    print("=" * 60)

    # Create output directory
    output_dir = Path("data/raw/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(TEST_STOCKS)} test stocks to: {output_dir}")
    print("-" * 60)

    success_count = 0
    fail_count = 0

    for symbol in TEST_STOCKS:
        data = crawl_single_stock(symbol, start_date="2020-01-01")

        if data is not None:
            # Save to CSV
            filename = output_dir / f"{symbol}_ohlcv.csv"
            data.to_csv(filename, index=False)
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total attempted: {len(TEST_STOCKS)}")
    print(f"[OK] Successful: {success_count}")
    print(f"[FAIL] Failed: {fail_count}")
    print(f"Output directory: {output_dir}")

    if success_count > 0:
        print("\n[SUCCESS] Test data ready!")
        print("\nNext steps:")
        print("1. Check downloaded data:")
        print(f"   ls {output_dir}")
        print("\n2. If test successful, download full datasets:")
        print("   python src/data/crawl_vietnam_stocks.py")
    else:
        print("\n[WARNING] All downloads failed!")
        print("\nPossible issues:")
        print("1. No internet connection")
        print("2. Yahoo Finance blocking requests")
        print("3. Stocks not available on Yahoo Finance")
        print("\nSolutions:")
        print("1. Try again later")
        print("2. Use Vietnamese data sources (see docs/DATA_EXPANSION_README.md)")
        print("3. Use existing vn30 data in data/raw/prices/")


if __name__ == "__main__":
    main()
