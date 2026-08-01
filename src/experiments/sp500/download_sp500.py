"""
Download S&P 500 OHLCV data from Hugging Face and convert to VN30 format.

Usage:
    python src/experiments/sp500/download_sp500.py
    python src/experiments/sp500/download_sp500.py --tickers AAPL MSFT GOOGL
    python src/experiments/sp500/download_sp500.py --output_dir data/raw/prices_sp500
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Bootstrap path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from src.common.data_adapters import adapt_to_vn30_format


# S&P 500 tickers (as of 2024)
SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "XOM", "HD", "CVX", "MA", "MRK", "ABBV", "PEP",
    "COST", "AVGO", "KO", "LLY", "WMT", "MCD", "TMO", "CSCO", "ACN", "ABT",
    "CRM", "DHR", "VZ", "ADBE", "NKE", "TXN", "NEE", "BMY", "PM", "RTX",
    "QCOM", "UNP", "ORCL", "HON", "INTU", "AMD", "AMGN", "LOW", "SBUX", "ISRG",
    "AMAT", "BKNG", "ELV", "T", "ADI", "PFE", "GILD", "MU", "MDLZ", "REGN",
    "SYK", "LRCX", "CVS", "PLTR", "BLK", "TMUS", "SPGI", "ZTS", "AXP", "DE",
    "MMM", "C", "NOW", "PYPL", "TJX", "VRTX", "MO", "SCHW", "CB", "GE",
    "SO", "BSX", "DUK", "EOG", "BDX", "ITW", "CI", "SLB", "MMC", "PGR",
    "FIS", "AON", "CL", "APD", "SHW", "CME", "ICE", "NSC", "EQIX", "FCX",
    "GD", "EMR", "MCO", "USB", "PNC", "WM", "HUM", "TGT", "F", "GM",
    "NOC", "ECL", "KLAC", "SNPS", "CDNS", "ADSK", "MCHP", "NXPI", "MNST", "KMB",
    "CTAS", "ABNB", "MRVL", "CRWD", "MAR", "ROP", "AEP", "PCG", "AZO", "ORLY",
    "CMG", "WELL", "SPG", "PSA", "O", "DLR", "AMT", "CCI", "SBAC", "EQT",
    "COP", "PXD", "DVN", "HAL", "MPC", "VLO", "PSX", "OXY", "HES", "BKR",
    "LMT", "BA", "CAT", "DE", "EMR", "ETN", "HON", "IR", "ITW", "MMM",
    "ROK", "PH", "CMI", "DOV", "FLS", "GNRC", "IEX", "JCI", "NDSN", "OTIS",
    "PNR", "RSG", "TT", "XYL", "A", "DHR", "IDXX", "IQV", "LH", "MTD",
    "RMD", "STE", "SYK", "TMO", "WAT", "ZBH", "ALGN", "BAX", "BIIB", "BSX",
    "DXCM", "EW", "HOLX", "IDXX", "ILMN", "INCY", "ISRG", "LH", "MRNA", "REGN",
    "TECH", "VRTX", "WAT", "XRAY", "ZBH", "ZTS", "A", "ABT", "ALGN", "AMGN",
    "BAX", "BDX", "BIIB", "BIO", "BSX", "CAH", "CI", "CNC", "COO", "CVS",
    "DGX", "DHR", "DOV", "DXCM", "ELV", "EW", "GILD", "HCA", "HOLX", "HUM",
    "IDXX", "ILMN", "INCY", "IQV", "ISRG", "JNJ", "LH", "LLY", "MCK", "MDT",
    "MRK", "MRNA", "MTD", "PFE", "REGN", "RMD", "STE", "SYK", "TECH", "TMO",
    "UNH", "UHS", "VRTX", "WAT", "WST", "XRAY", "ZBH", "ZTS",
]


def download_and_save(
    output_dir: str = None,
    tickers: list = None,
    hf_dataset: str = "siddharthmb/stocks-ohlcv",
):
    """
    Download S&P 500 OHLCV from Hugging Face, convert to VN30 format, save per-stock CSVs.

    Args:
        output_dir: Directory to save CSV files (default: data/raw/prices_sp500)
        tickers: List of ticker symbols to download (default: all SP500_TICKERS)
        hf_dataset: Hugging Face dataset name
    """
    from datasets import load_dataset

    if output_dir is None:
        output_dir = os.path.join(project_root, "data", "raw", "prices_sp500")

    if tickers is None:
        tickers = SP500_TICKERS

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading dataset: {hf_dataset}")
    ds = load_dataset(hf_dataset, split="train")
    df = ds.to_pandas()
    print(f"[INFO] Loaded {len(df):,} rows")

    # Filter to requested tickers
    df_filtered = df[df["act_symbol"].isin(tickers)]
    print(f"[INFO] Filtered to {len(tickers)} tickers: {len(df_filtered):,} rows")

    # Convert to VN30 format and save per-stock
    saved = 0
    for ticker in tickers:
        ticker_df = df_filtered[df_filtered["act_symbol"] == ticker]
        if len(ticker_df) == 0:
            print(f"[SKIP] {ticker}: No data found")
            continue

        vn30_df = adapt_to_vn30_format(ticker_df, source="stocks_ohlcv")
        vn30_df = vn30_df.sort_values("Date").reset_index(drop=True)

        output_file = os.path.join(output_dir, f"{ticker}.csv")
        vn30_df.to_csv(output_file, index=False)
        saved += 1
        print(f"[OK] {ticker}: {len(vn30_df):,} rows -> {output_file}")

    print(f"\n[SUCCESS] Saved {saved}/{len(tickers)} tickers to {output_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Download S&P 500 OHLCV data")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="List of ticker symbols (default: all S&P 500)"
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Output directory (default: data/raw/prices_sp500)"
    )
    parser.add_argument(
        "--hf_dataset", default="siddharthmb/stocks-ohlcv",
        help="Hugging Face dataset name"
    )
    args = parser.parse_args()

    download_and_save(
        output_dir=args.output_dir,
        tickers=args.tickers,
        hf_dataset=args.hf_dataset,
    )


if __name__ == "__main__":
    main()
