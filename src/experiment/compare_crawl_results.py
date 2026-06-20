"""
Compare crawl results between original and enhanced methods
Identify missing stocks and provide recommendations

Date: 2026-06-19
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_summaries(
    original_path: str,
    enhanced_path: str
) -> dict:
    """
    Compare original vs enhanced crawl results

    Args:
        original_path: Path to original summary CSV
        enhanced_path: Path to enhanced summary CSV

    Returns:
        Dictionary with comparison results
    """
    # Load summaries
    try:
        original_df = pd.read_csv(original_path)
        logger.info(f"Loaded original: {len(original_df)} stocks")
    except:
        logger.warning(f"Original summary not found: {original_path}")
        original_df = pd.DataFrame()

    try:
        enhanced_df = pd.read_csv(enhanced_path)
        logger.info(f"Loaded enhanced: {len(enhanced_df)} stocks")
    except:
        logger.error(f"Enhanced summary not found: {enhanced_path}")
        enhanced_df = pd.DataFrame()

    if original_df.empty and enhanced_df.empty:
        return {"error": "No data available"}

    # Compare
    original_stocks = set(original_df['symbol'].tolist()) if not original_df.empty else set()
    enhanced_stocks = set(enhanced_df['symbol'].tolist()) if not enhanced_df.empty else set()

    new_stocks = enhanced_stocks - original_stocks
    lost_stocks = original_stocks - enhanced_stocks
    common_stocks = original_stocks & enhanced_stocks

    results = {
        "original_count": len(original_stocks),
        "enhanced_count": len(enhanced_stocks),
        "new_stocks": list(new_stocks),
        "lost_stocks": list(lost_stocks),
        "common_stocks": list(common_stocks),
        "improvement": len(new_stocks) - len(lost_stocks)
    }

    return results


def analyze_quality(summary_path: str) -> dict:
    """
    Analyze data quality from summary file

    Args:
        summary_path: Path to summary CSV

    Returns:
        Dictionary with quality metrics
    """
    try:
        df = pd.read_csv(summary_path)

        return {
            "total_stocks": len(df),
            "avg_days": df['total_days'].mean(),
            "min_days": df['total_days'].min(),
            "max_days": df['total_days'].max(),
            "date_range": f"{df['start_date'].min()} to {df['end_date'].max()}",
            "low_data_stocks": len(df[df['total_days'] < 500])
        }
    except Exception as e:
        logger.error(f"Error analyzing {summary_path}: {e}")
        return {}


def print_comparison_report(dataset: str, results: dict, quality: dict):
    """Print comparison report for a dataset"""
    print(f"\n{'=' * 60}")
    print(f"{dataset.upper()} COMPARISON")
    print(f"{'=' * 60}")

    print(f"\n[STOCK COUNT]")
    print(f"Original: {results['original_count']}")
    print(f"Enhanced: {results['enhanced_count']}")
    print(f"Improvement: {results['improvement']:+d}")

    if results['new_stocks']:
        print(f"\n[NEW STOCKS] ({len(results['new_stocks'])})")
        for stock in sorted(results['new_stocks']):
            print(f"  + {stock}")

    if results['lost_stocks']:
        print(f"\n[LOST STOCKS] ({len(results['lost_stocks'])})")
        for stock in sorted(results['lost_stocks']):
            print(f"  - {stock}")

    if quality:
        print(f"\n[DATA QUALITY]")
        print(f"Total stocks: {quality['total_stocks']}")
        print(f"Avg days: {quality['avg_days']:.0f}")
        print(f"Min days: {quality['min_days']}")
        print(f"Max days: {quality['max_days']}")
        print(f"Date range: {quality['date_range']}")
        print(f"Low data stocks (<500 days): {quality['low_data_stocks']}")


def main():
    """Main execution"""
    print("=" * 60)
    print("CRAWL RESULTS COMPARISON")
    print("=" * 60)

    datasets = [
        ("VN30", "data/raw/vn30", "data/raw/vn30_enhanced"),
        ("VN100", "data/raw/vn100", "data/raw/vn100_enhanced"),
        ("HNX", "data/raw/hnx", "data/raw/hnx_enhanced")
    ]

    overall_results = {
        "original_total": 0,
        "enhanced_total": 0,
        "total_improvement": 0
    }

    for dataset, original_path, enhanced_path in datasets:
        # Compare
        original_summary = f"{original_path}/stock_summary.csv"
        enhanced_summary = f"{enhanced_path}/stock_summary.csv"

        results = compare_summaries(original_summary, enhanced_summary)
        quality = analyze_quality(enhanced_summary)

        print_comparison_report(dataset, results, quality)

        overall_results["original_total"] += results["original_count"]
        overall_results["enhanced_total"] += results["enhanced_count"]
        overall_results["total_improvement"] += results["improvement"]

    # Overall summary
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n[TOTAL STOCKS]")
    print(f"Original: {overall_results['original_total']}")
    print(f"Enhanced: {overall_results['enhanced_total']}")
    print(f"Improvement: {overall_results['total_improvement']:+d}")

    print(f"\n[SUCCESS RATE]")
    original_rate = (overall_results['original_total'] / 230) * 100  # Assuming 230 total target
    enhanced_rate = (overall_results['enhanced_total'] / 230) * 100
    print(f"Original: {original_rate:.1f}%")
    print(f"Enhanced: {enhanced_rate:.1f}%")
    print(f"Improvement: {enhanced_rate - original_rate:+.1f}%")

    print(f"\n[CONCLUSION]")
    if overall_results['total_improvement'] > 0:
        print("[SUCCESS] Enhanced crawler improved results!")
        print(f"[+] Gained {overall_results['total_improvement']} additional stocks")
    elif overall_results['total_improvement'] == 0:
        print("[INFO] Enhanced crawler maintained same results")
        print("[+] But with better error handling and retry logic")
    else:
        print("[WARNING] Enhanced crawler performed worse")
        print(f"[-] Lost {abs(overall_results['total_improvement'])} stocks")

    print(f"\n[RECOMMENDATIONS]")
    if overall_results['enhanced_total'] >= 200:
        print("[OK] Current data is sufficient for training")
        print("[+] Proceed with model training")
    else:
        print("[ACTION] Consider implementing web scrapers")
        print("[+] Could add 10-15% more stocks")


if __name__ == "__main__":
    main()
