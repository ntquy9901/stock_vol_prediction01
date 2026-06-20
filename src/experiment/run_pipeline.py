"""
Run complete volatility prediction pipeline on all available stocks.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import (
    list_available_data,
    validate_data_directory,
    run_complete_pipeline,
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR
)


def main():
    print("=" * 80)
    print("STOCK VOLATILITY PREDICTION - RUNNING ON REAL DATA")
    print("=" * 80)

    # Validate data directory
    print("\nStep 1: Validating data directory...")
    data_dir = Path('data/raw/prices')  # Update to actual data location
    results_dir = Path('results')

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        print("Please ensure OHLCV CSV files are in the data directory")
        return

    # List available stocks
    print(f"\nStep 2: Discovering available stocks in {data_dir}...")
    available_tickers = list_available_data(data_dir=data_dir)

    if not available_tickers:
        print("[ERROR] No CSV files found in data directory")
        return

    print(f"[OK] Found {len(available_tickers)} stocks: {', '.join(available_tickers[:10])}{'...' if len(available_tickers) > 10 else ''}")

    # Run complete pipeline
    print(f"\nStep 3: Running complete pipeline on {len(available_tickers)} stocks...")
    print(f"   Data directory: {data_dir}")
    print(f"   Results directory: {results_dir}")
    print(f"   Processing... (this may take several minutes)")

    try:
        # Run pipeline on all available stocks
        all_tickers = available_tickers  # Process all stocks
        print(f"\n[INFO] RUNNING IN PRODUCTION MODE: Processing all {len(all_tickers)} stocks")
        print(f"   Tickers: {', '.join(all_tickers)}")

        final_results = run_complete_pipeline(
            tickers=all_tickers,
            data_dir=data_dir,
            results_dir=results_dir,
            save_models=True,
            save_plots=True
        )

        # Print final summary
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        summary_stats = final_results['summary_stats']
        print(f"\nAGGREGATE PERFORMANCE ({len(all_tickers)} stocks):")
        print(f"   Mean QLIKE Loss: {summary_stats['mean_qlike_loss']:.6f} (lower is better)")
        print(f"   Mean Directional Accuracy: {summary_stats['mean_directional_accuracy']*100:.1f}% (target: >55%)")
        print(f"   Mean Theil's U: {summary_stats['mean_theil_u']:.3f} (target: <1.0)")
        print(f"   Stocks beating random walk: {summary_stats['stocks_beating_rw']}/{len(all_tickers)}")
        print(f"   Stocks with Dir Acc >= 55%: {summary_stats['stocks_with_dir_acc_55']}/{len(all_tickers)}")

        print(f"\nResults saved to: {results_dir}")
        print(f"   - Summary report: summary_report.csv")
        print(f"   - Aggregate results: aggregate_results.json")
        print(f"   - Individual stock results: results/<TICKER>/")

        print("\n" + "=" * 80)
        print("SUCCESS! Check the results directory for detailed outputs")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
