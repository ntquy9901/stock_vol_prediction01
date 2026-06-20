"""
Integration Pipeline for Stock Volatility Prediction System.

This module orchestrates the complete end-to-end workflow:
    1. Load OHLCV data (from CSV files)
    2. Calculate Parkinson volatility
    3. Create HAR features and 5-day target
    4. Train HAR-R baseline model
    5. Evaluate performance (QLIKE, directional accuracy, Theil's U)
    6. Save results and visualizations

Primary Functions:
    - load_ohlcv_data: Load OHLCV data from CSV file
    - process_single_stock: Complete pipeline for one stock
    - process_multiple_stocks: Batch processing for multiple stocks
    - generate_summary_report: Aggregate performance across stocks
    - run_complete_pipeline: End-to-end execution

Pipeline Flow:
    OHLCV Data → Parkinson Vol → HAR Features → HAR-R Model → Evaluation → Results

Author: Stock Volatility Prediction Team
Date: 2026-06-15
"""

import logging
import warnings
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# Import all project modules
from src.data_processing import (
    calculate_parkinson_volatility,
    process_ohlcv_data
)
from src.feature_engineering import (
    create_featureset,
    get_har_feature_importance
)
from src.model_training import (
    set_random_seeds,
    complete_har_r_training_pipeline,
    get_feature_coefficients
)
from src.evaluation import (
    generate_evaluation_report
)


# Configure module-level logger
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Constants
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Default paths
DEFAULT_DATA_DIR = Path('data')
DEFAULT_RESULTS_DIR = Path('results')
DEFAULT_MODELS_DIR = Path('models')

# VN30 stock tickers (example subset)
VN30_TICKERS = [
    'VCB', 'VIC', 'VHM', 'HPG', 'MSN', 'VNM', 'GVR', 'VPB', 'STB', 'TCB',
    'MBB', 'ACB', 'SSB', 'HDB', 'VIB', 'VTB', 'CTG', 'BID', 'TPB', 'TCM',
    'FPT', 'MWG', 'PLX', 'POW', 'GAS', 'BVI', 'PNJ', 'KDH', 'VIC', 'HPG'
]


# ============================================================================
# Data Loading
# ============================================================================

def load_ohlcv_data(ticker: str,
                    data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file for a specific stock.

    Expected CSV format:
        Date, Open, High, Low, Close, Volume
        2024-01-01, 100.0, 102.0, 98.0, 101.0, 1000000
        ...

    Args:
        ticker: Stock ticker symbol (e.g., 'VCB')
        data_dir: Directory containing CSV files (default: 'data')

    Returns:
        pd.DataFrame: OHLCV data with Date as index

    Raises:
        FileNotFoundError: If CSV file not found
        ValueError: If CSV format is invalid

    Example:
        >>> data = load_ohlcv_data('VCB')
        >>> print(data.head())
                     Open   High    Low  Close    Volume
        Date
        2024-01-01  100.0  102.0  98.0  101.0   1000000
        2024-01-02  101.0  103.0  99.0  102.0   1100000

    Note:
        CSV files should be named: {ticker}.csv or {ticker}_ohlcv.csv
        Date column must be parsable by pandas
    """
    # Try possible file names
    possible_names = [
        f"{ticker}.csv",
        f"{ticker}_ohlcv.csv",
        f"{ticker.upper()}.csv",
        f"{ticker.upper()}_OHLCV.csv"
    ]

    ohlcv_file = None
    for name in possible_names:
        file_path = data_dir / name
        if file_path.exists():
            ohlcv_file = file_path
            break

    if ohlcv_file is None:
        raise FileNotFoundError(
            f"OHLCV data file not found for ticker {ticker} in {data_dir}. "
            f"Tried: {possible_names}"
        )

    # Load CSV
    try:
        df = pd.read_csv(ohlcv_file)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {ohlcv_file}: {e}")

    # Parse date column
    date_columns = ['Date', 'date', 'DATE', 'Timestamp', 'timestamp']
    date_col = None
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        # Assume first column is date
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

    # Standardize column names
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'open' in col_lower:
            column_mapping[col] = 'Open'
        elif 'high' in col_lower:
            column_mapping[col] = 'High'
        elif 'low' in col_lower:
            column_mapping[col] = 'Low'
        elif 'close' in col_lower or col_lower in ['adj close', 'adjclose']:
            column_mapping[col] = 'Close'
        elif 'volume' in col_lower:
            column_mapping[col] = 'Volume'

    df.rename(columns=column_mapping, inplace=True)

    # Validate required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Found columns: {df.columns.tolist()}"
        )

    logger.info(f"✅ Loaded OHLCV data for {ticker}: {len(df)} rows from {ohlcv_file}")

    return df


# ============================================================================
# Single Stock Processing
# ============================================================================

def process_single_stock(ticker: str,
                       data_dir: Path = DEFAULT_DATA_DIR,
                       results_dir: Path = DEFAULT_RESULTS_DIR,
                       save_model: bool = True,
                       save_plots: bool = True) -> Dict[str, Any]:
    """
    Complete pipeline for processing a single stock.

    Pipeline steps:
    1. Load OHLCV data
    2. Calculate Parkinson volatility
    3. Create HAR features and 5-day target
    4. Train HAR-R baseline model
    5. Evaluate performance
    6. Save results

    Args:
        ticker: Stock ticker symbol
        data_dir: Directory containing OHLCV CSV files
        results_dir: Directory to save results
        save_model: If True, save trained model
        save_plots: If True, generate performance plots

    Returns:
        Dict containing:
            - 'ticker': Stock symbol
            - 'data_info': Data statistics (rows, date range)
            - 'model': Trained HAR-R model
            - 'train_results': Training metrics
            - 'test_results': Test performance metrics
            - 'feature_coefficients': Model coefficients DataFrame
            - 'benchmark_comparison': vs random walk and historical mean
            - 'status': 'SUCCESS' or 'FAILED'
            - 'error': Error message if failed

    Example:
        >>> results = process_single_stock('VCB')
        >>> print(f"Test QLIKE: {results['test_results']['qlike_loss']:.6f}")
        Test QLIKE: 0.001234

    Note:
        This is the PRIMARY function for single-stock analysis.
        All intermediate results are saved for debugging.
    """
    logger.info(f"🚀 Starting pipeline for {ticker}")

    results = {
        'ticker': ticker,
        'status': 'PROCESSING'
    }

    try:
        # Step 1: Load OHLCV data
        logger.info(f"Step 1/6: Loading OHLCV data for {ticker}...")
        ohlcv_data = load_ohlcv_data(ticker, data_dir=data_dir)

        results['data_info'] = {
            'rows': len(ohlcv_data),
            'date_start': ohlcv_data.index.min().strftime('%Y-%m-%d'),
            'date_end': ohlcv_data.index.max().strftime('%Y-%m-%d')
        }

        # Step 2: Calculate Parkinson volatility
        logger.info(f"Step 2/6: Calculating Parkinson volatility...")
        volatility = calculate_parkinson_volatility(ohlcv_data)

        # Step 3: Create HAR features and target
        logger.info(f"Step 3/6: Creating HAR features and 5-day target...")
        featureset = create_featureset(volatility)

        # Step 4: Train HAR-R model
        logger.info(f"Step 4/6: Training HAR-R baseline model...")
        set_random_seeds(RANDOM_SEED)

        model_path = None
        if save_model:
            os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
            model_path = DEFAULT_MODELS_DIR / f"{ticker}_har_r_model.pkl"

        training_results = complete_har_r_training_pipeline(
            featureset,
            save_model=save_model,
            model_path=str(model_path) if model_path else None
        )

        # Step 5: Evaluate performance
        logger.info(f"Step 5/6: Evaluating model performance...")

        # Create stock-specific results directory (always create, even without plots)
        stock_results_dir = results_dir / ticker
        os.makedirs(stock_results_dir, exist_ok=True)

        evaluation_report = generate_evaluation_report(
            training_results['y_train'],
            training_results['train_predictions'],
            training_results['y_test'],
            training_results['test_predictions'],
            model_name=f"{ticker} HAR-R Baseline",
            save_dir=str(stock_results_dir) if save_plots else None
        )

        # Step 6: Compile results
        logger.info(f"Step 6/6: Compiling results...")

        # Extract feature coefficients
        feature_coefficients = get_feature_coefficients(training_results['model'])

        results.update({
            'model': training_results['model'],
            'train_results': training_results['train_metrics'],
            'test_results': evaluation_report['test_metrics'],
            'overfitting_results': training_results['overfitting_results'],
            'feature_coefficients': feature_coefficients.to_dict('records'),
            'benchmark_comparison': evaluation_report['benchmark_comparison'].to_dict('records'),
            'target_achievement': evaluation_report['target_achievement'],
            'status': 'SUCCESS',
            'model_path': str(model_path) if model_path else None,
            'plots_path': str(stock_results_dir) if save_plots else None
        })

        # Save results summary
        results_summary = {
            'ticker': ticker,
            'status': 'SUCCESS',
            'data_info': results['data_info'],
            'test_metrics': results['test_results'],
            'target_achievement': results['target_achievement'],
            'processed_at': datetime.now().isoformat()
        }

        # Convert numpy types to native Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        results_summary = convert_numpy_types(results_summary)

        summary_path = stock_results_dir / f"{ticker}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)

        logger.info(f"✅ Pipeline completed successfully for {ticker}")
        logger.info(f"   Test QLIKE: {results['test_results']['qlike_loss']:.6f}")
        logger.info(f"   Test Dir Acc: {results['test_results']['directional_accuracy']*100:.1f}%")
        logger.info(f"   Test Theil U: {results['test_results']['theil_u']:.3f}")

        return results

    except Exception as e:
        logger.error(f"❌ Pipeline failed for {ticker}: {e}")
        results['status'] = 'FAILED'
        results['error'] = str(e)
        return results


# ============================================================================
# Multiple Stocks Processing
# ============================================================================

def process_multiple_stocks(tickers: List[str],
                           data_dir: Path = DEFAULT_DATA_DIR,
                           results_dir: Path = DEFAULT_RESULTS_DIR,
                           save_models: bool = True,
                           save_plots: bool = True,
                           max_workers: int = 1) -> Dict[str, Any]:
    """
    Process multiple stocks with batch processing.

    Args:
        tickers: List of stock ticker symbols
        data_dir: Directory containing OHLCV CSV files
        results_dir: Directory to save results
        save_models: If True, save trained models
        save_plots: If True, generate performance plots
        max_workers: Number of parallel workers (default: 1 = sequential)

    Returns:
        Dict containing:
            - 'total_stocks': Total number of stocks
            - 'successful': Number of successfully processed stocks
            - 'failed': Number of failed stocks
            - 'results': Dict of ticker → results
            - 'summary': Aggregate performance statistics

    Example:
        >>> tickers = ['VCB', 'VIC', 'VHM']
        >>> batch_results = process_multiple_stocks(tickers)
        >>> print(f"Processed: {batch_results['successful']}/{batch_results['total_stocks']}")
        Processed: 3/3

    Note:
        Sequential processing (max_workers=1) is safer for memory usage.
        Set max_workers > 1 for parallel processing (experimental).
    """
    logger.info(f"🚀 Starting batch processing for {len(tickers)} stocks")

    batch_results = {
        'total_stocks': len(tickers),
        'successful': 0,
        'failed': 0,
        'results': {},
        'summary': {}
    }

    # Process each stock
    for ticker in tickers:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {ticker} ({tickers.index(ticker)+1}/{len(tickers)})")
        logger.info(f"{'='*80}")

        stock_results = process_single_stock(
            ticker,
            data_dir=data_dir,
            results_dir=results_dir,
            save_model=save_models,
            save_plots=save_plots
        )

        batch_results['results'][ticker] = stock_results

        if stock_results['status'] == 'SUCCESS':
            batch_results['successful'] += 1
        else:
            batch_results['failed'] += 1

    # Generate aggregate summary
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 BATCH PROCESSING SUMMARY")
    logger.info(f"{'='*80}")

    successful_results = {
        ticker: results
        for ticker, results in batch_results['results'].items()
        if results['status'] == 'SUCCESS'
    }

    if successful_results:
        # Calculate aggregate statistics
        qlike_losses = [r['test_results']['qlike_loss'] for r in successful_results.values()]
        dir_accs = [r['test_results']['directional_accuracy'] for r in successful_results.values()]
        theil_us = [r['test_results']['theil_u'] for r in successful_results.values()]

        batch_results['summary'] = {
            'mean_qlike_loss': np.mean(qlike_losses),
            'mean_directional_accuracy': np.mean(dir_accs),
            'mean_theil_u': np.mean(theil_us),
            'std_qlike_loss': np.std(qlike_losses),
            'stocks_beating_rw': sum(1 for u in theil_us if u < 1.0),
            'stocks_with_dir_acc_55': sum(1 for acc in dir_accs if acc >= 0.55)
        }

        logger.info(f"✅ Successfully processed: {batch_results['successful']}/{batch_results['total_stocks']} stocks")
        logger.info(f"   Mean QLIKE Loss: {batch_results['summary']['mean_qlike_loss']:.6f}")
        logger.info(f"   Mean Directional Accuracy: {batch_results['summary']['mean_directional_accuracy']*100:.1f}%")
        logger.info(f"   Mean Theil's U: {batch_results['summary']['mean_theil_u']:.3f}")
        logger.info(f"   Stocks beating random walk: {batch_results['summary']['stocks_beating_rw']}/{len(successful_results)}")
        logger.info(f"   Stocks with Dir Acc ≥ 55%: {batch_results['summary']['stocks_with_dir_acc_55']}/{len(successful_results)}")
    else:
        logger.warning(f"⚠️ No stocks processed successfully")

    logger.info(f"❌ Failed: {batch_results['failed']}/{batch_results['total_stocks']} stocks")

    return batch_results


# ============================================================================
# Summary Report Generation
# ============================================================================

def generate_summary_report(batch_results: Dict[str, Any],
                           output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Generate summary report DataFrame from batch processing results.

    Args:
        batch_results: Results from process_multiple_stocks
        output_path: Path to save summary CSV (optional)

    Returns:
        pd.DataFrame: Summary report with columns:
            - ticker: Stock symbol
            - status: SUCCESS or FAILED
            - data_rows: Number of data points
            - test_qlike_loss: Test set QLIKE loss
            - test_directional_accuracy: Test set directional accuracy
            - test_theil_u: Test set Theil's U statistic
            - test_rmse: Test set RMSE
            - test_mse: Test set MSE
            - test_r2: Test set R²
            - beats_random_walk: Boolean (Theil's U < 1.0)
            - meets_dir_acc_target: Boolean (Dir Acc ≥ 55%)

    Example:
        >>> batch_results = process_multiple_stocks(['VCB', 'VIC'])
        >>> summary_df = generate_summary_report(batch_results)
        >>> print(summary_df)
           ticker  status  test_qlike_loss  test_directional_accuracy  test_theil_u
        0    VCB  SUCCESS         0.001234                      0.667         0.857
        1    VIC  SUCCESS         0.001567                      0.556         0.923

    Note:
        Failed stocks have NaN for test metrics.
        Summary CSV is saved if output_path provided.
    """
    summary_data = []

    for ticker, results in batch_results['results'].items():
        row = {
            'ticker': ticker,
            'status': results['status']
        }

        if results['status'] == 'SUCCESS':
            row.update({
                'data_rows': results['data_info']['rows'],
                'date_start': results['data_info']['date_start'],
                'date_end': results['data_info']['date_end'],
                'test_qlike_loss': results['test_results']['qlike_loss'],
                'test_directional_accuracy': results['test_results']['directional_accuracy'],
                'test_theil_u': results['test_results']['theil_u'],
                'test_rmse': results['test_results']['rmse'],
                'test_mse': results['test_results']['mse'],
                'test_r2': results['test_results']['r2'],
                'beats_random_walk': results['test_results']['theil_u'] < 1.0,
                'meets_dir_acc_target': results['test_results']['directional_accuracy'] >= 0.55,
                'meets_rmse_target': results['test_results']['rmse'] <= 0.20
            })
        else:
            row['error'] = results.get('error', 'Unknown error')

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV if path provided
    if output_path:
        summary_df.to_csv(output_path, index=False)
        logger.info(f"✅ Summary report saved to {output_path}")

    return summary_df


# ============================================================================
# Complete Pipeline Execution
# ============================================================================

def run_complete_pipeline(tickers: List[str],
                        data_dir: Path = DEFAULT_DATA_DIR,
                        results_dir: Path = DEFAULT_RESULTS_DIR,
                        save_models: bool = True,
                        save_plots: bool = True) -> Dict[str, Any]:
    """
    Run complete end-to-end pipeline for multiple stocks.

    This is the MAIN entry point for the volatility prediction system.

    Pipeline:
    1. Process all stocks (load → train → evaluate)
    2. Generate aggregate summary report
    3. Save results and visualizations

    Args:
        tickers: List of stock ticker symbols to process
        data_dir: Directory containing OHLCV CSV files
        results_dir: Directory to save all results
        save_models: If True, save trained models
        save_plots: If True, generate performance plots

    Returns:
        Dict containing:
            - 'batch_results': Full results from process_multiple_stocks
            - 'summary_df': Summary report DataFrame
            - 'summary_stats': Aggregate statistics

    Example:
        >>> tickers = ['VCB', 'VIC', 'VHM']
        >>> final_results = run_complete_pipeline(tickers)
        >>> print(final_results['summary_stats'])
        {'mean_qlike_loss': 0.001456, 'successful_stocks': 3, ...}

    Note:
        This function orchestrates the complete workflow.
        All results are saved to results_dir for further analysis.
    """
    logger.info("=" * 80)
    logger.info("🚀 STOCK VOLATILITY PREDICTION - COMPLETE PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  - Stocks: {len(tickers)} tickers")
    logger.info(f"  - Data directory: {data_dir}")
    logger.info(f"  - Results directory: {results_dir}")
    logger.info(f"  - Save models: {save_models}")
    logger.info(f"  - Save plots: {save_plots}")
    logger.info("=" * 80)

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Process all stocks
    logger.info("\n📊 PHASE 1: PROCESSING STOCKS")
    batch_results = process_multiple_stocks(
        tickers,
        data_dir=data_dir,
        results_dir=results_dir,
        save_models=save_models,
        save_plots=save_plots
    )

    # Step 2: Generate summary report
    logger.info("\n📊 PHASE 2: GENERATING SUMMARY REPORT")
    summary_df = generate_summary_report(
        batch_results,
        output_path=results_dir / 'summary_report.csv'
    )

    # Step 3: Save aggregate statistics
    logger.info("\n📊 PHASE 3: SAVING AGGREGATE RESULTS")
    aggregate_results = {
        'batch_results': batch_results,
        'summary_df': summary_df.to_dict('records'),
        'summary_stats': batch_results.get('summary', {}),
        'processed_at': datetime.now().isoformat()
    }

    # Save aggregate results as JSON
    aggregate_path = results_dir / 'aggregate_results.json'
    with open(aggregate_path, 'w') as f:
        json.dump(aggregate_results, f, indent=2, default=str)

    logger.info(f"✅ Aggregate results saved to {aggregate_path}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"  - Summary report: summary_report.csv")
    logger.info(f"  - Aggregate results: aggregate_results.json")
    logger.info(f"  - Individual stock results: {results_dir}/<TICKER>/")
    logger.info("=" * 80)

    return aggregate_results


# ============================================================================
# Utility Functions
# ============================================================================

def list_available_data(data_dir: Path = DEFAULT_DATA_DIR) -> List[str]:
    """
    List all available stock data files in data directory.

    Args:
        data_dir: Directory containing OHLCV CSV files

    Returns:
        List of ticker symbols found in data directory

    Example:
        >>> tickers = list_available_data()
        >>> print(f"Found {len(tickers)} stocks")
        Found 30 stocks
    """
    if not data_dir.exists():
        logger.warning(f"Data directory {data_dir} does not exist")
        return []

    available_files = list(data_dir.glob('*.csv'))
    tickers = []

    # Common non-stock patterns to exclude
    excluded_patterns = ['collection', 'summary', 'index', 'VNINDEX', 'VN30']

    for file_path in available_files:
        # Extract ticker from filename
        filename = file_path.stem  # Remove .csv
        # Remove common suffixes
        ticker = filename.replace('_ohlcv', '').replace('_OHLCV', '')

        # Check if ticker should be excluded
        ticker_upper = ticker.upper()
        should_exclude = any(pattern.lower() in ticker_upper.lower() for pattern in excluded_patterns)

        if not should_exclude and ticker_upper:
            tickers.append(ticker_upper)

    logger.info(f"Found {len(tickers)} stock data files in {data_dir}")

    return sorted(set(tickers))


def validate_data_directory(data_dir: Path = DEFAULT_DATA_DIR) -> bool:
    """
    Validate that data directory exists and contains CSV files.

    Args:
        data_dir: Directory to validate

    Returns:
        bool: True if directory is valid, False otherwise

    Example:
        >>> is_valid = validate_data_directory()
        >>> if not is_valid:
        ...     print("Data directory not found!")
    """
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        return False

    csv_files = list(data_dir.glob('*.csv'))
    if len(csv_files) == 0:
        logger.error(f"No CSV files found in {data_dir}")
        return False

    logger.info(f"✅ Data directory validated: {len(csv_files)} CSV files found")
    return True
