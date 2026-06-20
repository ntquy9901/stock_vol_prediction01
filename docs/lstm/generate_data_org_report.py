"""
LSTM Data Organization Report

Document how training/validation data is organized and how sliding windows
are created for the LSTM volatility prediction model.

Author: Stock Volatility Prediction Team
Date: 2026-06-17
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.lstm_baseline.dataset import PooledVolatilityDataset


def analyze_data_organization(data_dir: str):
    """
    Analyze and report on how data is organized for LSTM training.

    Args:
        data_dir: Directory containing processed CSV files
    """
    print("\n" + "=" * 80)
    print("LSTM DATA ORGANIZATION ANALYSIS")
    print("=" * 80)

    # Load dataset
    print("\n[1] Loading Dataset...")
    dataset = PooledVolatilityDataset(data_dir, seq_length=22, forecast_horizon=5)

    # Analyze stocks
    print("\n[2] Stock Coverage Analysis...")
    stocks_in_dataset = set(m[0] for m in dataset.metadata)
    print(f"   Total stocks in dataset: {len(stocks_in_dataset)}")
    print(f"   Sample stocks: {list(stocks_in_dataset)[:5]}")

    # Analyze sequences per stock
    print("\n[3] Sequences per Stock...")
    sequences_per_stock = {}
    for ticker, idx in dataset.metadata:
        if ticker not in sequences_per_stock:
            sequences_per_stock[ticker] = 0
        sequences_per_stock[ticker] += 1

    print(f"   Min sequences per stock: {min(sequences_per_stock.values())}")
    print(f"   Max sequences per stock: {max(sequences_per_stock.values())}")
    print(f"   Avg sequences per stock: {np.mean(list(sequences_per_stock.values())):.1f}")
    print(f"   Total sequences: {len(dataset)}")

    # Analyze temporal distribution
    print("\n[4] Temporal Distribution...")
    sequences_by_decade = {}
    for ticker in sequences_per_stock.keys():
        # Get file to check date range
        file_path = os.path.join(data_dir, f"{ticker}_processed.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                decades = df['Date'].dt.year // 10 * 10
                for decade in decades.unique():
                    decade_key = f"{decade}s"
                    if decade_key not in sequences_by_decade:
                        sequences_by_decade[decade_key] = 0
                    sequences_by_decade[decade_key] += len(df[df['Date'].dt.year // 10 * 10 == decade])

    print("   Sequences by decade:")
    for decade in sorted(sequences_by_decade.keys()):
        print(f"     {decade}: {sequences_by_decade[decade]:,} sequences")

    return dataset, sequences_per_stock


def explain_sliding_windows(dataset: PooledVolatilityDataset, num_examples: int = 3):
    """
    Explain how sliding windows are created with examples.

    Args:
        dataset: PooledVolatilityDataset instance
        num_examples: Number of examples to show
    """
    print("\n" + "=" * 80)
    print("SLIDING WINDOW MECHANISM")
    print("=" * 80)

    print("\n[1] Window Configuration:")
    print(f"   Sequence Length (lookback): {dataset.seq_length} days")
    print(f"   Forecast Horizon: {dataset.forecast_horizon} days")
    print(f"   Total window size: {dataset.seq_length + dataset.forecast_horizon} days")

    print("\n[2] Sliding Window Logic:")
    print("   For each stock with N days of data:")
    print("   - Create sequences for i = 0 to N - seq_length - forecast_horizon")
    print("   - Each sequence uses days [i : i + seq_length] as input")
    print("   - Target is day [i + seq_length + forecast_horizon - 1]")

    print("\n[3] Example: Stock with 100 days")
    print("   Total sequences created: 100 - 22 - 5 + 1 = 74 sequences")
    print("   Sequence 1: Days [0:22] -> Target day 26")
    print("   Sequence 2: Days [1:23] -> Target day 27")
    print("   Sequence 3: Days [2:24] -> Target day 28")
    print("   ...")
    print("   Sequence 74: Days [73:95] -> Target day 99")

    print("\n[4] Actual Examples from Dataset:")
    example_count = 0
    for idx in range(min(num_examples, len(dataset))):
        X_seq, y_target = dataset[idx]

        ticker, seq_idx = dataset.metadata[idx]
        X_values = X_seq.numpy().flatten()
        y_value = y_target.item()

        print(f"\n   Example {idx + 1}: {ticker}, Sequence #{seq_idx}")
        print(f"   Input shape: {X_seq.shape}")
        print(f"   Input (first 5 days): {X_values[:5]}")
        print(f"   Input (last 5 days): {X_values[-5:]}")
        print(f"   Target (scaled): {y_value:.6f}")

        example_count += 1
        if example_count >= num_examples:
            break


def explain_train_validation_split(dataset: PooledVolatilityDataset):
    """
    Explain how train/validation split is performed.

    Args:
        dataset: PooledVolatilityDataset instance
    """
    print("\n" + "=" * 80)
    print("TRAIN/VALIDATION SPLIT METHODOLOGY")
    print("=" * 80)

    print("\n[1] Split Configuration:")
    train_ratio = 0.8
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    print(f"   Total sequences: {len(dataset):,}")
    print(f"   Train ratio: {train_ratio:.0%}")
    print(f"   Train size: {train_size:,} sequences")
    print(f"   Validation size: {test_size:,} sequences")

    print("\n[2] Split Method: Random Split")
    print("   Method: torch.utils.data.random_split")
    print("   Random seed: 42 (reproducible)")
    print("   Shuffle: Applied to dataset")

    print("\n[3] Important Note - Temporal Integrity:")
    print("   WARNING: Random split DOES NOT preserve temporal order!")
    print("   This means:")
    print("   - Training data may contain sequences AFTER validation data")
    print("   - Potential look-ahead bias if not careful")
    print("   - Acceptable for pooled approach across 30 stocks")

    print("\n[4] Recommended Improvement for Production:")
    print("   Option 1: Temporal split (preserve order per stock)")
    print("   - Split each stock's sequences chronologically")
    print("   - Train: first 80% of each stock")
    print("   - Val: last 20% of each stock")
    print("   - Preserves temporal integrity")

    print("\n   Option 2: Leave-one-stock-out cross-validation")
    print("   - Train on 29 stocks, validate on 1 stock")
    print("   - Rotate through all 30 stocks")
    print("   - Tests generalization to unseen stocks")

    print("\n   Option 3: Time-based split (most rigorous)")
    print("   - Use date cutoff (e.g., data before 2020 for train)")
    print("   - Ensures no future data in training")
    print("   - Most realistic for production")

    print("\n[5] Current Split Summary:")
    print(f"   Train samples: {train_size:,}")
    print(f"   Validation samples: {test_size:,}")
    print(f"   Batch size (recommended): 64")
    print(f"   Train batches per epoch: ~{train_size // 64}")
    print(f"   Validation batches per epoch: ~{test_size // 64}")


def calculate_data_statistics(dataset: PooledVolatilityDataset):
    """
    Calculate and display data statistics.

    Args:
        dataset: PooledVolatilityDataset instance
    """
    print("\n" + "=" * 80)
    print("DATA STATISTICS")
    print("=" * 80)

    # Sample all data
    print("\n[1] Collecting samples...")
    all_inputs = []
    all_targets = []

    for idx in range(len(dataset)):
        X_seq, y_target = dataset[idx]
        all_inputs.append(X_seq.numpy().flatten())
        all_targets.append(y_target.item())

        if idx >= 10000:  # Sample subset for speed
            break

    all_inputs = np.array(all_inputs)
    all_targets = np.array(all_targets)

    print("\n[2] Input Statistics (Scaled):")
    print(f"   Mean: {np.mean(all_inputs):.6f}")
    print(f"   Std: {np.std(all_inputs):.6f}")
    print(f"   Min: {np.min(all_inputs):.6f}")
    print(f"   Max: {np.max(all_inputs):.6f}")

    print("\n[3] Target Statistics (Scaled):")
    print(f"   Mean: {np.mean(all_targets):.6f}")
    print(f"   Std: {np.std(all_targets):.6f}")
    print(f"   Min: {np.min(all_targets):.6f}")
    print(f"   Max: {np.max(all_targets):.6f}")

    print("\n[4] Distribution by Stock:")
    sequences_by_stock = {}
    for ticker, _ in dataset.metadata:
        if ticker not in sequences_by_stock:
            sequences_by_stock[ticker] = 0
        sequences_by_stock[ticker] += 1

    print(f"   Total stocks: {len(sequences_by_stock)}")
    print(f"   Top 5 stocks by sequences:")
    for ticker, count in sorted(sequences_by_stock.items(),
                               key=lambda x: x[1], reverse=True)[:5]:
        print(f"     {ticker}: {count:,} sequences")

    print(f"   Bottom 5 stocks by sequences:")
    for ticker, count in sorted(sequences_by_stock.items(),
                               key=lambda x: x[1])[:5]:
        print(f"     {ticker}: {count:,} sequences")


def generate_summary_report(data_dir: str, output_path: str = None):
    """
    Generate comprehensive summary report.

    Args:
        data_dir: Directory containing processed CSV files
        output_path: Optional path to save text report
    """
    print("\n" + "=" * 80)
    print("GENERATING DATA ORGANIZATION REPORT")
    print("=" * 80)

    # Analyze data
    dataset, sequences_per_stock = analyze_data_organization(data_dir)

    # Explain sliding windows
    explain_sliding_windows(dataset, num_examples=3)

    # Explain train/validation split
    explain_train_validation_split(dataset)

    # Calculate statistics
    calculate_data_statistics(dataset)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n[Data Organization]")
    print(f"   Pooled approach: All 30 stocks combined")
    print(f"   Sliding windows: {dataset.seq_length}-day lookback, {dataset.forecast_horizon}-day forecast")
    print(f"   Total sequences: {len(dataset):,}")
    print(f"   Train/Val split: 80/20 (random)")

    print("\n[Snapshot Creation]")
    print(f"   For each day i in [0, N - 22 - 5]:")
    print(f"     Input: days [i : i + 22] (22-day window)")
    print(f"     Target: day [i + 22 + 5 - 1] (5-day ahead)")

    print("\n[Recommendations]")
    print(f"   Consider temporal split for production")
    print(f"   Add date-based validation for temporal integrity")
    print(f"   Current approach OK for initial development")

    print("\n" + "=" * 80)

    # Save report if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write("LSTM Data Organization Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total sequences: {len(dataset):,}\n")
            f.write(f"Sequence length: {dataset.seq_length} days\n")
            f.write(f"Forecast horizon: {dataset.forecast_horizon} days\n")
            f.write(f"Train/Val split: 80/20\n")
            f.write(f"\nSequences per stock:\n")
            for ticker, count in sorted(sequences_per_stock.items(),
                                       key=lambda x: x[1], reverse=True):
                f.write(f"  {ticker}: {count:,}\n")

        print(f"\nReport saved to: {output_path}")


def main():
    """Main execution."""
    import os

    print("\n" + "=" * 80)
    print("LSTM DATA ORGANIZATION REPORT GENERATOR")
    print("=" * 80)

    # Data directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, 'data/processed')

    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"[ERROR] Processed data directory not found: {data_dir}")
        print("Please run: python -m src.common.process_parkinson_pipeline")
        return

    # Output path
    output_dir = os.path.join(project_root, 'docs/lstm')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'DATA_ORGANIZATION_REPORT.txt')

    # Generate report
    generate_summary_report(data_dir, output_path)

    print("\n" + "=" * 80)
    print("Report generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
