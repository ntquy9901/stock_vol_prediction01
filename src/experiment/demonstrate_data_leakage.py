"""
Demonstrate Data Leakage Issue: Random Split vs Temporal Split

This script demonstrates the difference between random split and temporal split,
and shows why random split causes data leakage for time series data.

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from datetime import datetime


class SimpleTimeSeriesDataset(Dataset):
    """Simple time series dataset for demonstration."""

    def __init__(self, n_samples=1000, start_date='2006-01-01'):
        # Create synthetic time series data
        self.dates = pd.date_range(start_date, periods=n_samples, freq='B')  # Business days
        self.values = np.random.randn(n_samples).cumsum()  # Random walk
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.FloatTensor([self.values[idx]]),
            torch.FloatTensor([self.values[idx]])
        )


def demonstrate_random_split_leakage(dataset):
    """
    Demonstrate data leakage with random split.

    Problem: Random split mixes time periods, causing data leakage.
    """
    print("\n" + "="*80)
    print("RANDOM SPLIT - DATA LEAKAGE DEMONSTRATION")
    print("="*80)

    # Random split 80/20
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nSplit method: RANDOM (torch.utils.data.random_split)")
    print(f"Train size: {len(train_dataset)} (80%)")
    print(f"Test size: {len(test_dataset)} (20%)")

    # Get actual dates in train and test sets
    train_indices = train_dataset.indices
    test_indices = test_dataset.indices

    train_dates = [dataset.dates[i] for i in train_indices]
    test_dates = [dataset.dates[i] for i in test_indices]

    print(f"\nTrain date range: {min(train_dates)} to {max(train_dates)}")
    print(f"Test date range:  {min(test_dates)} to {max(test_dates)}")

    # Check for overlap
    train_min_date = min(train_dates)
    train_max_date = max(train_dates)
    test_min_date = min(test_dates)

    if train_max_date >= test_min_date:
        print(f"\n[X] DATA LEAKAGE DETECTED!")
        print(f"   Train data goes up to: {train_max_date}")
        print(f"   Test data starts from: {test_min_date}")
        print(f"   OVERLAP: Test data contains dates from {train_min_date} to {train_max_date}")
        print(f"   Model sees future patterns during training!")

        # Count how many test samples are from training period
        leaked_samples = sum(1 for d in test_dates if d <= train_max_date)
        leak_percentage = (leaked_samples / len(test_dates)) * 100
        print(f"   {leaked_samples}/{len(test_dates)} test samples ({leak_percentage:.1f}%) are from training period")

    return train_dataset, test_dataset


def demonstrate_temporal_split_no_leakage(dataset):
    """
    Demonstrate proper temporal split without data leakage.

    Benefit: Chronological split prevents data leakage.
    """
    print("\n" + "="*80)
    print("TEMPORAL SPLIT - NO DATA LEAKAGE")
    print("="*80)

    # Temporal split 80/20
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # CHRONOLOGICAL split (not random!)
    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"\nSplit method: TEMPORAL (chronological)")
    print(f"Train size: {len(train_dataset)} (80%)")
    print(f"Test size: {len(test_dataset)} (20%)")

    # Get actual dates in train and test sets
    train_dates = [dataset.dates[i] for i in train_indices]
    test_dates = [dataset.dates[i] for i in test_indices]

    print(f"\nTrain date range: {min(train_dates)} to {max(train_dates)}")
    print(f"Test date range:  {min(test_dates)} to {max(test_dates)}")

    # Check for overlap
    train_max_date = max(train_dates)
    test_min_date = min(test_dates)

    if train_max_date < test_min_date:
        print(f"\n[OK] NO DATA LEAKAGE!")
        print(f"   Train data ends: {train_max_date}")
        print(f"   Test data starts: {test_min_date}")
        print(f"   GAP: {(test_min_date - train_max_date).days} calendar days")
        print(f"   Model sees only past data, test on completely unseen future data")

    return train_dataset, test_dataset


def demonstrate_3_way_temporal_split(dataset):
    """
    Demonstrate 3-way temporal split (70/15/15).

    Benefit: Proper validation during training.
    """
    print("\n" + "="*80)
    print("3-WAY TEMPORAL SPLIT (70/15/15) - BEST PRACTICE")
    print("="*80)

    # Temporal split 70/15/15
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - val_size

    # CHRONOLOGICAL split
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"\nSplit method: TEMPORAL (chronological 70/15/15)")
    print(f"Train size: {len(train_dataset)} (70%)")
    print(f"Val size:   {len(val_dataset)} (15%)")
    print(f"Test size:  {len(test_dataset)} (15%)")

    # Get actual dates
    train_dates = [dataset.dates[i] for i in train_indices]
    val_dates = [dataset.dates[i] for i in val_indices]
    test_dates = [dataset.dates[i] for i in test_indices]

    print(f"\nTrain date range: {min(train_dates)} to {max(train_dates)}")
    print(f"Val date range:   {min(val_dates)} to {max(val_dates)}")
    print(f"Test date range:  {min(test_dates)} to {max(test_dates)}")

    # Check for data leakage
    train_max_date = max(train_dates)
    val_max_date = max(val_dates)
    test_min_date = min(test_dates)

    print(f"\nData leakage check:")
    if train_max_date < min(val_dates):
        print(f"  [OK] Train -> Val: No gap (OK, val follows train)")
    else:
        print(f"  [X] Train -> Val: OVERLAP!")

    if val_max_date < test_min_date:
        print(f"  [OK] Val -> Test: No gap (OK, test follows val)")
    else:
        print(f"  [X] Val -> Test: OVERLAP!")

    print(f"\nBenefits:")
    print(f"  • Train: Learn patterns from past data")
    print(f"  • Val:   Tune hyperparameters (early stopping)")
    print(f"  • Test:  Final evaluation on unseen data")
    print(f"  • No data leakage: Strict chronological order")

    return train_dataset, val_dataset, test_dataset


def main():
    """Main demonstration."""
    print("="*80)
    print("DATA LEAKAGE DEMONSTRATION - RANDOM VS TEMPORAL SPLIT")
    print("="*80)
    print(f"Demonstration date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create sample dataset
    dataset = SimpleTimeSeriesDataset(n_samples=1000, start_date='2006-01-01')

    print(f"\nDataset info:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Date range: {dataset.dates[0]} to {dataset.dates[-1]}")
    print(f"  Duration: {(dataset.dates[-1] - dataset.dates[0]).days / 365:.1f} years")

    # Demonstrate random split (WRONG)
    train_rand, test_rand = demonstrate_random_split_leakage(dataset)

    # Demonstrate 2-way temporal split (BETTER)
    train_temp, test_temp = demonstrate_temporal_split_no_leakage(dataset)

    # Demonstrate 3-way temporal split (BEST)
    train_3way, val_3way, test_3way = demonstrate_3_way_temporal_split(dataset)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - WHICH METHOD TO USE?")
    print("="*80)

    print("\n[X] RANDOM SPLIT (80/20):")
    print("   Pros: None for time series")
    print("   Cons: Data leakage, overestimated metrics, unreliable results")

    print("\n[!]️  TEMPORAL SPLIT (80/20):")
    print("   Pros: No data leakage, reliable metrics")
    print("   Cons: No validation set, cannot tune hyperparameters")

    print("\n[OK] TEMPORAL SPLIT (70/15/15):")
    print("   Pros: No data leakage, proper validation, reliable test metrics")
    print("   Cons: Slightly less training data (70% vs 80%)")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("\nFor time series forecasting (volatility, prices, etc.):")
    print("  -> Use 3-way temporal split (70/15/15)")
    print("  -> Train on earliest 70% of data")
    print("  -> Validate on next 15% (early stopping, model selection)")
    print("  -> Test on last 15% (final unbiased evaluation)")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
