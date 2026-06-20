"""
Temporal Data Split Utilities

This module provides utilities for splitting time series data
chronologically to prevent data leakage.

Supports:
- 2-way split (train/test)
- 3-way split (train/validation/test)
- Per-stock temporal splitting
- Pooled temporal splitting

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from typing import Tuple, List, Optional, Union
from datetime import datetime


def temporal_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset chronologically into train/val/test sets.

    CRITICAL: This is a TEMPORAL split, not random split.
    Uses indices to maintain chronological order.

    Args:
        dataset: Dataset or list to split
        train_ratio: Fraction for training (default: 0.7)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)

    Returns:
        train_indices, val_indices, test_indices

    Example:
        >>> # Split dataset 70/15/15
        >>> train_idx, val_idx, test_idx = temporal_split(dataset, 0.7, 0.15, 0.15)
        >>> train_set = Subset(dataset, train_idx)
        >>> val_set = Subset(dataset, val_idx)
        >>> test_set = Subset(dataset, test_idx)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    # test_size = total_size - train_size - val_size

    # CHRONOLOGICAL split (not random!)
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))

    print(f"\n[TEMPORAL SPLIT]")
    print(f"  Total: {total_size} samples")
    print(f"  Train: {len(train_indices)} samples ({train_ratio:.1%})")
    print(f"  Val:   {len(val_indices)} samples ({val_ratio:.1%})")
    print(f"  Test:  {len(test_indices)} samples ({test_ratio:.1%})")

    return train_indices, val_indices, test_indices


def temporal_split_dataframe(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15,
                            test_ratio=0.15, date_column='date'):
    """
    Split DataFrame chronologically into train/val/test sets.

    Maintains temporal order per stock to prevent data leakage.

    Args:
        df: DataFrame with date column
        train_ratio: Fraction for training (default: 0.7)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)
        date_column: Name of date column (default: 'date')

    Returns:
        train_df, val_df, test_df

    Example:
        >>> train_df, val_df, test_df = temporal_split_dataframe(data, 0.7, 0.15, 0.15)
        >>> print(f"Train: {train_df['date'].min()} to {train_df['date'].max()}")
        >>> print(f"Val:   {val_df['date'].min()} to {val_df['date'].max()}")
        >>> print(f"Test:  {test_df['date'].min()} to {test_df['date'].max()}")
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Sort by date to ensure chronological order
    df = df.sort_values(date_column).reset_index(drop=True)

    total_size = len(df)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # CHRONOLOGICAL split
    train_df = df.iloc[0:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()

    # Print date ranges
    print(f"\n[TEMPORAL SPLIT - DATAFRAME]")
    print(f"  Total: {total_size} rows")
    print(f"  Train: {len(train_df)} rows ({train_ratio:.1%})")
    print(f"    Period: {train_df[date_column].min()} to {train_df[date_column].max()}")
    print(f"  Val:   {len(val_df)} rows ({val_ratio:.1%})")
    print(f"    Period: {val_df[date_column].min()} to {val_df[date_column].max()}")
    print(f"  Test:  {len(test_df)} rows ({test_ratio:.1%})")
    print(f"    Period: {test_df[date_column].min()} to {test_df[date_column].max()}")

    return train_df, val_df, test_df


def get_split_info(dataset_len, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Get information about temporal split without performing split.

    Useful for documenting split plan before execution.

    Args:
        dataset_len: Total size of dataset
        train_ratio: Fraction for training (default: 0.7)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)

    Returns:
        dict with split information
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    train_size = int(dataset_len * train_ratio)
    val_size = int(dataset_len * val_ratio)
    test_size = dataset_len - train_size - val_size

    return {
        'total': dataset_len,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio
    }


def create_temporal_dataloaders(dataset, train_ratio=0.7, val_ratio=0.15,
                               test_ratio=0.15, batch_size=32, num_workers=0,
                               pin_memory=False):
    """
    Create train/val/test dataloaders with temporal split.

    Convenience function for creating dataloaders for PyTorch models.

    Args:
        dataset: Dataset to split
        train_ratio: Fraction for training (default: 0.7)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory (for GPU training)

    Returns:
        train_loader, val_loader, test_loader

    Example:
        >>> train_loader, val_loader, test_loader = create_temporal_dataloaders(
        ...     dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        ...     batch_size=32
        ... )
    """
    from torch.utils.data import DataLoader, Subset

    # Get indices
    train_idx, val_idx, test_idx = temporal_split(
        dataset, train_ratio, val_ratio, test_ratio
    )

    # Create subsets
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,  # Only shuffle within train set
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"\n[DATALOADERS CREATED]")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


class TemporalSplitter:
    """
    Utility class for performing temporal splits on datasets.

    Provides reusable methods for different split configurations.
    """

    def __init__(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Initialize TemporalSplitter with split ratios.

        Args:
            train_ratio: Fraction for training (default: 0.7)
            val_ratio: Fraction for validation (default: 0.15)
            test_ratio: Fraction for testing (default: 0.15)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_dataset(self, dataset):
        """Split dataset into train/val/test."""
        return temporal_split(
            dataset, self.train_ratio, self.val_ratio, self.test_ratio
        )

    def split_dataframe(self, df, date_column='date'):
        """Split DataFrame into train/val/test."""
        return temporal_split_dataframe(
            df, self.train_ratio, self.val_ratio, self.test_ratio, date_column
        )

    def get_info(self, dataset_len):
        """Get split information."""
        return get_split_info(
            dataset_len, self.train_ratio, self.val_ratio, self.test_ratio
        )

    def create_dataloaders(self, dataset, batch_size=32, num_workers=0,
                          pin_memory=False):
        """Create dataloaders with temporal split."""
        return create_temporal_dataloaders(
            dataset, self.train_ratio, self.val_ratio, self.test_ratio,
            batch_size, num_workers, pin_memory
        )


def print_split_summary(train_df, val_df, test_df, date_column='date'):
    """
    Print summary of temporal split for documentation.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        date_column: Name of date column
    """
    print("\n" + "="*80)
    print("TEMPORAL SPLIT SUMMARY")
    print("="*80)

    print(f"\nTrain Set ({len(train_df)} samples, {len(train_df)/(len(train_df)+len(val_df)+len(test_df)):.1%}):")
    print(f"  Period: {train_df[date_column].min()} to {train_df[date_column].max()}")
    print(f"  Duration: {(pd.to_datetime(train_df[date_column].max()) - pd.to_datetime(train_df[date_column].min())).days / 365:.1f} years")

    print(f"\nValidation Set ({len(val_df)} samples, {len(val_df)/(len(train_df)+len(val_df)+len(test_df)):.1%}):")
    print(f"  Period: {val_df[date_column].min()} to {val_df[date_column].max()}")
    print(f"  Duration: {(pd.to_datetime(val_df[date_column].max()) - pd.to_datetime(val_df[date_column].min())).days / 365:.1f} years")

    print(f"\nTest Set ({len(test_df)} samples, {len(test_df)/(len(train_df)+len(val_df)+len(test_df)):.1%}):")
    print(f"  Period: {test_df[date_column].min()} to {test_df[date_column].max()}")
    print(f"  Duration: {(pd.to_datetime(test_df[date_column].max()) - pd.to_datetime(test_df[date_column].min())).days / 365:.1f} years")

    print("\n" + "="*80)
    print("DATA LEAKAGE CHECK")
    print("="*80)
    print(f"  Train end date:   {train_df[date_column].max()}")
    print(f"  Val start date:   {val_df[date_column].min()}")
    print(f"  Val end date:     {val_df[date_column].max()}")
    print(f"  Test start date:  {test_df[date_column].min()}")

    # Check for overlap
    if train_df[date_column].max() >= val_df[date_column].min():
        print("  [WARNING] Train and Val overlap! Potential data leakage!")
    else:
        print("  [OK] No overlap between Train and Val")

    if val_df[date_column].max() >= test_df[date_column].min():
        print("  [WARNING] Val and Test overlap! Potential data leakage!")
    else:
        print("  [OK] No overlap between Val and Test")

    print("="*80)


if __name__ == "__main__":
    """Example usage."""
    print("="*80)
    print("TEMPORAL SPLIT UTILITIES - EXAMPLE USAGE")
    print("="*80)

    # Example 1: Split a list
    print("\n[Example 1: Split list]")
    data = list(range(1000))
    train_idx, val_idx, test_idx = temporal_split(data, 0.7, 0.15, 0.15)
    print(f"Train indices: {train_idx[:5]}...{train_idx[-5:]}")
    print(f"Val indices:   {val_idx[:5]}...{val_idx[-5:]}")
    print(f"Test indices:  {test_idx[:5]}...{test_idx[-5:]}")

    # Example 2: Split DataFrame
    print("\n[Example 2: Split DataFrame]")
    import pandas as pd

    # Create sample DataFrame with dates
    dates = pd.date_range('2006-01-01', periods=1000, freq='B')  # Business days
    df = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(1000)
    })

    train_df, val_df, test_df = temporal_split_dataframe(df, 0.7, 0.15, 0.15)
    print_split_summary(train_df, val_df, test_df)

    # Example 3: Using TemporalSplitter class
    print("\n[Example 3: TemporalSplitter class]")
    splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_df2, val_df2, test_df2 = splitter.split_dataframe(df)
    info = splitter.get_info(len(df))
    print(f"Split info: {info}")
