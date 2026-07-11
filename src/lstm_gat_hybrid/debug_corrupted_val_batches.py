"""
Debug script to identify which stocks/sequences have corrupted targets in validation set.

Run: python -m src.lstm_gat_hybrid.debug_corrupted_val_batches
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.lstm_gat_hybrid.dataset_with_graph_method import create_multi_stock_dataloaders_with_graph_method_fixed


def debug_validation_targets():
    """Load validation loader and check for extreme target values"""

    print("="*80)
    print("DEBUGGING CORRUPTED VALIDATION BATCHES")
    print("="*80)

    # Load dataloaders (same args as training)
    train_loader, val_loader, test_loader, datasets = create_multi_stock_dataloaders_with_graph_method_fixed(
        data_dir='data/processed',
        seq_length=22,
        forecast_horizon=5,
        graph_method='knn',
        graph_threshold=None,
        k_neighbors=8,
        batch_size=11,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        num_workers=0,
        normalize=True,
        remove_outliers=True,
        n_std=3.0,
        data_augmentation=False,  # Disable for debugging
        augmentation_prob=0.0,
        augmentation_factor=0.0
    )

    val_dataset = datasets[1]

    # Extract base dataset if wrapped in Subset
    if hasattr(val_dataset, 'dataset'):
        val_dataset = val_dataset.dataset

    stock_names = val_dataset.stock_names
    print(f"\nStocks in validation: {len(stock_names)}")
    print(f"Validation sequences: {len(val_dataset)}")

    # Scan all batches for extreme values
    print("\n" + "="*80)
    print("SCANNING BATCHES FOR EXTREME TARGET VALUES")
    print("="*80)

    extreme_batches = []

    for batch_idx, (x, adj_matrix, y, _) in enumerate(val_loader):
        # Check for extreme values
        y_flat = y.flatten()
        max_val = y_flat.abs().max().item()

        if max_val > 100:  # Corrupted
            mean_val = y_flat.mean().item()
            min_val = y_flat.min().item()

            # Find which sequence(s) in this batch have extreme values
            batch_size, num_stocks = y.shape

            extreme_sequences = []
            for seq_idx in range(batch_size):
                for stock_idx in range(num_stocks):
                    val = y[seq_idx, stock_idx].item()
                    if abs(val) > 100:
                        # Map to original dataset index
                        dataset_idx = batch_idx * batch_size + seq_idx
                        extreme_sequences.append({
                            'seq_idx': seq_idx,
                            'stock_idx': stock_idx,
                            'stock_name': stock_names[stock_idx],
                            'value': val,
                            'dataset_idx': dataset_idx
                        })

            extreme_batches.append({
                'batch_idx': batch_idx,
                'max': max_val,
                'mean': mean_val,
                'min': min_val,
                'extreme_sequences': extreme_sequences
            })

    # Print summary
    print(f"\nFound {len(extreme_batches)} corrupted batches:")
    print("-" * 80)

    for batch_info in extreme_batches:
        print(f"\nBatch {batch_info['batch_idx']}:")
        print(f"  max={batch_info['max']:.2f}, mean={batch_info['mean']:.2f}, min={batch_info['min']:.2f}")
        print(f"  Extreme sequences ({len(batch_info['extreme_sequences'])}):")

        for seq_info in batch_info['extreme_sequences']:
            print(f"    - seq={seq_info['seq_idx']}, stock={seq_info['stock_name']}, value={seq_info['value']:.2f}")

    # Now check the raw (non-normalized) targets for these sequences
    print("\n" + "="*80)
    print("CHECKING RAW TARGETS FOR CORRUPTED SEQUENCES")
    print("="*80)

    # Get the raw dataset (before normalization) to check original values
    # We'll need to access the sequences before __getitem__ normalization
    for batch_info in extreme_batches[:3]:  # Check first 3 corrupted batches
        print(f"\nBatch {batch_info['batch_idx']}:")
        for seq_info in batch_info['extreme_sequences'][:5]:  # First 5 extreme values
            dataset_idx = seq_info['dataset_idx']
            if dataset_idx < len(val_dataset.sequences):
                x_raw, adj_raw, y_raw = val_dataset.sequences[dataset_idx]
                stock_idx_local = seq_info['stock_idx']
                raw_target = y_raw[stock_idx_local]
                print(f"  {seq_info['stock_name']} (seq {dataset_idx}): normalized={seq_info['value']:.2f}, raw={raw_target:.6f}")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. If raw targets are also extreme (> 0.1), it's a data preprocessing bug")
    print("2. If raw targets are reasonable but normalized values are extreme,")
    print("   it's a normalization mismatch (train/val have different ranges)")
    print("3. Consider adding outlier clipping to normalization: clip at ±5 std")


if __name__ == '__main__':
    debug_validation_targets()
