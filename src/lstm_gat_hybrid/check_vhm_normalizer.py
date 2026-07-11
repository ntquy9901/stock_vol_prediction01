"""
Check VHM normalizer statistics to understand the normalization mismatch.

Run: python -m src.lstm_gat_hybrid.check_vhm_normalizer
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.lstm_gat_hybrid.dataset_with_graph_method import create_multi_stock_dataloaders_with_graph_method_fixed


def check_vhm_normalizer():
    """Check VHM's normalizer and compare train/val ranges"""

    print("="*80)
    print("CHECKING VHM NORMALIZER")
    print("="*80)

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
        data_augmentation=False,
        augmentation_prob=0.0,
        augmentation_factor=0.0
    )

    train_dataset = datasets[0]
    val_dataset = datasets[1]

    # Extract base datasets if wrapped in Subset
    if hasattr(train_dataset, 'dataset'):
        train_dataset = train_dataset.dataset
    if hasattr(val_dataset, 'dataset'):
        val_dataset = val_dataset.dataset

    # Get VHM normalizer
    if 'VHM' not in train_dataset.target_normalizers:
        print("❌ VHM not in normalizers!")
        return

    vhm_normalizer = train_dataset.target_normalizers['VHM']

    print(f"\nVHM VolatilityNormalizer (fit on TRAINING data):")
    print(f"  mean: {vhm_normalizer.mean:.8f}")
    print(f"  std:  {vhm_normalizer.std:.8f}")
    print(f"  1/std (amplification factor): {1.0 / vhm_normalizer.std:.2f}")

    # Collect VHM targets from training and validation (raw, before normalization)
    print(f"\n" + "="*80)
    print("VHM TARGET RANGES (raw, before normalization)")
    print("="*80)

    train_vhm_raw = []
    for idx in range(len(train_dataset)):
        _, _, y_raw = train_dataset.sequences[idx]
        stock_idx = train_dataset.stock_names.index('VHM')
        train_vhm_raw.append(y_raw[stock_idx].item())

    val_vhm_raw = []
    for idx in range(len(val_dataset)):
        _, _, y_raw = val_dataset.sequences[idx]
        stock_idx = val_dataset.stock_names.index('VHM')
        val_vhm_raw.append(y_raw[stock_idx].item())

    train_vhm_raw = np.array(train_vhm_raw)
    val_vhm_raw = np.array(val_vhm_raw)

    print(f"\nTraining VHM (raw):")
    print(f"  mean: {train_vhm_raw.mean():.8f}")
    print(f"  std:  {train_vhm_raw.std():.8f}")
    print(f"  min:  {train_vhm_raw.min():.8f}")
    print(f"  max:  {train_vhm_raw.max():.8f}")

    print(f"\nValidation VHM (raw):")
    print(f"  mean: {val_vhm_raw.mean():.8f}")
    print(f"  std:  {val_vhm_raw.std():.8f}")
    print(f"  min:  {val_vhm_raw.min():.8f}")
    print(f"  max:  {val_vhm_raw.max():.8f}")

    # Simulate normalization on validation extreme values
    print(f"\n" + "="*80)
    print("NORMALIZATION SIMULATION")
    print("="*80)

    print(f"\nWhen validation VHM max ({val_vhm_raw.max():.8f}) is normalized:")
    normalized_max = (val_vhm_raw.max() - vhm_normalizer.mean) / vhm_normalizer.std
    print(f"  (val_max - train_mean) / train_std = {normalized_max:.2f}")

    # Find the problematic sequences
    print(f"\n" + "="*80)
    print("PROBLEMATIC VALIDATION SEQUENCES (normalized value > 100)")
    print("="*80)

    problematic = []
    for idx in range(len(val_dataset)):
        _, _, y_raw = val_dataset.sequences[idx]
        stock_idx = val_dataset.stock_names.index('VHM')
        raw_val = y_raw[stock_idx].item()
        norm_val = (raw_val - vhm_normalizer.mean) / vhm_normalizer.std

        if abs(norm_val) > 100:
            problematic.append((idx, raw_val, norm_val))

    print(f"\nFound {len(problematic)} problematic sequences:")
    for idx, raw, norm in problematic[:10]:
        print(f"  Seq {idx:3d}: raw={raw:.8f} → normalized={norm:.2f}")

    print(f"\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    print(f"VHM has very low variance in training (std={train_vhm_raw.std():.8f})")
    print(f"but has higher values in validation (max={val_vhm_raw.max():.8f}).")
    print(f"\nWhen normalized, validation spikes become extreme:")
    print(f"  train std is {train_vhm_raw.std():.8f} → 1/std = {1/train_vhm_raw.std():.0f}x amplification")
    print(f"  validation max {val_vhm_raw.max():.8f} becomes normalized ~{normalized_max:.0f}")
    print(f"\nThis is a NORMALIZATION MISMATCH, not a data bug.")


if __name__ == '__main__':
    check_vhm_normalizer()
