"""Check for extreme values in dataset"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.lstm_baseline.dataset import PooledVolatilityDataset

# Load dataset
dataset = PooledVolatilityDataset('data/processed', seq_length=22, forecast_horizon=5)

# Get all targets (raw)
all_targets_raw = np.array(dataset.targets)

print(f"Total samples: {len(all_targets_raw)}")
print(f"\nRaw targets statistics:")
print(f"  Min: {all_targets_raw.min():.6f}")
print(f"  Max: {all_targets_raw.max():.6f}")
print(f"  Mean: {all_targets_raw.mean():.6f}")
print(f"  Std: {all_targets_raw.std():.6f}")

# Check for extreme values
extreme_threshold = 0.01
extreme_count = np.sum(all_targets_raw > extreme_threshold)
print(f"\nTargets > {extreme_threshold}: {extreme_count} ({100*extreme_count/len(all_targets_raw):.2f}%)")

# Show top 10 extreme values
top_10_indices = np.argsort(all_targets_raw)[-10:]
print(f"\nTop 10 extreme values:")
for i, idx in enumerate(reversed(top_10_indices)):
    print(f"  {i+1}. {all_targets_raw[idx]:.6f} (sample {idx})")

# Check if zeros exist
zero_count = np.sum(all_targets_raw == 0)
print(f"\nZeros in targets: {zero_count}")
