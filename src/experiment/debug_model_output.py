"""
Debug script to check if model outputs vary with different weights.
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.lstm_har_baseline.model import HARVolatilityLSTM
from src.lstm_har_baseline.dataset import HARVolatilityDataset
from torch.utils.data import DataLoader

print("=" * 80)
print("MODEL OUTPUT VARIANCE DEBUG")
print("=" * 80)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Create dataset
dataset = HARVolatilityDataset('data/processed', seq_length=22, forecast_horizon=5)
train_size = int(0.8 * len(dataset))
_, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size],
    generator=torch.Generator().manual_seed(42)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# Get one batch
X_batch, y_batch = next(iter(test_loader))
X_batch = X_batch.to(device)

print(f"\nInput batch shape: {X_batch.shape}")
print(f"Input range: [{X_batch.min().item():.4f}, {X_batch.max().item():.4f}]")
print(f"Input mean: {X_batch.mean().item():.4f}, std: {X_batch.std().item():.4f}")

# Test model outputs with different initializations
print("\n" + "=" * 80)
print("CHECK: Do model outputs vary with different random initializations?")
print("=" * 80)

predictions_list = []
for i in range(10):
    model = HARVolatilityLSTM(hidden_size=64, num_layers=2, dropout=0.0).to(device)
    model.eval()

    with torch.no_grad():
        predictions = model(X_batch)
        predictions_list.append(predictions.cpu())

    mean_pred = predictions.mean().item()
    std_pred = predictions.std().item()
    print(f"Init {i+1}: Pred mean={mean_pred:.6f}, std={std_pred:.6f}, range=[{predictions.min().item():.6f}, {predictions.max().item():.6f}]")

# Convert to numpy for analysis
predictions_array = np.array([p.numpy() for p in predictions_list])  # Shape: (10, batch_size, 1)

print("\n" + "=" * 80)
print("ANALYSIS: Prediction Variance Across Initializations")
print("=" * 80)

# Calculate variance across different initializations
pred_variance = np.var(predictions_array, axis=0)  # Shape: (batch_size, 1)
mean_variance = pred_variance.mean()

print(f"Mean prediction variance across 10 random initializations: {mean_variance:.10f}")

if mean_variance < 1e-8:
    print("[X] CRITICAL BUG: Model outputs are NEARLY IDENTICAL regardless of initialization!")
    print("    This means the model is NOT learning from inputs at all.")
    print("    Possible causes:")
    print("    1. Model architecture bug (output not connected to inputs)")
    print("    2. Data preprocessing bug (all inputs are identical)")
    print("    3. Target scaler bug (all targets scaled to same value)")
else:
    print(f"[OK] Model outputs vary appropriately (variance = {mean_variance:.6f})")

# Check specific predictions
print("\n" + "=" * 80)
print("SPECIFIC PREDICTION SAMPLES")
print("=" * 80)

for sample_idx in [0, 10, 20]:
    print(f"\nSample {sample_idx}:")
    for init_idx in range(5):
        pred_val = predictions_array[init_idx, sample_idx, 0]
        print(f"  Init {init_idx+1}: {pred_val:.8f}")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
