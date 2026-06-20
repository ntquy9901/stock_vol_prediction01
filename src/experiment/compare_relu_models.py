"""
Compare SimpleLSTM vs LSTM-HAR to understand ReLU issue.
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.lstm_baseline.model import SimpleVolatilityLSTM
from src.lstm_har_baseline.model import HARVolatilityLSTM
from src.lstm_baseline.dataset import PooledVolatilityDataset
from src.lstm_har_baseline.dataset import HARVolatilityDataset
from torch.utils.data import DataLoader

print("=" * 80)
print("SIMPLE LSTM vs LSTM-HAR ReLU COMPARISON")
print("=" * 80)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "=" * 80)
print("CHECK 1: Dataset Comparison")
print("=" * 80)

# Create SimpleLSTM dataset
simple_dataset = PooledVolatilityDataset('data/processed', seq_length=22, forecast_horizon=5)
print(f"\nSimpleLSTM Dataset:")
print(f"  Total samples: {len(simple_dataset)}")

# Get one sample
X_simple, y_simple = simple_dataset[0]
print(f"  Input shape: {X_simple.shape}")
print(f"  Input range: [{X_simple.min():.6f}, {X_simple.max():.6f}]")
print(f"  Input mean: {X_simple.mean():.6f}, std: {X_simple.std():.6f}")
print(f"  Target: {y_simple.item():.6f}")

# Create LSTM-HAR dataset
har_dataset = HARVolatilityDataset('data/processed', seq_length=22, forecast_horizon=5)
print(f"\nLSTM-HAR Dataset:")
print(f"  Total samples: {len(har_dataset)}")

# Get one sample
X_har, y_har = har_dataset[0]
print(f"  Input shape: {X_har.shape}")
print(f"  Input range: [{X_har.min():.6f}, {X_har.max():.6f}]")
print(f"  Input mean: {X_har.mean():.6f}, std: {X_har.std():.6f}")
print(f"  Target: {y_har.item():.6f}")

print("\n" + "=" * 80)
print("CHECK 2: Model Output Distribution (with ReLU)")
print("=" * 80)

# Test SimpleLSTM with multiple initializations
print("\nSimpleLSTM (128 hidden units):")
simple_loader = DataLoader(simple_dataset, batch_size=64, shuffle=False, num_workers=0)
X_simple_batch, y_simple_batch = next(iter(simple_loader))
X_simple_batch = X_simple_batch.to(device)

simple_zero_count = 0
for i in range(10):
    model = SimpleVolatilityLSTM(hidden_size=128).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(X_simple_batch)

    is_zero = (predictions.abs().sum() < 1e-6)
    if is_zero:
        simple_zero_count += 1

    mean_pred = predictions.mean().item()
    std_pred = predictions.std().item()
    zero_marker = "[ALL ZEROS]" if is_zero else ""
    print(f"  Init {i+1}: Pred mean={mean_pred:.6f}, std={std_pred:.6f} {zero_marker}")

print(f"\n  Zero outputs: {simple_zero_count}/10 ({simple_zero_count*10}%)")

# Test LSTM-HAR with multiple initializations
print("\nLSTM-HAR (64 hidden units):")
har_loader = DataLoader(har_dataset, batch_size=64, shuffle=False, num_workers=0)
X_har_batch, y_har_batch = next(iter(har_loader))
X_har_batch = X_har_batch.to(device)

har_zero_count = 0
for i in range(10):
    model = HARVolatilityLSTM(hidden_size=64, num_layers=2, dropout=0.0).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(X_har_batch)

    is_zero = (predictions.abs().sum() < 1e-6)
    if is_zero:
        har_zero_count += 1

    mean_pred = predictions.mean().item()
    std_pred = predictions.std().item()
    zero_marker = "[ALL ZEROS]" if is_zero else ""
    print(f"  Init {i+1}: Pred mean={mean_pred:.6f}, std={std_pred:.6f} {zero_marker}")

print(f"\n  Zero outputs: {har_zero_count}/10 ({har_zero_count*10}%)")

print("\n" + "=" * 80)
print("CHECK 3: Target Scaling Comparison")
print("=" * 80)

# Check target distributions
simple_targets = []
har_targets = []

for i in range(0, min(1000, len(simple_dataset)), 10):
    _, y_simple = simple_dataset[i]
    simple_targets.append(y_simple.item())

    _, y_har = har_dataset[i]
    har_targets.append(y_har.item())

simple_targets = np.array(simple_targets)
har_targets = np.array(har_targets)

print(f"\nSimpleLSTM Targets (raw):")
print(f"  Range: [{simple_targets.min():.6f}, {simple_targets.max():.6f}]")
print(f"  Mean: {simple_targets.mean():.6f}, Std: {simple_targets.std():.6f}")
print(f"  Non-zero ratio: {(simple_targets != 0).sum() / len(simple_targets):.2%}")

print(f"\nLSTM-HAR Targets (scaled):")
print(f"  Range: [{har_targets.min():.6f}, {har_targets.max():.6f}]")
print(f"  Mean: {har_targets.mean():.6f}, Std: {har_targets.std():.6f}")
print(f"  Non-zero ratio: {(har_targets != 0).sum() / len(har_targets):.2%}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print(f"\n1. Zero Output Rate:")
print(f"   SimpleLSTM: {simple_zero_count}/10 ({simple_zero_count*10}%)")
print(f"   LSTM-HAR: {har_zero_count}/10 ({har_zero_count*10}%)")

if simple_zero_count < har_zero_count:
    print(f"   → SimpleLSTM has FEWER zero outputs (better)")
elif simple_zero_count > har_zero_count:
    print(f"   → SimpleLSTM has MORE zero outputs (worse)")
else:
    print(f"   → Both have SAME zero output rate")

print(f"\n2. Input Scale:")
print(f"   SimpleLSTM: mean={X_simple.mean():.6f}, std={X_simple.std():.6f}")
print(f"   LSTM-HAR: mean={X_har.mean():.6f}, std={X_har.std():.6f}")

print(f"\n3. Target Scale:")
print(f"   SimpleLSTM: mean={simple_targets.mean():.6f}, std={simple_targets.std():.6f}")
print(f"   LSTM-HAR: mean={har_targets.mean():.6f}, std={har_targets.std():.6f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if simple_zero_count == 0:
    print("\n[OK] SimpleLSTM does NOT have zero output issue")
    print("Possible reasons:")
    print("1. Larger hidden size (128 vs 64) helps")
    print("2. Different input scale helps avoid negative fc outputs")
    print("3. Single layer may be more stable")
elif simple_zero_count < har_zero_count:
    print(f"\n[PARTIAL] SimpleLSTM has FEWER issues ({simple_zero_count} vs {har_zero_count})")
    print("But still affected by ReLU problem")
else:
    print(f"\n[WARNING] SimpleLSTM has SAME issue ({simple_zero_count} vs {har_zero_count})")
    print("Both models affected by ReLU zero output bug")

print("\nRecommendation: Remove ReLU from BOTH models")
print("=" * 80)
