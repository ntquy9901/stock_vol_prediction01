"""Debug evaluation metrics"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from torch.utils.data import DataLoader
from src.lstm_baseline.dataset import PooledVolatilityDataset
from src.lstm_baseline.model import SimpleVolatilityLSTM
from src.common.evaluation import evaluate_predictions

# Load dataset
dataset = PooledVolatilityDataset('data/processed', seq_length=22, forecast_horizon=5)

# Create small test set
test_size = 100
test_indices = list(range(len(dataset) - test_size, len(dataset)))

print(f"Testing on {test_size} samples...")

# Get predictions and actuals
predictions_scaled = []
actuals_scaled = []

for idx in test_indices:
    X, y_scaled = dataset[idx]
    predictions_scaled.append(y_scaled.item())  # Use actual as prediction for now
    actuals_scaled.append(y_scaled.item())

predictions_scaled = np.array(predictions_scaled)
actuals_scaled = np.array(actuals_scaled)

print(f"Scaled predictions shape: {predictions_scaled.shape}")
print(f"Scaled predictions range: [{predictions_scaled.min():.6f}, {predictions_scaled.max():.6f}]")

# Inverse transform
predictions = dataset.target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
actuals = dataset.target_scaler.inverse_transform(actuals_scaled.reshape(-1, 1)).flatten()

print(f"\nRaw predictions shape: {predictions.shape}")
print(f"Raw predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
print(f"Raw actuals range: [{actuals.min():.6f}, {actuals.max():.6f}]")

# Calculate metrics
metrics = evaluate_predictions(actuals, predictions)
print("\nMetrics (predictions = actuals, should be perfect):")
for name, value in metrics.items():
    print(f"  {name}: {value:.6f}")
