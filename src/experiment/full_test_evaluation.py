"""Full test set evaluation"""
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

# Split
train_size = int(0.8 * len(dataset))
test_indices = list(range(train_size, len(dataset)))

print(f"Test set size: {len(test_indices)}")

# Load model
model = SimpleVolatilityLSTM(hidden_size=32)
model.load_state_dict(torch.load('results_simple_lstm/best_simple_lstm.pth', weights_only=True))
model.eval()

# Get predictions on test set
predictions_scaled = []
actuals_scaled = []

for idx in test_indices:
    X, y_scaled = dataset[idx]
    with torch.no_grad():
        pred_scaled = model(X.unsqueeze(0))
    predictions_scaled.append(pred_scaled.item())
    actuals_scaled.append(y_scaled.item())

predictions_scaled = np.array(predictions_scaled)
actuals_scaled = np.array(actuals_scaled)

# Inverse transform
predictions = dataset.target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
actuals = dataset.target_scaler.inverse_transform(actuals_scaled.reshape(-1, 1)).flatten()

print(f"\nPredictions statistics:")
print(f"  Min: {predictions.min():.6f}")
print(f"  Max: {predictions.max():.6f}")
print(f"  Mean: {predictions.mean():.6f}")
print(f"  Std: {predictions.std():.6f}")

print(f"\nActuals statistics:")
print(f"  Min: {actuals.min():.6f}")
print(f"  Max: {actuals.max():.6f}")
print(f"  Mean: {actuals.mean():.6f}")
print(f"  Std: {actuals.std():.6f}")

# Calculate metrics
metrics = evaluate_predictions(actuals, predictions)
print(f"\nTest Metrics:")
for name, value in metrics.items():
    if name == 'Directional_Acc':
        print(f"  {name}: {value:.2f}%")
    else:
        print(f"  {name}: {value:.6f}")
