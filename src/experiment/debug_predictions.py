"""Debug model predictions"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from src.lstm_baseline.dataset import PooledVolatilityDataset
from src.lstm_baseline.model import SimpleVolatilityLSTM

# Load dataset
dataset = PooledVolatilityDataset('data/processed', seq_length=22, forecast_horizon=5)

# Get a sample
X_sample, y_sample = dataset[0]

print("Input shape:", X_sample.shape)
print("Target (scaled):", y_sample.item())

# Load model
model = SimpleVolatilityLSTM(hidden_size=32)
model.load_state_dict(torch.load('results_simple_lstm/best_simple_lstm.pth', weights_only=True))
model.eval()

# Predict
with torch.no_grad():
    pred = model(X_sample.unsqueeze(0))
    
print("Prediction (scaled):", pred.item())

# Inverse transform
pred_raw = dataset.target_scaler.inverse_transform([[pred.item()]])[0, 0]
y_raw = dataset.target_scaler.inverse_transform([[y_sample.item()]])[0, 0]

print("\nAfter inverse-transform:")
print(f"Prediction (raw): {pred_raw:.6f}")
print(f"Target (raw): {y_raw:.6f}")
print(f"Difference: {abs(pred_raw - y_raw):.6f}")

# Check a few more samples
print("\nChecking 10 samples:")
errors = []
for i in range(10):
    X, y = dataset[i]
    with torch.no_grad():
        pred = model(X.unsqueeze(0))
    pred_raw = dataset.target_scaler.inverse_transform([[pred.item()]])[0, 0]
    y_raw = dataset.target_scaler.inverse_transform([[y.item()]])[0, 0]
    error = abs(pred_raw - y_raw)
    errors.append(error)
    print(f"  Sample {i}: pred={pred_raw:.6f}, actual={y_raw:.6f}, error={error:.6f}")

print(f"\nMean error: {np.mean(errors):.6f}")
print(f"Max error: {np.max(errors):.6f}")
