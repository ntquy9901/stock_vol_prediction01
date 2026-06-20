"""
Debug script to monitor what happens during first few training epochs
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from src.cryptomamba_baseline.model_enhanced import create_cryptomamba_model_enhanced
from src.common.har_features import generate_har_features

# Load and prepare data
data_dir = Path("data/processed")
data_files = list(data_dir.glob("*.csv"))
df = pd.read_csv(data_files[0])
df = generate_har_features(df)

# Create sequences
seq_length = 22
forecast_horizon = 5

X_list = []
y_list = []

for i in range(len(df) - seq_length - forecast_horizon):
    X_seq = df[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].iloc[i:i+seq_length].values
    y_target = df['parkinson_volatility'].iloc[i + seq_length + forecast_horizon - 1]

    X_list.append(X_seq)
    y_list.append(y_target)

X = torch.FloatTensor(np.array(X_list))
y = torch.FloatTensor(np.array(y_list)).unsqueeze(1)

# Split temporally
n = len(X)
train_end = int(0.70 * n)

X_train, y_train = X[:train_end], y[:train_end]

print(f"Training data: {len(X_train)} samples")
print(f"X_train range: [{X_train.min():.8f}, {X_train.max():.8f}]")
print(f"y_train range: [{y_train.min():.8f}, {y_train.max():.8f}]")

# Create model
model = create_cryptomamba_model_enhanced()

# Test BEFORE training
print(f"\n=== BEFORE TRAINING ===")
model.eval()
with torch.no_grad():
    pred_before = model(X_train[:10])
print(f"Predictions range: [{pred_before.min():.8f}, {pred_before.max():.8f}]")
print(f"Predictions mean: {pred_before.mean():.8f}")
print(f"Targets range: [{y_train[:10].min():.8f}, {y_train[:10].max():.8f}]")

# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
criterion = nn.MSELoss()

# Train for 5 epochs
print(f"\n=== TRAINING FOR 5 EPOCHS ===")
for epoch in range(5):
    model.train()

    # Simple batch (first 100 samples)
    batch_X = X_train[:100]
    batch_y = y_train[:100]

    optimizer.zero_grad()

    predictions = model(batch_X)
    loss = criterion(predictions, batch_y)

    loss.backward()
    optimizer.step()

    # Check predictions
    model.eval()
    with torch.no_grad():
        pred_after = model(batch_X[:10])

    print(f"\nEpoch {epoch+1}:")
    print(f"  Loss: {loss.item():.8f}")
    print(f"  Predictions range: [{pred_after.min():.8f}, {pred_after.max():.8f}]")
    print(f"  Predictions mean: {pred_after.mean():.8f}")
    print(f"  Targets range: [{batch_y[:10].min():.8f}, {batch_y[:10].max():.8f}]")
    print(f"  % Zeros: {(pred_after == 0).float().mean().item() * 100:.1f}%")

    # Check gradient norms
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"  Gradient norm: {total_norm:.6f}")

    model.train()

print(f"\n=== AFTER 5 EPOCHS ===")
model.eval()
with torch.no_grad():
    final_pred = model(X_train[:100])
print(f"Final predictions range: [{final_pred.min():.8f}, {final_pred.max():.8f}]")
print(f"Final predictions mean: {final_pred.mean():.8f}")
print(f"Final % Zeros: {(final_pred == 0).float().mean().item() * 100:.1f}%")
