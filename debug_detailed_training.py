"""
Detailed debug of training process with normalization
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from src.cryptomamba_baseline.model_enhanced import create_cryptomamba_model_enhanced
from src.common.har_features import generate_har_features
from src.common.data_normalization import normalize_for_training, denormalize_predictions

# Load data
data_dir = Path('data/processed')
df = pd.read_csv(list(data_dir.glob('*.csv'))[0])
df = generate_har_features(df)

# Create sequences
seq_length = 22
forecast_horizon = 5
X_list, y_list = [], []

for i in range(len(df) - seq_length - forecast_horizon):
    X_seq = df[['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']].iloc[i:i+seq_length].values
    y_target = df['parkinson_volatility'].iloc[i + seq_length + forecast_horizon - 1]
    X_list.append(X_seq)
    y_list.append(y_target)

X = np.array(X_list)
y = np.array(y_list).reshape(-1, 1)

# Normalize
X_norm, y_norm, target_normalizer, feature_stats = normalize_for_training(X, y)

# Split
n = len(X_norm)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_train = torch.FloatTensor(X_norm[:train_end])
y_train = torch.FloatTensor(y_norm[:train_end])
X_val = torch.FloatTensor(X_norm[train_end:val_end])
y_val = torch.FloatTensor(y_norm[train_end:val_end])

print(f'Training data: {len(X_train)} samples')
print(f'y_train norm range: [{y_train.min():.3f}, {y_train.max():.3f}]')

# Create model
model = create_cryptomamba_model_enhanced()
print(f'\\nModel created with 116,161 parameters')

# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
criterion = nn.MSELoss()

print(f'\\n=== DETAILED TRAINING DEBUG ===')

for epoch in range(5):
    model.train()

    # Train on first batch
    batch_X = X_train[:100]
    batch_y = y_train[:100]

    optimizer.zero_grad()

    predictions_norm = model(batch_X)
    loss = criterion(predictions_norm, batch_y)

    loss.backward()
    optimizer.step()

    # Check what happened
    model.eval()
    with torch.no_grad():
        train_pred_norm = model(batch_X[:10])
        val_pred_norm = model(X_val[:10])

        # Denormalize for inspection
        train_pred_orig = target_normalizer.inverse_transform(train_pred_norm.numpy().flatten())
        val_pred_orig = target_normalizer.inverse_transform(val_pred_norm.numpy().flatten())
        val_targets_orig = target_normalizer.inverse_transform(y_val[:10].numpy().flatten())

    print(f'\\nEpoch {epoch+1}:')
    print(f'  Loss: {loss.item():.6f}')
    print(f'  Train pred_norm: [{train_pred_norm.min():.3f}, {train_pred_norm.max():.3f}], mean={train_pred_norm.mean():.3f}')
    print(f'  Val pred_norm: [{val_pred_norm.min():.3f}, {val_pred_norm.max():.3f}], mean={val_pred_norm.mean():.3f}')
    print(f'  Train pred_orig: [{train_pred_orig.min():.6f}, {train_pred_orig.max():.6f}]')
    print(f'  Val pred_orig: [{val_pred_orig.min():.6f}, {val_pred_orig.max():.6f}]')
    print(f'  Val targets_orig: [{val_targets_orig.min():.6f}, {val_targets_orig.max():.6f}]')
    print(f'  Train %Zeros: {(train_pred_norm==0).float().mean()*100:.0f}%')
    print(f'  Val %Zeros: {(val_pred_norm==0).float().mean()*100:.0f}%')

    # Check gradients
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f'  Gradient norm: {total_norm:.6f}')

    model.train()

print(f'\\n=== FINAL CHECK ===')
model.eval()
with torch.no_grad():
    final_train_pred = model(X_train[:100])
    final_val_pred = model(X_val[:100])

print(f'Final train pred_norm: [{final_train_pred.min():.3f}, {final_train_pred.max():.3f}]')
print(f'Final val pred_norm: [{final_val_pred.min():.3f}, {final_val_pred.max():.3f}]')
print(f'Final train %Zeros: {(final_train_pred==0).float().mean()*100:.0f}%')
print(f'Final val %Zeros: {(final_val_pred==0).float().mean()*100:.0f}%')

if final_train_pred.max().item() > 0.1 and final_train_pred.min().item() > -0.1:
    print(f'\\n[POSSIBLE SUCCESS] Predictions have reasonable variance')
    print(f'But may need more training to converge to correct values')
else:
    print(f'\\n[STILL FAILING] Model predictions still problematic')
