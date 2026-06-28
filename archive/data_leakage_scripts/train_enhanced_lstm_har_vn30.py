"""
Train Enhanced LSTM-HAR on VN30-only data
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.lstm_har_enhanced.dataset_enhanced import EnhancedHARDataset
from src.lstm_har_enhanced.model_enhanced import EnhancedHARVolatilityLSTM
from src.common.evaluation import evaluate_predictions

# Parameters
data_dir = 'data/processed/vn30_only'
seq_length = 22
forecast_horizon = 5
batch_size = 32
hidden_size = 128
num_layers = 3
dropout = 0.2
learning_rate = 0.0005
weight_decay = 1e-5
num_epochs = 70
patience = 15

print('='*80)
print('ENHANCED LSTM-HAR - VN30-ONLY TRAINING')
print('='*80)

# Create result directory
output_dir = Path('results/enhanced_lstm_har_vn30_2026-06-20')
output_dir.mkdir(exist_ok=True, parents=True)
print(f'Results will be saved to: {output_dir}')

# Create dataset
print('\n1. Creating VN30 dataset...')
dataset = EnhancedHARDataset(data_dir, seq_length=seq_length, forecast_horizon=forecast_horizon)
print(f'  Dataset size: {len(dataset)}')

# Split
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f'  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')

# Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n2. Creating model (device: {device})...')

model = EnhancedHARVolatilityLSTM(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
model = model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training
print(f'\n3. Training (max {num_epochs} epochs)...')

best_val_loss = float('inf')
best_epoch = 0
epochs_no_improve = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save(model.state_dict(), output_dir / 'best_enhanced_lstm_har_vn30.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'  Early stopping at epoch {epoch+1}')
            break

    # Print progress every epoch
    print(f'  Epoch {epoch+1:2d}/{num_epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}', end='')
    if val_loss < best_val_loss:
        print(' ✓ (Best)')
    else:
        print('')

print(f'\n  Best epoch: {best_epoch+1}, Best val loss: {best_val_loss:.6f}')

# Evaluate
print('\n4. Evaluating on test set...')
model.load_state_dict(torch.load(output_dir / 'best_enhanced_lstm_har_vn30.pth'))
model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        predictions = model(X_batch)

        predictions_np = dataset.target_scaler.inverse_transform(predictions.cpu().numpy())
        targets_np = dataset.target_scaler.inverse_transform(y_batch.numpy().reshape(-1, 1))

        all_predictions.extend(predictions_np.flatten())
        all_targets.extend(targets_np.flatten())

y_pred = np.array(all_predictions)
y_true = np.array(all_targets)

metrics = evaluate_predictions(y_true, y_pred)

print('\n' + '='*80)
print('ENHANCED LSTM-HAR (VN30-ONLY) RESULTS')
print('='*80)

print(f'\nTest Metrics:')
print(f'  MSE:  {metrics["mse"]:.6f}')
print(f'  RMSE: {metrics["rmse"]:.6f}')
print(f'  MAE:  {metrics["mae"]:.6f}')
print(f'  R²:   {metrics["r2"]:.6f}')
print(f'  QLIKE: {metrics["qlike"]:.6f}')
print(f'  Dir Acc: {metrics["directional_accuracy"]:.2f}%')

print('\n' + '='*80)
print('SUCCESS CRITERIA CHECK')
print('='*80)
print(f'  RMSE Target (<0.20): {"PASS" if metrics["rmse"] < 0.20 else "FAIL"} - Actual: {metrics["rmse"]:.6f}')
print(f'  Dir Acc Target (>55%): {"PASS" if metrics["directional_accuracy"] > 55 else "FAIL"} - Actual: {metrics["directional_accuracy"]:.2f}%')

print('\n' + '='*80)
print('FINAL COMPARISON vs HAR-R BASELINE')
print('='*80)
print(f'  HAR-R Dir Acc: 51.53%')
print(f'  Enhanced LSTM-HAR Dir Acc: {metrics["directional_accuracy"]:.2f}%')
print(f'  Improvement: {metrics["directional_accuracy"] - 51.53:.2f}%')

print(f'\n  HAR-R RMSE: 0.000513')
print(f'  Enhanced LSTM-HAR RMSE: {metrics["rmse"]:.6f}')

# Save results
import json
results = {
    'model': 'Enhanced LSTM-HAR (VN30-Only)',
    'dataset': '30 VN30 stocks',
    'timestamp': datetime.now().strftime('%Y-%m-%d_%H%M%S'),
    'configuration': {
        'hidden_size': hidden_size,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'weight_decay': weight_decay
    },
    'best_epoch': best_epoch + 1,
    'best_val_loss': float(best_val_loss),
    'test_metrics': {
        'mse': float(metrics['mse']),
        'rmse': float(metrics['rmse']),
        'mae': float(metrics['mae']),
        'r2': float(metrics['r2']),
        'qlike': float(metrics['qlike']),
        'directional_accuracy': float(metrics['directional_accuracy'])
    },
    'comparison_with_baseline': {
        'har_r_dir_acc': 51.53,
        'dir_acc_improvement': float(metrics['directional_accuracy'] - 51.53),
        'har_r_rmse': 0.000513,
        'rmse_difference': float(metrics['rmse'] - 0.000513)
    }
}

with open(output_dir / 'training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n✅ Training complete! Results saved to: {output_dir}')