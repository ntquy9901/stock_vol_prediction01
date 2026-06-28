"""
Train Enhanced LSTM-HAR on VN30 with Progress Tracking
Improved version with real-time progress monitoring
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
from tqdm import tqdm

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
print('ENHANCED LSTM-HAR - VN30-ONLY TRAINING (WITH PROGRESS)')
print('='*80)

# Create result directory
output_dir = Path('results/enhanced_lstm_har_vn30_2026-06-20')
output_dir.mkdir(exist_ok=True, parents=True)
print(f'Results will be saved to: {output_dir}')

# Create dataset
print('\n[1/4] Creating VN30 dataset...')
dataset = EnhancedHARDataset(data_dir, seq_length=seq_length, forecast_horizon=forecast_horizon)
print(f'  ✓ Dataset size: {len(dataset):,} samples')

# Split
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f'  ✓ Train: {len(train_dataset):,}, Val: {len(val_dataset):,}, Test: {len(test_dataset):,}')

# Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n[2/4] Creating model (device: {device})...')

model = EnhancedHARVolatilityLSTM(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
model = model.to(device)
print(f'  ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters')

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training
print(f'\n[3/4] Training (max {num_epochs} epochs, patience {patience})...')
print('='*80)

best_val_loss = float('inf')
best_epoch = 0
epochs_no_improve = 0

train_losses = []
val_losses = []

# Create progress bar
pbar = tqdm(range(num_epochs), desc='Training', unit='epoch')

for epoch in pbar:
    # Train
    model.train()
    train_loss = 0
    train_batches = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1

    train_loss /= train_batches
    train_losses.append(train_loss)

    # Validate
    model.eval()
    val_loss = 0
    val_batches = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()
            val_batches += 1

    val_loss /= val_batches
    val_losses.append(val_loss)

    # Update progress bar with metrics
    pbar.set_postfix({
        'Train': f'{train_loss:.6f}',
        'Val': f'{val_loss:.6f}',
        'Best': f'{best_val_loss:.6f}'
    })

    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_no_improve = 0
        # Save best model
        torch.save(model.state_dict(), output_dir / 'best_enhanced_lstm_har_vn30.pth')
        # Also save optimizer state for potential resume
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, output_dir / 'checkpoint.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'\n  ✓ Early stopping at epoch {epoch+1}')
            break

print(f'\n  ✓ Best epoch: {best_epoch+1}, Best val loss: {best_val_loss:.6f}')

# Save learning curves
print('\n  Saving learning curves...')
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot 1: Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss', linewidth=2)
plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val Loss', linewidth=2)
plt.axvline(x=best_epoch+1, color='g', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Loss Difference
plt.subplot(1, 2, 2)
loss_diff = np.array(val_losses) - np.array(train_losses)
plt.plot(range(1, len(loss_diff) + 1), loss_diff, 'purple', linewidth=2)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Epoch')
plt.ylabel('Val Loss - Train Loss')
plt.title('Overfitting Monitor')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
print(f'  ✓ Learning curves saved to: {output_dir / "learning_curves.png"}')

# Evaluate
print(f'\n[4/4] Evaluating on test set...')
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
print('ENHANCED LSTM-HAR (VN30-ONLY) - FINAL RESULTS')
print('='*80)

print(f'\n📊 Test Metrics:')
print(f'  MSE:  {metrics["mse"]:.8f}')
print(f'  RMSE: {metrics["rmse"]:.6f}')
print(f'  MAE:  {metrics["mae"]:.6f}')
print(f'  R²:   {metrics["r2"]:.6f}')
print(f'  QLIKE: {metrics["qlike"]:.6f}')
print(f'  Dir Acc: {metrics["directional_accuracy"]:.2f}%')

print('\n' + '='*80)
print('🎯 SUCCESS CRITERIA CHECK')
print('='*80)
rmse_status = "✅ PASS" if metrics["rmse"] < 0.20 else "❌ FAIL"
dir_acc_status = "✅ PASS" if metrics["directional_accuracy"] > 55 else "❌ FAIL"

print(f'  RMSE Target (<0.20): {rmse_status} - Actual: {metrics["rmse"]:.6f}')
print(f'  Dir Acc Target (>55%): {dir_acc_status} - Actual: {metrics["directional_accuracy"]:.2f}%')

print('\n' + '='*80)
print('📈 COMPARISON vs HAR-R BASELINE')
print('='*80)
print(f'  HAR-R Dir Acc: 51.53%')
print(f'  Enhanced LSTM-HAR Dir Acc: {metrics["directional_accuracy"]:.2f}%')
improvement = metrics["directional_accuracy"] - 51.53
print(f'  Improvement: {improvement:+.2f}% {"✅" if improvement > 0 else "❌"}')

print(f'\n  HAR-R RMSE: 0.000513')
print(f'  Enhanced LSTM-HAR RMSE: {metrics["rmse"]:.6f}')
rmse_diff = metrics["rmse"] - 0.000513
print(f'  Difference: {rmse_diff:+.6f} {"✅" if rmse_diff < 0 else "⚠️"}')

print(f'\n  HAR-R R²: 0.105')
print(f'  Enhanced LSTM-HAR R²: {metrics["r2"]:.6f}')
r2_improvement = (metrics["r2"] - 0.105) / 0.105 * 100
print(f'  Improvement: {r2_improvement:+.1f}% {"✅" if r2_improvement > 0 else "❌"}')

# Save results
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
    'training': {
        'best_epoch': best_epoch + 1,
        'best_val_loss': float(best_val_loss),
        'total_epochs': epoch + 1,
        'early_stopped': epochs_no_improve >= patience
    },
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
        'dir_acc_improvement': float(improvement),
        'har_r_rmse': 0.000513,
        'rmse_difference': float(rmse_diff),
        'har_r_r2': 0.105,
        'r2_improvement_percent': float(r2_improvement)
    },
    'files_generated': [
        'best_enhanced_lstm_har_vn30.pth',
        'checkpoint.pth',
        'learning_curves.png',
        'training_results.json'
    ]
}

with open(output_dir / 'training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + '='*80)
print('✅ TRAINING COMPLETE!')
print('='*80)
print(f'\n📁 Results saved to: {output_dir}')
print(f'   - Model: best_enhanced_lstm_har_vn30.pth')
print(f'   - Curves: learning_curves.png')
print(f'   - Results: training_results.json')
print(f'\n⏱️  Training completed in: {datetime.now().strftime("%H:%M:%S")}')
print('='*80)