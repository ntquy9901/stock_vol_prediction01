"""
Simple LSTM Training Script with 3-Way Temporal Split

This module trains the Simple LSTM model using:
- 70% train data (2006-2020)
- 15% validation data (2020-2021) - for early stopping
- 15% test data (2021-2026) - final evaluation

Key improvements:
- Proper temporal split (no data leakage)
- Validation during training (early stopping)
- Compare val vs test metrics (detect overfitting)

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm_baseline.model import SimpleVolatilityLSTM
from lstm_baseline.dataset import PooledVolatilityDataset
from common.temporal_split import TemporalSplitter
from common.evaluation import evaluate_predictions


def train_simple_lstm_with_val(data_dir: str, output_dir: str = None):
    """
    Train Simple LSTM with 3-way temporal split (70/15/15).

    Args:
        data_dir: Directory containing processed CSV files
        output_dir: Output directory (default: results/simple_lstm_val_YYYY-MM-DD_HHMMSS)
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = f'results/simple_lstm_val_{timestamp}'

    print("=" * 80)
    print("SIMPLE LSTM TRAINING - 3-WAY TEMPORAL SPLIT (70/15/15)")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Training configuration (OVERFITTING FIX 2026-06-20)
    # Overfitting Fix: Strengthen regularization to improve generalization
    # Learning Rate Fix: 0.001 → 0.0005 (more stable convergence)
    # Note: Simple LSTM has 1 layer by design, so dropout not applicable
    config = {
        'hidden_size': 128,        # Already optimal
        'learning_rate': 0.0005,    # REDUCED: 0.001 → 0.0005 (prevent overfitting)
        'batch_size': 32,
        'seq_length': 22,
        'weight_decay': 1e-5,       # INCREASED: 1e-6 -> 1e-5 (stronger L2 regularization)
        'dropout': 0.1,              # Kept for consistency (not used for 1-layer)
        'num_layers': 1              # 1 layer by design for simple model
    }

    print("\n[CONFIGURATION]")
    print(f"  Hidden Size: {config['hidden_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Seq Length: {config['seq_length']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Num Layers: {config['num_layers']}")
    print(f"  Features: 1 (raw parkinson volatility)")
    print(f"  Train/Val/Test Split: 70% / 15% / 15%")

    # Create dataset
    print("\n1. Creating pooled dataset...")
    dataset = PooledVolatilityDataset(data_dir,
                                seq_length=config['seq_length'],
                                forecast_horizon=5)

    print(f"Total dataset size: {len(dataset)} sequences")

    # Create temporal splitter (70/15/15)
    print("\n2. Performing temporal split...")
    splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    split_info = splitter.get_info(len(dataset))

    print(f"Split plan:")
    print(f"  Train: {split_info['train_size']} sequences (70%)")
    print(f"  Val:   {split_info['val_size']} sequences (15%)")
    print(f"  Test:  {split_info['test_size']} sequences (15%)")

    # Create dataloaders with temporal split
    print("\n3. Creating dataloaders...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_gpu = device.type == 'cuda'
    num_workers = 0  # Windows: use 0 for stability

    train_loader, val_loader, test_loader = splitter.create_dataloaders(
        dataset,
        batch_size=config['batch_size'],
        num_workers=num_workers,
        pin_memory=use_gpu
    )

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Initialize Simple LSTM model
    print("\n4. Initializing Simple LSTM model...")
    model = SimpleVolatilityLSTM(hidden_size=config['hidden_size'])

    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: 1-layer LSTM with raw parkinson volatility")

    # Training configuration
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                   lr=config['learning_rate'],
                                   weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop with validation
    print("\n5. Training with validation...")
    num_epochs = 100
    patience = 20                  # INCREASED: 10 → 20 (give model more time to improve)
    min_epochs = 15                # NEW: Don't stop before epoch 15 (prevent premature stopping)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    print(f"Starting training: {num_epochs} epochs")
    epoch_times = []
    best_epoch = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model_val.pth'))
        else:
            epochs_without_improvement += 1

        # Print progress
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} "
              f"(Best Val: {best_val_loss:.6f} @ epoch {best_epoch+1})")

        # Plot learning curves every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_learning_curves(train_losses, val_losses, output_dir, epoch)

        # Early stopping (only after min_epochs)
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Final learning curves
    plot_learning_curves(train_losses, val_losses, output_dir, num_epochs)

    # Evaluate best model on validation set
    print("\n6. Evaluating best model on validation set...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model_val.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Generate predictions on validation set
    model.eval()
    all_predictions_val = []
    all_targets_val = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)

            # Inverse transform
            predictions_np = dataset.target_scaler.inverse_transform(predictions.cpu().numpy())
            targets_np = dataset.target_scaler.inverse_transform(y_batch.numpy().reshape(-1, 1))

            all_predictions_val.extend(predictions_np.flatten())
            all_targets_val.extend(targets_np.flatten())

    # Calculate validation metrics
    y_pred_val = np.array(all_predictions_val)
    y_true_val = np.array(all_targets_val)

    val_metrics = evaluate_predictions(y_true_val, y_pred_val)

    print(f"\n{'='*80}")
    print("VALIDATION RESULTS (Best Model @ Epoch {})".format(best_epoch + 1))
    print(f"{'='*80}")
    print(f"Val MSE: {val_metrics['mse']:.2e}")
    print(f"Val RMSE: {val_metrics['rmse']:.6f}")
    print(f"Val MAE: {val_metrics['mae']:.6f}")
    print(f"Val R²: {val_metrics['r2']:.6f}")
    print(f"Val QLIKE: {val_metrics['qlike']:.6f}")
    print(f"Val Dir Acc: {val_metrics['directional_accuracy']:.2f}%")

    # Evaluate best model on test set
    print("\n7. Evaluating best model on test set...")

    all_predictions_test = []
    all_targets_test = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)

            # Inverse transform
            predictions_np = dataset.target_scaler.inverse_transform(predictions.cpu().numpy())
            targets_np = dataset.target_scaler.inverse_transform(y_batch.numpy().reshape(-1, 1))

            all_predictions_test.extend(predictions_np.flatten())
            all_targets_test.extend(targets_np.flatten())

    # Calculate test metrics
    y_pred_test = np.array(all_predictions_test)
    y_true_test = np.array(all_targets_test)

    test_metrics = evaluate_predictions(y_true_test, y_pred_test)

    print(f"\n{'='*80}")
    print("TEST RESULTS (Best Model @ Epoch {})".format(best_epoch + 1))
    print(f"{'='*80}")
    print(f"Test MSE: {test_metrics['mse']:.2e}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test R²: {test_metrics['r2']:.6f}")
    print(f"Test QLIKE: {test_metrics['qlike']:.6f}")
    print(f"Test Dir Acc: {test_metrics['directional_accuracy']:.2f}%")

    # Compare val vs test metrics
    print(f"\n{'='*80}")
    print("VALIDATION VS TEST COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':<15} {'Validation':<15} {'Test':<15} {'Difference':<15}")
    print("-" * 60)

    mse_diff = test_metrics['mse'] - val_metrics['mse']
    rmse_diff = test_metrics['rmse'] - val_metrics['rmse']
    mae_diff = test_metrics['mae'] - val_metrics['mae']
    r2_diff = test_metrics['r2'] - val_metrics['r2']
    qlike_diff = test_metrics['qlike'] - val_metrics['qlike']
    dir_acc_diff = test_metrics['directional_accuracy'] - val_metrics['directional_accuracy']

    print(f"{'MSE':<15} {val_metrics['mse']:<15.2e} {test_metrics['mse']:<15.2e} {mse_diff:>+14.2e}")
    print(f"{'RMSE':<15} {val_metrics['rmse']:<15.6f} {test_metrics['rmse']:<15.6f} {rmse_diff:>+14.6f}")
    print(f"{'MAE':<15} {val_metrics['mae']:<15.6f} {test_metrics['mae']:<15.6f} {mae_diff:>+14.6f}")
    print(f"{'R²':<15} {val_metrics['r2']:<15.6f} {test_metrics['r2']:<15.6f} {r2_diff:>+14.6f}")
    print(f"{'QLIKE':<15} {val_metrics['qlike']:<15.6f} {test_metrics['qlike']:<15.6f} {qlike_diff:>+14.6f}")
    print(f"{'Dir Acc':<15} {val_metrics['directional_accuracy']:<15.2f}% {test_metrics['directional_accuracy']:<15.2f}% {dir_acc_diff:>+14.2f}%")

    # Overfitting check
    print(f"\n{'='*80}")
    print("OVERFITTING CHECK")
    print(f"{'='*80}")

    # Check if test performance is significantly worse than validation
    if rmse_diff > 0.0001:  # More than 10% degradation in RMSE
        print("[WARNING] Test RMSE significantly higher than Val RMSE - Possible overfitting!")
    if r2_diff < -0.05:  # More than 5% degradation in R²
        print("[WARNING] Test R² significantly lower than Val R² - Possible overfitting!")
    if dir_acc_diff < -2:  # More than 2% degradation in Dir Acc
        print("[WARNING] Test Dir Acc significantly lower than Val Dir Acc - Possible overfitting!")

    if rmse_diff <= 0.0001 and r2_diff >= -0.05 and dir_acc_diff >= -2:
        print("[OK] Test performance similar to validation - No significant overfitting")

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    results = {
        'model': 'Simple LSTM with Validation',
        'timestamp': timestamp,
        'features': 'Raw Parkinson volatility (1 feature)',
        'split_method': 'Temporal (70/15/15)',
        'configuration': config,
        'best_epoch': best_epoch + 1,
        'best_val_loss': float(best_val_loss),
        'validation_metrics': {
            'mse': float(val_metrics['mse']),
            'rmse': float(val_metrics['rmse']),
            'mae': float(val_metrics['mae']),
            'r2': float(val_metrics['r2']),
            'qlike': float(val_metrics['qlike']),
            'directional_accuracy': float(val_metrics['directional_accuracy'])
        },
        'test_metrics': {
            'mse': float(test_metrics['mse']),
            'rmse': float(test_metrics['rmse']),
            'mae': float(test_metrics['mae']),
            'r2': float(test_metrics['r2']),
            'qlike': float(test_metrics['qlike']),
            'directional_accuracy': float(test_metrics['directional_accuracy'])
        },
        'val_test_diff': {
            'mse_diff': float(mse_diff),
            'rmse_diff': float(rmse_diff),
            'mae_diff': float(mae_diff),
            'r2_diff': float(r2_diff),
            'qlike_diff': float(qlike_diff),
            'dir_acc_diff': float(dir_acc_diff)
        }
    }

    with open(os.path.join(output_dir, 'simple_lstm_val_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}")

    return model, val_metrics, test_metrics


def plot_learning_curves(train_losses, val_losses, output_dir, epoch):
    """Plot learning curves with train and validation."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Simple LSTM Learning Curves (with Validation)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    plot_path = os.path.join(output_dir, f'learning_curves_epoch_{epoch+1}_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[OK] Learning curves saved: {plot_path}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    data_dir = 'data/processed'

    # Train simple LSTM with validation
    model, val_metrics, test_metrics = train_simple_lstm_with_val(data_dir)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Final Validation RMSE: {val_metrics['rmse']:.6f}")
    print(f"Final Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"RMSE Difference (Test - Val): {test_metrics['rmse'] - val_metrics['rmse']:+.6f}")
    print("="*80)
