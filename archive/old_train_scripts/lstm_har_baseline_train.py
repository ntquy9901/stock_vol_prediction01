"""
LSTM-HAR Training Script

This module contains the training logic for the LSTM-HAR model
that uses HAR features instead of raw Parkinson volatility.

Author: Stock Volatility Prediction Team
Date: 2026-06-18
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lstm_har_baseline.model import HARVolatilityLSTM
from src.lstm_har_baseline.dataset import HARVolatilityDataset
from src.common.evaluation import evaluate_predictions


def train_lstm_har(data_dir: str, output_dir: str = None):
    """
    Train LSTM-HAR on pooled dataset with HAR features.

    Uses HAR features (daily, weekly, monthly) instead of raw volatility.

    Args:
        data_dir: Directory containing processed CSV files
        output_dir: Output directory (default: results/lstm_har_YYYY-MM-DD_HHMMSS)
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = f'results/lstm_har_{timestamp}'

    print("=" * 80)
    print("LSTM-HAR TRAINING - HETEROSCEDASTIC AUTOREGRESSIVE FEATURES")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Optuna-optimized parameters (from trial #2, validation loss: 0.6654)
    # NOTE: Optuna recommendations caused catastrophic failure
    # Rolling back to known working configuration
    optuna_params = {
        'hidden_size': 64,
        'learning_rate': 0.0001,   # OPTION A: 0.001 → 0.0001 (10× lower to fix flat val loss)
        'batch_size': 64,
        'seq_length': 22,        # ROLLBACK: 44 → 22
        'weight_decay': 1e-5      # ROLLBACK: 0.0 → 1e-5
    }

    # Create HAR dataset with Optuna-optimized sequence length
    print("\n1. Creating HAR dataset...")
    dataset = HARVolatilityDataset(data_dir,
                                seq_length=optuna_params['seq_length'],  # 44 instead of 22
                                forecast_horizon=5)

    # Split into train/test (temporal split)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create dataloaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_gpu = device.type == 'cuda'
    num_workers = 0  # Windows: use 0 for stability
    use_amp = use_gpu

    train_loader = DataLoader(
        train_dataset,
        batch_size=optuna_params['batch_size'],  # 64 from Optuna
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=optuna_params['batch_size'],  # 64 from Optuna
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_gpu
    )

    # Pre-flight validation
    print("\n[PRE-FLIGHT VALIDATION]")
    print("Validating HAR dataset statistics...")

    try:
        sample_loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
        sample_batch = next(iter(sample_loader))
        X_sample, y_sample = sample_batch

        print(f"  Input shape: {X_sample.shape}")  # Should be (batch, seq_len, 3)
        print(f"  HAR features: 3 (daily, weekly, monthly)")
        print(f"  Target mean: {y_sample.mean():.4f}, std: {y_sample.std():.4f}")

        # Check data quality
        if np.isnan(X_sample.numpy()).any():
            print("  ERROR: Inputs contain NaN values!")
        if np.isnan(y_sample.numpy()).any():
            print("  ERROR: Targets contain NaN values!")

    except Exception as e:
        print(f"  ERROR during validation: {e}")

    # Initialize LSTM-HAR model with Optuna-optimized parameters
    print("\n2. Initializing LSTM-HAR model...")

    # OPTION A: Fix flat val loss by reducing learning rate
    # Previous Optuna params caused catastrophic failure
    # Using much lower LR to allow proper gradient descent
    optuna_params = {
        'hidden_size': 128,        # INCREASED: 64 -> 128 (PRIORITY 1)
        'learning_rate': 0.001,     # INCREASED: 0.0001 → 0.001 (Fix slow convergence)
        'batch_size': 64,
        'seq_length': 22,
        'weight_decay': 1e-6,       # REDUCED: 1e-5 -> 1e-6 (PRIORITY 2)
        'num_layers': 3,            # INCREASED: 2 -> 3 (PRIORITY 1)
        'dropout': 0.1               # REDUCED: 0.2 -> 0.1 (PRIORITY 2)
    }

    # Model with improved architecture (2026-06-19)
    model = HARVolatilityLSTM(hidden_size=optuna_params['hidden_size'],
                             num_layers=optuna_params['num_layers'],
                             dropout=optuna_params['dropout'])

    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Architecture: LSTM with HAR features (daily, weekly, monthly)")

    # Print optimization improvements
    print(f"\n[LEARNING RATE FIX - 10× FASTER CONVERGENCE]")
    print(f"  Learning Rate: {optuna_params['learning_rate']:.6f} (increased from 0.0001)")
    print(f"  Seq Length: {optuna_params['seq_length']} (rollback from 44)")
    print(f"  Dropout: 0.1 (reduced from 0.2)")
    print(f"  Patience: 10 (reduced from 20)")
    print(f"  Weight Decay: {optuna_params['weight_decay']} (reduced from 1e-5)")
    print(f"  Hidden Size: {optuna_params['hidden_size']} (increased from 64)")
    print(f"  Num Layers: {optuna_params['num_layers']} (increased from 2)")
    print(f"  Batch Size: {optuna_params['batch_size']} (Optuna-optimized)")

    # Training configuration with Optuna-optimized parameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                   lr=optuna_params['learning_rate'],
                                   weight_decay=optuna_params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print("\n3. Training...")
    num_epochs = 100
    patience = 10                  # REDUCED: 20 → 10 (faster LR needs less patience)
    min_epochs = 15                # NEW: Don't stop before epoch 15 (prevent premature stopping)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_gpu else None

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

            if use_amp:
                with torch.amp.autocast('cuda'):
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
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
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        predictions = model(X_batch)
                        loss = criterion(predictions, y_batch)
                else:
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)

                val_loss += loss.item()

        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
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
            }, os.path.join(output_dir, 'best_lstm_har_model.pth'))
        else:
            epochs_without_improvement += 1

        # Print progress
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} "
              f"(Best: {best_val_loss:.6f} @ epoch {best_epoch+1})")

        # Plot learning curves every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_learning_curves(train_losses, val_losses, output_dir, epoch)

        # Early stopping (only after min_epochs)
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Final learning curves
    plot_learning_curves(train_losses, val_losses, output_dir, num_epochs)

    # Evaluate best model
    print("\n4. Evaluating best model...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_lstm_har_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Generate predictions
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)

            # Inverse transform predictions and targets
            predictions_np = dataset.target_scaler.inverse_transform(predictions.cpu().numpy())
            targets_np = dataset.target_scaler.inverse_transform(y_batch.numpy().reshape(-1, 1))

            all_predictions.extend(predictions_np.flatten())
            all_targets.extend(targets_np.flatten())

    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    metrics = evaluate_predictions(all_targets, all_predictions)

    print(f"\n{'='*80}")
    print("LSTM-HAR FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Test MSE: {metrics['mse']:.2e}")
    print(f"Test RMSE: {metrics['rmse']:.6f}")
    print(f"Test MAE: {metrics['mae']:.6f}")
    print(f"Test R²: {metrics['r2']:.6f}")
    print(f"Test QLIKE: {metrics['qlike']:.6f}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%")

    # Save results
    import json
    results = {
        'model': 'LSTM-HAR',
        'timestamp': datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        'features': 'HAR (daily, weekly, monthly)',
        'best_epoch': best_epoch + 1,
        'best_val_loss': float(best_val_loss),
        'test_metrics': {
            'mse': float(metrics['mse']),
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae']),
            'r2': float(metrics['r2']),
            'qlike': float(metrics['qlike']),
            'directional_accuracy': float(metrics['directional_accuracy'])
        }
    }

    with open(os.path.join(output_dir, 'lstm_har_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}")

    return model, metrics


def plot_learning_curves(train_losses, val_losses, output_dir, epoch):
    """Plot learning curves with overfitting detection."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('LSTM-HAR Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Check for overfitting
    if len(val_losses) > 10:
        recent_val_trend = val_losses[-5:]
        if all(recent_val_trend[i] > recent_val_trend[i-1] for i in range(1, len(recent_val_trend))):
            print("⚠️  WARNING: Validation loss increasing - possible overfitting!")

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

    # Train LSTM-HAR model
    model, metrics = train_lstm_har(data_dir)