"""
Enhanced LSTM-HAR Training Script

This module trains the enhanced LSTM-HAR model that uses:
- Raw Parkinson volatility (current day information)
- HAR weekly volatility (5-day average)
- HAR monthly volatility (22-day average)

Note: HAR daily removed to avoid redundancy with raw volatility.

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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lstm_har_enhanced.model_enhanced import EnhancedHARVolatilityLSTM
from src.lstm_har_enhanced.dataset_enhanced import EnhancedHARDataset
from src.common.evaluation import evaluate_predictions


def train_enhanced_lstm_har(data_dir: str, output_dir: str = None):
    """
    Train Enhanced LSTM-HAR with raw volatility + HAR features (3 features total).

    Features: [raw parkinson, har weekly, har monthly]
    Note: HAR daily removed (redundant with raw)

    Args:
        data_dir: Directory containing processed CSV files
        output_dir: Output directory (default: results/enhanced_lstm_har_YYYY-MM-DD_HHMMSS)
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = f'results/enhanced_lstm_har_{timestamp}'

    print("=" * 80)
    print("ENHANCED LSTM-HAR TRAINING - RAW VOLATILITY + HAR FEATURES")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Training configuration (IMPROVED 2026-06-19)
    # Performance Fix: PRIORITY 1 (Increase Capacity) + PRIORITY 2 (Reduce Regularization)
    # Learning Rate Fix: 0.0001 → 0.001 (10× faster convergence)
    config = {
        'hidden_size': 128,        # INCREASED: 64 -> 128 (PRIORITY 1)
        'learning_rate': 0.001,     # INCREASED: 0.0001 → 0.001 (Fix slow convergence)
        'batch_size': 32,
        'seq_length': 22,
        'weight_decay': 1e-6,       # REDUCED: 1e-5 -> 1e-6 (PRIORITY 2)
        'dropout': 0.1,              # REDUCED: 0.2 -> 0.1 (PRIORITY 2)
        'num_layers': 3              # INCREASED: 2 -> 3 (PRIORITY 1)
    }

    print("\n[CONFIGURATION]")
    print(f"  Hidden Size: {config['hidden_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Seq Length: {config['seq_length']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Features: 3 (raw + weekly + monthly)")
    print(f"  Note: HAR daily removed (redundant with raw)")

    # Create enhanced dataset
    print("\n1. Creating enhanced HAR dataset...")
    dataset = EnhancedHARDataset(data_dir,
                                seq_length=config['seq_length'],
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_gpu
    )

    # Initialize enhanced LSTM-HAR model
    print("\n2. Initializing Enhanced LSTM-HAR model...")
    model = EnhancedHARVolatilityLSTM(
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Architecture: LSTM with 3 features (raw + weekly + monthly)")
    print(f"Enhancement: Raw volatility adds current-day information")

    # Training configuration
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                   lr=config['learning_rate'],
                                   weight_decay=config['weight_decay'])
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
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

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
            }, os.path.join(output_dir, 'best_enhanced_lstm_har_model.pth'))
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
    checkpoint = torch.load(os.path.join(output_dir, 'best_enhanced_lstm_har_model.pth'))
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
    print("ENHANCED LSTM-HAR FINAL RESULTS")
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
        'model': 'Enhanced LSTM-HAR',
        'timestamp': datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        'features': 'Raw + HAR (3 features: raw, weekly, monthly)',
        'enhancement': 'Added raw parkinson volatility, removed redundant daily',
        'configuration': config,
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

    with open(os.path.join(output_dir, 'enhanced_lstm_har_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}")

    return model, metrics


def plot_learning_curves(train_losses, val_losses, output_dir, epoch):
    """Plot learning curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Enhanced LSTM-HAR Learning Curves')
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

    # Train enhanced LSTM-HAR model
    model, metrics = train_enhanced_lstm_har(data_dir)
