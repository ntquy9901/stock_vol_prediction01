"""
TimesNet Training Script for Volatility Prediction

Implements TimesNet training with:
- Temporal data split (70/15/15)
- 6 mandatory metrics (MSE, RMSE, MAE, R², QLIKE, Dir Acc)
- Early stopping with patience=15
- Learning curve plotting
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.timesnet_baseline.model import create_timesnet_model
from src.timesnet_baseline.config import TimesNetConfig
from src.timesnet_baseline.dataset import create_timesnet_dataloaders
from src.common.evaluation import evaluate_predictions


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=15, min_delta=1e-6, min_epochs=20):
        """
        Args:
            patience: Epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            min_epochs: Minimum epochs before stopping can trigger
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.epoch = 0

    def __call__(self, val_loss, epoch):
        """
        Check if should stop training

        Args:
            val_loss: Current validation loss
            epoch: Current epoch number

        Returns:
            True if should stop, False otherwise
        """
        self.epoch = epoch

        # Don't stop before minimum epochs
        if epoch < self.min_epochs:
            return False

        # First epoch
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        # Check for improvement
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        # Check if patience exceeded
        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


def plot_learning_curves(train_losses, val_losses, save_path):
    """
    Plot training and validation learning curves

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Zoomed view (last 30 epochs)
    if len(train_losses) > 30:
        axes[1].plot(range(len(train_losses)-30, len(train_losses)),
                   train_losses[-30:], label='Train Loss', linewidth=2)
        axes[1].plot(range(len(val_losses)-30, len(val_losses)),
                   val_losses[-30:], label='Val Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Loss (Last 30 Epochs)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[plot_learning_curves] Saved plot to {save_path}")


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """
    Train for one epoch

    Args:
        model: TimesNet model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        config: Training configuration

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (x_har, x_temporal, y) in enumerate(dataloader):
        # Move to device
        x_har = x_har.to(device)
        x_temporal = x_temporal.to(device)
        y = y.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(x_har, x_temporal)

        # Compute loss
        loss = criterion(predictions, y)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Print progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx+1}/{len(dataloader)}: Loss={loss.item():.6f}", flush=True)

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"  Average Train Loss: {avg_loss:.6f}", flush=True)
    return avg_loss


def validate(model, dataloader, criterion, device, dataset_for_denorm):
    """
    Validate model

    Args:
        model: TimesNet model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        dataset_for_denorm: Dataset for denormalization

    Returns:
        Tuple of (avg_loss, metrics_dict, all_predictions, all_targets)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_har, x_temporal, y in dataloader:
            # Move to device
            x_har = x_har.to(device)
            x_temporal = x_temporal.to(device)
            y = y.to(device)

            # Forward pass
            predictions = model(x_har, x_temporal)

            # Compute loss
            loss = criterion(predictions, y)
            total_loss += loss.item()
            n_batches += 1

            # Store predictions and targets
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # Calculate average loss
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

    # Denormalize predictions and targets
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    if hasattr(dataset_for_denorm, 'dataset'):
        denorm_func = dataset_for_denorm.dataset.denormalize_predictions
    else:
        denorm_func = dataset_for_denorm.denormalize_predictions

    all_predictions_denorm = denorm_func(all_predictions)
    all_targets_denorm = denorm_func(all_targets)

    # Calculate 6 mandatory metrics
    metrics = evaluate_predictions(all_targets_denorm, all_predictions_denorm)

    return avg_loss, metrics, all_predictions_denorm, all_targets_denorm


def save_results(results, config, save_dir):
    """
    Save results to JSON file

    Args:
        results: Results dictionary
        config: Configuration object
        save_dir: Directory to save results
    """
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    results_path = save_dir / 'timesnet_baseline_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[save_results] Saved results to {results_path}")


def train_timesnet_baseline():
    """Main training function for TimesNet baseline"""

    print("="*80)
    print("TIMESNET BASELINE TRAINING - VOLATILITY PREDICTION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create configuration
    config = TimesNetConfig()
    print(f"\nConfiguration:")
    print(f"  - Sequence length: {config.seq_len}")
    print(f"  - Forecast horizon: {config.forecast_horizon}")
    print(f"  - Hidden dim: {config.d_model}")
    print(f"  - Num kernels: {config.num_kernels}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Max epochs: {config.num_epochs}")
    print(f"  - Patience: {config.patience}")

    # Set device
    device = torch.device(config.device)
    print(f"\nDevice: {device}")

    # Create dataloaders
    print(f"\nCreating dataloaders...")
    data_dir = 'data/processed'
    train_loader, val_loader, test_loader, datasets = create_timesnet_dataloaders(
        data_dir=data_dir,
        seq_length=config.seq_len,
        forecast_horizon=config.forecast_horizon,
        batch_size=config.batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        num_workers=config.num_workers,
        normalize=True
    )

    train_dataset, val_dataset, test_dataset = datasets

    # Create model
    print(f"\nCreating TimesNet model...")
    model = create_timesnet_model(config)
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {n_params:,}")
    print(f"  - Trainable parameters: {n_trainable:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta, min_epochs=config.min_epochs)

    # Training history
    train_losses = []
    val_losses = []

    # Create results directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    results_dir = Path(f'results/timesnet_baseline_{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nResults directory: {results_dir}")

    # Training loop
    print(f"\nStarting training loop...", flush=True)
    print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val Dir Acc':>12} | {'Val RMSE':>12}", flush=True)
    print("-" * 70, flush=True)

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(config.num_epochs):
        print(f"\n[Epoch {epoch+1}/{config.num_epochs}] Training...", flush=True)

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        train_losses.append(train_loss)
        print(f"  Train Loss: {train_loss:.6f}", flush=True)

        # Validate
        print(f"  Validating...", flush=True)
        val_loss, val_metrics, _, _ = validate(model, val_loader, criterion, device, val_dataset)
        val_losses.append(val_loss)
        print(f"  Val Loss: {val_loss:.6f}", flush=True)

        # Print progress summary (flush=True for real-time output)
        print(f"{epoch+1:>5} | {train_loss:>12.6f} | {val_loss:>12.6f} | {val_metrics['directional_accuracy']:>11.2f}% | {val_metrics['rmse']:>12.6f}", flush=True)
        import sys
        sys.stdout.flush()  # Force flush output buffer

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), results_dir / 'best_timesnet_model.pth')

        # Plot learning curves every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_path = results_dir / f'learning_curves_epoch_{epoch+1}.png'
            plot_learning_curves(train_losses, val_losses, plot_path)

        # Early stopping check
        if early_stopping(val_loss, epoch):
            print(f"\n[Early stopping] Triggered at epoch {epoch+1}", flush=True)
            print(f"  - Best epoch: {best_epoch}", flush=True)
            print(f"  - Best val loss: {best_val_loss:.6f}", flush=True)
            break

    # Training complete
    print(f"\nTraining completed at epoch {epoch+1}")
    print(f"  - Best epoch: {best_epoch}")
    print(f"  - Best val loss: {best_val_loss:.6f}")

    # Plot final learning curves
    plot_path = results_dir / 'learning_curves_final.png'
    plot_learning_curves(train_losses, val_losses, plot_path)

    # Load best model for test evaluation
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(results_dir / 'best_timesnet_model.pth'))

    # Test evaluation
    print(f"\nTest set evaluation:")
    test_loss, test_metrics, test_predictions, test_targets = validate(
        model, test_loader, criterion, device, test_dataset
    )

    print(f"\nTest Results:")
    print(f"  - MSE: {test_metrics['mse']:.6f}")
    print(f"  - RMSE: {test_metrics['rmse']:.6f}")
    print(f"  - MAE: {test_metrics['mae']:.6f}")
    print(f"  - R²: {test_metrics['r2']:.6f}")
    print(f"  - QLIKE: {test_metrics['qlike']:.6f}")
    print(f"  - Dir Acc: {test_metrics['directional_accuracy']:.2f}%")

    # Calculate validation-test differences
    val_loss_final, val_metrics_final, _, _ = validate(model, val_loader, criterion, device, val_dataset)

    val_test_diff = {
        'mse_diff': float(test_metrics['mse'] - val_metrics_final['mse']),
        'rmse_diff': float(test_metrics['rmse'] - val_metrics_final['rmse']),
        'mae_diff': float(test_metrics['mae'] - val_metrics_final['mae']),
        'r2_diff': float(test_metrics['r2'] - val_metrics_final['r2']),
        'qlike_diff': float(test_metrics['qlike'] - val_metrics_final['qlike']),
        'dir_acc_diff': float(test_metrics['directional_accuracy'] - val_metrics_final['directional_accuracy'])
    }

    # Save results
    results = {
        'model': 'TimesNet Baseline',
        'timestamp': timestamp,
        'config': config.to_dict(),
        'training_summary': {
            'num_epochs_trained': epoch + 1,
            'best_epoch': best_epoch,
            'best_val_loss': float(best_val_loss)
        },
        'validation_metrics': {
            'mse': float(val_metrics_final['mse']),
            'rmse': float(val_metrics_final['rmse']),
            'mae': float(val_metrics_final['mae']),
            'r2': float(val_metrics_final['r2']),
            'qlike': float(val_metrics_final['qlike']),
            'directional_accuracy': float(val_metrics_final['directional_accuracy'])
        },
        'test_metrics': {
            'mse': float(test_metrics['mse']),
            'rmse': float(test_metrics['rmse']),
            'mae': float(test_metrics['mae']),
            'r2': float(test_metrics['r2']),
            'qlike': float(test_metrics['qlike']),
            'directional_accuracy': float(test_metrics['directional_accuracy'])
        },
        'val_test_diff': val_test_diff
    }

    save_results(results, config, results_dir)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_dir}")
    print(f"Test Dir Acc: {test_metrics['directional_accuracy']:.2f}%")
    print(f"\nComparison with LSTM-HAR Enhanced (67.90% Dir Acc):")
    diff = test_metrics['directional_accuracy'] - 67.90
    if diff > 0:
        print(f"  [SUCCESS] TimesNet beats LSTM-HAR by {diff:.2f}%")
    elif diff >= -2:
        print(f"  [COMPETITIVE] TimesNet within 2% of LSTM-HAR ({diff:+.2f}%)")
    else:
        print(f"  [NEEDS WORK] TimesNet below LSTM-HAR by {abs(diff):.2f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == '__main__':
    results = train_timesnet_baseline()
