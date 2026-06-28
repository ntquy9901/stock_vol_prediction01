"""
LSTM-GAT Hybrid Training Script for Multi-Stock Volatility Prediction

Implements comprehensive training with:
- Temporal data split (70/15/15)
- 6 mandatory metrics (MSE, RMSE, MAE, R², QLIKE, Dir Acc)
- Early stopping with patience=15
- Learning curve plotting
- Model checkpointing
- Attention visualization
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

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.lstm_gat_hybrid.model import create_lstm_gat_model
from src.lstm_gat_hybrid.config import LSTMGATConfig
from src.lstm_gat_hybrid.dataset import create_multi_stock_dataloaders
from src.common.evaluation import evaluate_predictions


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=15, min_delta=1e-6, min_epochs=20):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.epoch = 0

    def __call__(self, val_loss, epoch):
        if epoch < self.min_epochs:
            return False

        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


def plot_learning_curves(train_losses, val_losses, save_path):
    """Plot training and validation learning curves"""
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

    print(f"[plot_learning_curves] Saved plot to {save_path}", flush=True)


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, adj_matrix, y, _ in dataloader:
        # Move to device
        x = x.to(device)
        adj_matrix = adj_matrix.to(device)
        y = y.to(device)

        # Reshape y for loss computation: [batch, num_stocks] -> [batch * num_stocks]
        batch_size, num_stocks = y.shape
        y_flat = y.reshape(batch_size * num_stocks)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(x, adj_matrix)

        # Reshape predictions: [batch, num_stocks * output_dim] -> [batch * num_stocks, output_dim]
        predictions_flat = predictions.reshape(batch_size * num_stocks, -1)

        # Compute loss
        loss = criterion(predictions_flat, y_flat.unsqueeze(1))

        # Moderate loss scaling (10x instead of 1000x) to balance gradient flow
        loss = loss * 10.0

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Print progress every 20 batches
        if (n_batches + 1) % 20 == 0:
            print(f"    Batch {n_batches+1}/{len(dataloader)}: Loss={loss.item():.6f}", flush=True)

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    print(f"  Average Train Loss: {avg_loss:.6f}", flush=True)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x, adj_matrix, y, _ in dataloader:
            # Move to device
            x = x.to(device)
            adj_matrix = adj_matrix.to(device)
            y = y.to(device)

            # Reshape y
            batch_size, num_stocks = y.shape
            y_flat = y.reshape(batch_size * num_stocks)

            # Forward pass
            predictions = model(x, adj_matrix)
            predictions_flat = predictions.reshape(batch_size * num_stocks, -1)

            # Compute loss
            loss = criterion(predictions_flat, y_flat.unsqueeze(1))

            # Moderate loss scaling (10x instead of 1000x) to balance gradient flow
            loss = loss * 10.0

            total_loss += loss.item()
            n_batches += 1

            # Store predictions and targets
            # predictions_flat: [batch * num_stocks, 1] -> extract predictions
            all_predictions.extend(predictions_flat.squeeze(1).cpu().numpy())
            all_targets.extend(y_flat.cpu().numpy())

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

    # Convert to numpy and flatten
    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()

    # Calculate 6 mandatory metrics (using raw predictions - no denormalization for multi-stock)
    metrics = evaluate_predictions(all_targets, all_predictions)

    return avg_loss, metrics


def save_results(results, config, save_dir):
    """Save results to JSON file"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results_path = save_dir / 'lstm_gat_hybrid_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[save_results] Saved results to {results_path}", flush=True)


def train_lstm_gat_hybrid():
    """Main training function for LSTM-GAT Hybrid"""

    print("="*80)
    print("LSTM-GAT HYBRID TRAINING - MULTI-STOCK VOLATILITY PREDICTION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Create configuration
    config = LSTMGATConfig()
    print(f"\nConfiguration:")
    print(f"  - Num stocks: {config.num_stocks}")
    print(f"  - LSTM hidden dim: {config.lstm_hidden_dim}")
    print(f"  - GAT hidden dim: {config.gat_hidden_dim}")
    print(f"  - GAT num heads: {config.gat_num_heads}")
    print(f"  - Graph method: {config.graph_method}")
    print(f"  - Fusion method: {config.fusion_method}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Max epochs: {config.num_epochs}")
    print(f"  - Patience: {config.patience}")

    # Set device
    device = torch.device(config.device)
    print(f"\nDevice: {device}")

    # Create dataloaders with anti-overfitting techniques
    print(f"\nCreating multi-stock dataloaders with Phase 1 anti-overfitting techniques...")
    train_loader, val_loader, test_loader, datasets = create_multi_stock_dataloaders(
        data_dir='data/processed',
        seq_length=config.seq_length,
        forecast_horizon=config.forecast_horizon,
        graph_method=config.graph_method,
        batch_size=config.batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        num_workers=config.num_workers,
        normalize=True,
        # Phase 1: Data-Centric Anti-Overfitting Techniques
        remove_outliers=True,
        n_std=3.0,
        data_augmentation=True,
        augmentation_prob=0.3,
        augmentation_factor=0.1
    )

    # Create model
    print(f"\nCreating LSTM-GAT Hybrid model...")
    model = create_lstm_gat_model(config)
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
    results_dir = Path(f'results/lstm_gat_hybrid_{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nResults directory: {results_dir}")

    # Training loop
    print(f"\nStarting training loop...")
    print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val Dir Acc':>12} | {'Val RMSE':>12}", flush=True)
    print("-" * 70, flush=True)

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(config.num_epochs):
        print(f"\n[Epoch {epoch+1}/{config.num_epochs}] Training...", flush=True)

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        train_losses.append(train_loss)

        # Validate
        print(f"  Validating...", flush=True)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Print progress
        print(f"{epoch+1:>5} | {train_loss:>12.6f} | {val_loss:>12.6f} | {val_metrics['directional_accuracy']:>11.2f}% | {val_metrics['rmse']:>12.6f}", flush=True)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), results_dir / 'best_lstm_gat_model.pth')

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
    model.load_state_dict(torch.load(results_dir / 'best_lstm_gat_model.pth'))

    # Test evaluation
    print(f"\nTest set evaluation:")
    test_loss, test_metrics = validate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  - MSE: {test_metrics['mse']:.6f}")
    print(f"  - RMSE: {test_metrics['rmse']:.6f}")
    print(f"  - MAE: {test_metrics['mae']:.6f}")
    print(f"  - R²: {test_metrics['r2']:.6f}")
    print(f"  - QLIKE: {test_metrics['qlike']:.6f}")
    print(f"  - Dir Acc: {test_metrics['directional_accuracy']:.2f}%")

    # Save results
    results = {
        'model': 'LSTM-GAT Hybrid',
        'timestamp': timestamp,
        'config': config.to_dict(),
        'training_summary': {
            'num_epochs_trained': epoch + 1,
            'best_epoch': best_epoch,
            'best_val_loss': float(best_val_loss)
        },
        'test_metrics': {
            'mse': float(test_metrics['mse']),
            'rmse': float(test_metrics['rmse']),
            'mae': float(test_metrics['mae']),
            'r2': float(test_metrics['r2']),
            'qlike': float(test_metrics['qlike']),
            'directional_accuracy': float(test_metrics['directional_accuracy'])
        }
    }

    save_results(results, config, results_dir)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_dir}")
    print(f"Test Dir Acc: {test_metrics['directional_accuracy']:.2f}%")

    print(f"\nComparison with baselines:")
    print(f"  - LSTM-HAR Enhanced: 67.90%")
    print(f"  - LSTM-GAT Hybrid: {test_metrics['directional_accuracy']:.2f}%")

    diff = test_metrics['directional_accuracy'] - 67.90
    if diff > 0:
        print(f"  [SUCCESS] LSTM-GAT beats LSTM-HAR by {diff:.2f}%")
    else:
        print(f"  [INFO] LSTM-GAT gap to LSTM-HAR: {abs(diff):.2f}%")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == '__main__':
    results = train_lstm_gat_hybrid()
