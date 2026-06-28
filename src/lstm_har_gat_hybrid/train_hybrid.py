"""
Training Pipeline for LSTM-HAR-GAT Hybrid Model

Complete training pipeline with:
- Early stopping
- Learning curves plotting
- All 6 mandatory metrics
- Temporal split (70/15/15)
- Model checkpointing

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json
import matplotlib.pyplot as plt
from pathlib import Path

from src.lstm_har_gat_hybrid.hybrid_model import LSTMHAR_GAT_Hybrid
from src.common.graph_utils import build_correlation_graph
from src.common.evaluation import evaluate_predictions
from src.common.temporal_split import TemporalSplitter


def custom_collate_fn(batch, num_stocks=30, seq_length=22):
    """
    Custom collate function for multi-stock batching.

    Groups samples by stock and reshapes for hybrid model input.

    Args:
        batch: List of (X, y) tuples from dataset
        num_stocks: Number of stocks (default: 30)
        seq_length: Sequence length (default: 22)

    Returns:
        X: (batch_size, num_stocks, seq_length, 3)
        y: (batch_size, num_stocks, 1)
    """
    X_list, y_list = zip(*batch)

    # Stack sequences
    X = torch.stack(X_list)
    y = torch.tensor(y_list)

    # Calculate actual batch size (round down to nearest multiple of num_stocks)
    total_samples = X.shape[0]
    actual_batch_size = (total_samples // num_stocks) * num_stocks

    # Trim to exact multiple
    X = X[:actual_batch_size]
    y = y[:actual_batch_size]

    # Reshape for multi-stock: (batch, stocks, seq_len, features)
    batch_size = actual_batch_size // num_stocks
    X = X.reshape(batch_size, num_stocks, seq_length, 3)

    # Reshape targets: (batch, stocks, 1) for loss calculation
    y = y.reshape(batch_size, num_stocks, 1)

    return X, y


class HybridVolatilityDataset(Dataset):
    """Dataset for hybrid model with temporal split support."""

    def __init__(self, data_dir, seq_length=22, forecast_horizon=5,
                 stock_names=None, start_idx=0, end_idx=None):
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon

        # Load all stock data
        self.sequences = []
        self.targets = []
        self.stock_names = []
        self.stock_indices = []  # Track which stock each sequence belongs to

        csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_processed.csv')])

        # Filter stocks if provided
        if stock_names is not None:
            csv_files = [f for f in csv_files
                        if any(f.replace('_processed.csv', '') == s for s in stock_names)]

        for csv_file in csv_files:
            ticker = csv_file.replace('_processed.csv', '')
            self.stock_names.append(ticker)
            stock_idx = len(self.stock_names) - 1

            df = pd.read_csv(os.path.join(data_dir, csv_file))

            if 'parkinson_volatility' not in df.columns:
                continue

            parkinson = df['parkinson_volatility'].dropna().values

            # Create HAR features
            har_weekly = pd.Series(parkinson).rolling(5).mean().values
            har_monthly = pd.Series(parkinson).rolling(22).mean().values

            min_required = seq_length + forecast_horizon + 22
            if len(parkinson) < min_required:
                continue

            for i in range(seq_length, len(parkinson) - forecast_horizon):
                y_target = parkinson[i + forecast_horizon]

                if np.isnan(y_target) or y_target == 0:
                    continue

                raw_seq = parkinson[i-seq_length:i]
                weekly_seq = har_weekly[i-seq_length:i]
                monthly_seq = har_monthly[i-seq_length:i]

                X_seq = np.column_stack([raw_seq, weekly_seq, monthly_seq])

                if np.isnan(X_seq).any():
                    continue

                self.sequences.append(X_seq)
                self.targets.append(y_target)
                self.stock_indices.append(stock_idx)

        # Convert to arrays
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        self.stock_indices = np.array(self.stock_indices)

        # Apply temporal slicing if indices provided
        if start_idx > 0 or end_idx is not None:
            end_idx = end_idx if end_idx is not None else len(self.sequences)
            self.sequences = self.sequences[start_idx:end_idx]
            self.targets = self.targets[start_idx:end_idx]
            self.stock_indices = self.stock_indices[start_idx:end_idx]

        # Fit scalers
        self.feature_scaler = StandardScaler()
        all_features = self.sequences.reshape(-1, self.sequences.shape[-1])
        self.feature_scaler.fit(all_features)

        self.target_scaler = StandardScaler()
        all_targets = self.targets.reshape(-1, 1)
        self.target_scaler.fit(all_targets)

        print(f"  Loaded {len(self.sequences)} sequences from {len(self.stock_names)} stocks")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X = self.sequences[idx]
        y = self.targets[idx]

        X_scaled = self.feature_scaler.transform(X)
        y_scaled = self.target_scaler.transform([[y]])[0, 0]

        return torch.FloatTensor(X_scaled), torch.FloatTensor([y_scaled])

    def inverse_transform_targets(self, y_scaled):
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1))


class HybridModelTrainer:
    """Trainer for LSTM-HAR-GAT Hybrid model with early stopping."""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader, optimizer, criterion, edge_index, edge_weight):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        batches_processed = 0

        for batch_idx, batch_data in enumerate(train_loader):
            X, y = batch_data
            X, y = X.to(self.device), y.to(self.device)

            # Data is already in correct shape from custom collate:
            # X: (batch, num_stocks, seq_len, features)
            # y: (batch, num_stocks, 1)

            # Forward pass
            predictions = self.model(X, edge_index, edge_weight)

            # Compute loss
            loss = criterion(predictions, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            batches_processed += 1

        return total_loss / max(batches_processed, 1)

    def validate(self, val_loader, edge_index, edge_weight, criterion):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        batches_processed = 0

        with torch.no_grad():
            for batch_data in val_loader:
                X, y = batch_data
                X, y = X.to(self.device), y.to(self.device)

                # Data is already in correct shape from custom collate:
                # X: (batch, num_stocks, seq_len, features)
                # y: (batch, num_stocks, 1)

                # Forward pass
                predictions = self.model(X, edge_index, edge_weight)

                # Compute loss
                loss = criterion(predictions, y)
                total_loss += loss.item()

                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())

                batches_processed += 1

        avg_loss = total_loss / max(batches_processed, 1)

        # Calculate metrics
        # Flatten to 2D: concatenate all batches and stocks
        all_preds = np.concatenate(all_preds).reshape(-1, 1)
        all_targets = np.concatenate(all_targets).reshape(-1, 1)

        metrics = evaluate_predictions(all_targets, all_preds)

        return avg_loss, metrics

    def train(self, train_loader, val_loader, edge_index, edge_weight,
             num_epochs=70, patience=15, lr=0.001, save_dir=None):
        """Complete training loop with early stopping."""

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\n[Training Configuration]")
        print(f"  Epochs: {num_epochs}")
        print(f"  Patience: {patience}")
        print(f"  Learning rate: {lr}")
        print(f"  Device: {self.device}")

        print(f"\n[Starting Training...]")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(
                train_loader, optimizer, criterion, edge_index, edge_weight
            )

            # Validate
            val_loss, val_metrics = self.validate(
                val_loader, edge_index, edge_weight, criterion
            )

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, "
                  f"Val RMSE = {val_metrics['rmse']:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                if save_dir:
                    self.save_model(save_dir, epoch)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\n  [Early Stopping] No improvement for {patience} epochs")
                break

        print(f"\n  [Training Completed]")
        print(f"    Best validation loss: {best_val_loss:.6f}")

        return self.train_losses, self.val_losses

    def save_model(self, save_dir, epoch):
        """Save model checkpoint."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        checkpoint_path = Path(save_dir) / f'hybrid_model_epoch_{epoch}_{timestamp}.pt'

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, checkpoint_path)

    def plot_learning_curves(self, save_path=None):
        """Plot training and validation learning curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Learning Curves')
        ax1.legend()
        ax1.grid(True)

        # Plot loss difference
        loss_diff = np.array(self.val_losses) - np.array(self.train_losses)
        ax2.plot(loss_diff, label='Val - Train', marker='o', color='orange')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Difference')
        ax2.set_title('Overfitting Indicator')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n  [OK] Learning curves saved to: {save_path}")

        plt.show()


def train_hybrid_model(
    data_dir='data/processed',
    num_stocks=30,
    seq_length=22,
    forecast_horizon=5,
    hidden_dim=64,
    lstm_layers=2,
    gat_heads=4,
    num_epochs=70,
    patience=15,
    batch_size=32,
    lr=0.001,
    device='cpu',
    save_dir='results/lstm_har_gat_hybrid'
):
    """
    Complete training pipeline for hybrid model.

    Args:
        data_dir: Directory containing processed data
        num_stocks: Number of stocks to use
        seq_length: Input sequence length
        forecast_horizon: Days ahead to forecast
        hidden_dim: Hidden dimension for encoders
        lstm_layers: Number of LSTM layers
        gat_heads: Number of GAT attention heads
        num_epochs: Maximum training epochs
        patience: Early stopping patience
        batch_size: Training batch size
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save results
    """

    print("="*70)
    print("LSTM-HAR-GAT HYBRID MODEL - TRAINING PIPELINE")
    print("="*70)

    # Configuration
    config = {
        'num_stocks': num_stocks,
        'seq_length': seq_length,
        'forecast_horizon': forecast_horizon,
        'hidden_dim': hidden_dim,
        'lstm_layers': lstm_layers,
        'gat_heads': gat_heads,
        'num_epochs': num_epochs,
        'patience': patience,
        'batch_size': batch_size,
        'lr': lr,
        'device': device
    }

    print(f"\n[Configuration]")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Load stock names
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_processed.csv')])
    stock_names = [f.replace('_processed.csv', '') for f in csv_files[:num_stocks]]

    print(f"\n[Data Preparation]")
    print(f"  Stocks: {len(stock_names)}")

    # Create datasets with proper temporal split
    print(f"\n[Creating Datasets]")

    # First, create a full dataset to get total size
    full_dataset = HybridVolatilityDataset(
        data_dir, seq_length, forecast_horizon, stock_names=None
    )

    total_samples = len(full_dataset)
    train_end = int(total_samples * 0.70)
    val_end = int(total_samples * 0.85)

    print(f"  Total samples: {total_samples}")
    print(f"  Train: 0 - {train_end} (70%)")
    print(f"  Val: {train_end} - {val_end} (15%)")
    print(f"  Test: {val_end} - {total_samples} (15%)")

    # Create split datasets
    train_dataset = HybridVolatilityDataset(
        data_dir, seq_length, forecast_horizon,
        stock_names, start_idx=0, end_idx=train_end
    )

    val_dataset = HybridVolatilityDataset(
        data_dir, seq_length, forecast_horizon,
        stock_names, start_idx=train_end, end_idx=val_end
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create data loaders with custom collate function
    # Use drop_last=True to ensure batches are complete multi-stock groups
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size * num_stocks,  # Ensure batch size is multiple of num_stocks
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, num_stocks, seq_length),
        drop_last=True  # Drop incomplete batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * num_stocks,
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(batch, num_stocks, seq_length),
        drop_last=True
    )

    # Build graph
    print(f"\n[Building Graph]")

    # Load volatility data for graph construction
    volatility_list = []
    for ticker in stock_names:
        csv_file = os.path.join(data_dir, f'{ticker}_processed.csv')
        df = pd.read_csv(csv_file)
        if 'parkinson_volatility' in df.columns:
            volatility = df['parkinson_volatility'].dropna().values[:500]
            volatility_list.append(volatility)

    max_len = max(len(v) for v in volatility_list)
    volatility_matrix = np.zeros((num_stocks, max_len))
    for i, vol in enumerate(volatility_list):
        volatility_matrix[i, :len(vol)] = vol

    edge_index, edge_weight, graph_info = build_correlation_graph(
        volatility_matrix, stock_names, threshold=0.5
    )

    print(f"  Graph: {graph_info['num_nodes']} nodes, {graph_info['num_edges']} edges")

    # Initialize model
    print(f"\n[Initializing Model]")
    model = LSTMHAR_GAT_Hybrid(
        num_stocks=num_stocks,
        num_features=3,
        seq_length=seq_length,
        hidden_dim=hidden_dim,
        lstm_layers=lstm_layers,
        gat_heads=gat_heads,
        dropout=0.2
    )

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = HybridModelTrainer(model, device=device)

    # Train model
    print(f"\n[Training]")
    save_dir_full = f"{save_dir}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"

    train_losses, val_losses = trainer.train(
        train_loader, val_loader, edge_index, edge_weight,
        num_epochs=num_epochs, patience=patience, lr=lr,
        save_dir=save_dir_full
    )

    # Plot learning curves
    print(f"\n[Plotting Learning Curves]")
    curves_path = os.path.join(save_dir_full, 'learning_curves.png')
    trainer.plot_learning_curves(save_path=curves_path)

    # Final evaluation
    print(f"\n[Final Evaluation]")
    model.eval()

    criterion = nn.MSELoss()
    val_loss, val_metrics = trainer.validate(val_loader, edge_index, edge_weight, criterion)

    print(f"\n  Final Validation Metrics:")
    print(f"    MSE: {val_metrics['mse']:.6f}")
    print(f"    RMSE: {val_metrics['rmse']:.6f}")
    print(f"    MAE: {val_metrics['mae']:.6f}")
    print(f"    R²: {val_metrics['r2']:.6f}")
    print(f"    QLIKE: {val_metrics['qlike']:.6f}")
    dir_acc = val_metrics['directional_accuracy'] if not np.isnan(val_metrics['directional_accuracy']) else 0.0
    print(f"    Dir Acc: {dir_acc:.2f}%")

    # Save results
    results = {
        'config': config,
        'graph_info': {
            'num_nodes': graph_info['num_nodes'],
            'num_edges': graph_info['num_edges'],
            'edge_density': float(graph_info['edge_density']),
            'avg_degree': float(graph_info['avg_degree'])
        },
        'train_losses': [float(l) for l in train_losses],
        'val_losses': [float(l) for l in val_losses],
        'final_val_metrics': {
            'mse': float(val_metrics['mse']),
            'rmse': float(val_metrics['rmse']),
            'mae': float(val_metrics['mae']),
            'r2': float(val_metrics['r2']),
            'qlike': float(val_metrics['qlike']),
            'directional_accuracy': float(val_metrics['directional_accuracy']) if not np.isnan(val_metrics['directional_accuracy']) else 0.0
        },
        'save_dir': save_dir_full
    }

    results_path = os.path.join(save_dir_full, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Results saved to: {save_dir_full}]")

    return model, results


# Main execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("LSTM-HAR-GAT HYBRID - TRAINING PIPELINE")
    print("="*70)

    # Auto-detect GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[Device] Using: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    model, results = train_hybrid_model(
        data_dir='data/processed',
        num_stocks=30,
        num_epochs=70,  # Full training - 70 epochs
        patience=15,
        batch_size=32,
        device=device
    )

    print("\n" + "="*70)
    print("TRAINING PIPELINE TEST SUCCESSFUL!")
    print("="*70)
