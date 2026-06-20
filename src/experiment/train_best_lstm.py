"""
Train LSTM with Best Hyperparameters from Optuna

After optimization completes, use this script to train final model
with the best hyperparameters found.

Usage:
    python train_best_lstm.py --params results/lstm_optimization_*/best_params.json

Author: Stock Volatility Prediction Team
Date: 2026-06-18
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.lstm_baseline.dataset import PooledVolatilityDataset
from src.common.evaluation import evaluate_predictions


class OptimizableLSTM(nn.Module):
    """LSTM with tunable hyperparameters."""

    def __init__(self, hidden_size=128, dropout=0.0):
        super(OptimizableLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        # Single LSTM layer
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass."""
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        output = self.relu(output)
        return output


def train_with_best_params(data_dir, params, output_dir=None):
    """
    Train LSTM with best hyperparameters.

    Args:
        data_dir: Directory with processed data
        params: Dict of best hyperparameters
        output_dir: Output directory
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = f'results/lstm_optimized_{timestamp}'

    print("=" * 80)
    print("TRAINING LSTM WITH BEST HYPERPARAMETERS")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Display parameters
    print("\nBest Hyperparameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Extract parameters
    hidden_size = params['hidden_size']
    dropout = params['dropout']
    lr = params['learning_rate']
    batch_size = params['batch_size']
    weight_decay = params['weight_decay']
    seq_length = params['seq_length']
    num_epochs = params.get('num_epochs', 50)  # Train longer with best params

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_gpu = device.type == 'cuda'
    num_workers = 2 if use_gpu else 0

    print(f"\nDevice: {device}")
    print(f"Using GPU: {use_gpu}")

    # Create dataset
    print(f"\n1. Creating dataset (seq_length={seq_length})...")
    dataset = PooledVolatilityDataset(data_dir, seq_length=seq_length, forecast_horizon=5)

    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_gpu
    )

    # Initialize model
    print("\n2. Initializing LSTM...")
    model = OptimizableLSTM(hidden_size=hidden_size, dropout=dropout)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print(f"\n3. Training for {num_epochs} epochs...")
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []

    import time
    epoch_times = []

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

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                      os.path.join(output_dir, 'best_lstm.pth'))

        # Print progress
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_time = np.mean(epoch_times[-5:]) if len(epoch_times) >= 5 else epoch_time
            eta = avg_time * (num_epochs - epoch - 1)
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f} | "
                  f"Time: {epoch_time:.1f}s, ETA: {eta/60:.1f}min")

    print(f"\nTraining completed: {num_epochs} epochs (best epoch: {best_epoch})")

    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_lstm.pth'),
                                   map_location=device))

    # Evaluation
    print("\n4. Evaluating on test set...")
    model.eval()

    predictions_scaled = []
    actuals_scaled = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred_scaled = model(X_batch).cpu().numpy().flatten()
            actual_scaled = y_batch.numpy().flatten()

            predictions_scaled.extend(pred_scaled)
            actuals_scaled.extend(actual_scaled)

    predictions_scaled = np.array(predictions_scaled)
    actuals_scaled = np.array(actuals_scaled)

    # Inverse-transform
    predictions = dataset.target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    actuals = dataset.target_scaler.inverse_transform(actuals_scaled.reshape(-1, 1)).flatten()

    # Calculate metrics
    metrics = evaluate_predictions(actuals, predictions)

    print("\n5. Test Results:")
    print("-" * 80)
    for metric_name, value in metrics.items():
        if metric_name == 'Directional_Acc':
            print(f"{metric_name}: {value:.2f}%")
        else:
            print(f"{metric_name}: {value:.6f}")

    # Save results
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(os.path.join(output_dir, 'test_metrics.csv'), index=False)

    # Save model info
    info = {
        'model_type': 'Optimized LSTM (1-layer, raw volatility)',
        'hyperparameters': params,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'best_epoch': best_epoch,
        'training_time_minutes': sum(epoch_times) / 60
    }

    with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nModel saved to: {output_dir}/best_lstm.pth")
    print(f"Results saved to: {output_dir}/")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Total time: {sum(epoch_times)/60:.1f} minutes")
    print(f"QLIKE: {metrics['QLIKE']:.6f} (target: < 0.20)")
    print(f"Directional Accuracy: {metrics['Directional_Acc']:.2f}% (target: > 55%)")
    print("=" * 80)

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM with best hyperparameters')
    parser.add_argument('--params', type=str, required=True,
                       help='Path to best_params.json from Optuna')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: results/lstm_optimized_*)')

    args = parser.parse_args()

    # Load best parameters
    print(f"\nLoading best parameters from: {args.params}")

    if not os.path.exists(args.params):
        print(f"[ERROR] File not found: {args.params}")
        sys.exit(1)

    with open(args.params, 'r') as f:
        best_params = json.load(f)

    # Data directory
    data_dir = os.path.join(project_root, 'data/processed')

    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        sys.exit(1)

    # Train with best parameters
    model, metrics = train_with_best_params(data_dir, best_params, args.output)

    print("\n[SUCCESS] Training completed successfully!")
    print(f"\nTo compare with HAR:")
    print(f"  python compare_models.py")
