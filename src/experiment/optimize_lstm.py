"""
LSTM Hyperparameter Optimization with Optuna

Optimize LSTM hyperparameters WITHOUT changing architecture:
- Keep: 1-layer LSTM, raw volatility input (no HAR features)
- Tune: hidden_size, learning_rate, batch_size, dropout, weight_decay, seq_length

Author: Stock Volatility Prediction Team
Date: 2026-06-18
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
from datetime import datetime
import json

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

        # Single LSTM layer (keep original architecture)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

        # Activation (ensure non-negative)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass."""
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Extract last timestep
        last_output = lstm_out[:, -1, :]

        # Apply dropout
        last_output = self.dropout(last_output)

        # Output layer
        output = self.fc(last_output)

        # Ensure non-negative
        output = self.relu(output)

        return output


def train_with_params(data_dir, params, device='cuda', trial=None):
    """
    Train LSTM with given hyperparameters.

    Args:
        data_dir: Directory with processed data
        params: Dict of hyperparameters
        device: 'cuda' or 'cpu'
        trial: Optuna trial for pruning

    Returns:
        metrics: Dict of evaluation metrics
    """
    # Extract parameters
    hidden_size = params['hidden_size']
    dropout = params['dropout']
    lr = params['learning_rate']
    batch_size = params['batch_size']
    weight_decay = params['weight_decay']
    seq_length = params['seq_length']
    num_epochs = params.get('num_epochs', 30)

    # Create dataset with tunable seq_length
    print(f"Creating dataset (seq_length={seq_length})...")
    dataset = PooledVolatilityDataset(data_dir, seq_length=seq_length, forecast_horizon=5)

    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Device setup
    use_gpu = device == 'cuda' and torch.cuda.is_available()
    num_workers = 2 if use_gpu else 0

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
    model = OptimizableLSTM(hidden_size=hidden_size, dropout=dropout)
    model = model.to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
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

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(test_loader)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Pruning based on intermediate value
        if trial is not None and epoch >= 5:
            # Report intermediate value for pruning
            trial.report(val_loss, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

    # Final evaluation on test set
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

    return metrics


def objective(trial, data_dir, device):
    """
    Optuna objective function.

    Args:
        trial: Optuna trial
        data_dir: Data directory
        device: 'cuda' or 'cpu'

    Returns:
        float: QLIKE score (to minimize)
    """
    # Suggest hyperparameters
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'weight_decay': trial.suggest_categorical('weight_decay', [0, 1e-5, 1e-4, 1e-3]),
        'seq_length': trial.suggest_categorical('seq_length', [10, 22, 30, 44]),
        'num_epochs': 30
    }

    print(f"\n[Trial {trial.number}] Testing params: {params}")

    try:
        # Train with suggested parameters
        metrics = train_with_params(data_dir, params, device, trial)

        # Return QLIKE (to minimize)
        qlike = metrics['QLIKE']

        print(f"Trial {trial.number}: QLIKE={qlike:.6f}, Dir_Acc={metrics['Directional_Acc']:.2f}%")

        return qlike

    except optuna.TrialPruned:
        raise


def main():
    """Main optimization function."""
    print("=" * 80)
    print("LSTM HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("=" * 80)

    # Setup
    data_dir = os.path.join(project_root, 'data/processed')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nDevice: {device}")
    print(f"Data directory: {data_dir}")

    # Check data
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        return

    # Create study
    print("\n1. Creating Optuna study...")
    study = optuna.create_study(
        study_name=f'lstm_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        direction='minimize',  # Minimize QLIKE
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # Run optimization
    print("\n2. Starting optimization (50 trials)...")
    print("   This may take 1-2 hours depending on GPU availability")
    print("=" * 80)

    study.optimize(
        lambda trial: objective(trial, data_dir, device),
        n_trials=50,
        timeout=7200,  # 2 hours max
        show_progress_bar=True
    )

    # Results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)

    # Best trial
    print("\n3. Best Trial:")
    print(f"   Trial ID: {study.best_trial.number}")
    print(f"   QLIKE: {study.best_value:.6f}")

    print("\n   Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"     {key}: {value}")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = f'results/lstm_optimization_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Save best params
    best_params_file = os.path.join(output_dir, 'best_params.json')
    with open(best_params_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)

    # Save all trials
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(output_dir, 'all_trials.csv'), index=False)

    print(f"\n4. Results saved to: {output_dir}/")
    print(f"   - best_params.json")
    print(f"   - all_trials.csv")

    # Display top 5 trials
    print("\n5. Top 5 Trials:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else float('inf'))[:5]

    for i, trial in enumerate(top_trials, 1):
        if trial.value:
            print(f"\n   Rank {i}:")
            print(f"     Trial ID: {trial.number}")
            print(f"     QLIKE: {trial.value:.6f}")
            print(f"     Params: {trial.params}")

    print("\n" + "=" * 80)
    print("[SUCCESS] Optimization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
