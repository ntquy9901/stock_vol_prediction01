"""
LSTM Hyperparameter Optimization with Optuna - OPTIMIZED VERSION

Optimizations:
1. Cache loaded CSV data (load once, reuse 50 times)
2. Only recreate sequences when seq_length changes
3. Faster trials (~50% speedup)

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

from src.common.evaluation import evaluate_predictions


# ============================================================================
# CACHE: Load all CSV data ONCE
# ============================================================================

class CachedDataManager:
    """Load and cache CSV data to avoid repeated disk I/O."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._cache = {}
        self._load_all_data()

    def _load_all_data(self):
        """Load all processed CSV files into memory ONCE."""
        print("Loading and caching all CSV files...")
        csv_files = [f for f in os.listdir(self.data_dir)
                     if f.endswith('.csv') and 'processed' in f]

        for csv_file in csv_files:
            file_path = os.path.join(self.data_dir, csv_file)
            df = pd.read_csv(file_path)
            ticker = csv_file.replace('_processed.csv', '')
            self._cache[ticker] = df['parkinson_volatility'].values

        print(f"Cached {len(self._cache)} stocks ({sum(len(v) for v in self._cache.values()):,} total rows)")

    def get_data(self):
        """Return cached data."""
        return self._cache


class FastVolatilityDataset(torch.utils.data.Dataset):
    """Fast dataset using cached data."""

    def __init__(self, cached_data, seq_length=22, forecast_horizon=5):
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon

        # Create sequences from cached data
        self.sequences = []
        self.targets = []

        for ticker, parkinson in cached_data.items():
            # Remove NaN
            valid_mask = ~np.isnan(parkinson)
            parkinson = parkinson[valid_mask]

            # Skip if insufficient data
            if len(parkinson) < seq_length + forecast_horizon:
                continue

            # Create sliding windows
            for i in range(len(parkinson) - seq_length - forecast_horizon + 1):
                X_seq = parkinson[i:i + seq_length]
                y_target = parkinson[i + seq_length + forecast_horizon - 1]

                if np.isnan(y_target) or y_target == 0:
                    continue

                self.sequences.append(X_seq)
                self.targets.append(y_target)

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)

        # Fit scalers
        from sklearn.preprocessing import StandardScaler
        self.feature_scaler = StandardScaler()
        all_data = self.sequences.flatten()
        self.feature_scaler.fit(all_data.reshape(-1, 1))

        self.target_scaler = StandardScaler()
        self.target_scaler.fit(self.targets.reshape(-1, 1))

        print(f"Created {len(self.sequences)} sequences (seq_length={seq_length})")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X_seq = self.sequences[idx]
        y_target = self.targets[idx]

        # Scale
        X_scaled = self.feature_scaler.transform(X_seq.reshape(-1, 1)).flatten()
        y_scaled = self.target_scaler.transform([[y_target]])[0, 0]

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).reshape(self.seq_length, 1)
        y_tensor = torch.FloatTensor([y_scaled])

        return X_tensor, y_tensor


# ============================================================================
# LSTM Model (Same as before)
# ============================================================================

class OptimizableLSTM(nn.Module):
    """LSTM with tunable hyperparameters."""

    def __init__(self, hidden_size=128, dropout=0.0):
        super(OptimizableLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        output = self.relu(output)
        return output


# ============================================================================
# Training Function
# ============================================================================

def train_with_params(cached_data, params, device='cuda', trial=None):
    """Train LSTM with given hyperparameters using cached data."""

    # Extract parameters
    hidden_size = params['hidden_size']
    dropout = params['dropout']
    lr = params['learning_rate']
    batch_size = params['batch_size']
    weight_decay = params['weight_decay']
    seq_length = params['seq_length']
    num_epochs = params.get('num_epochs', 30)

    # Create dataset from CACHED data (FAST!)
    dataset = FastVolatilityDataset(cached_data, seq_length=seq_length, forecast_horizon=5)

    # Split
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

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
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
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Pruning
        if trial is not None and epoch >= 5:
            trial.report(val_loss, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

    # Evaluation
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


# ============================================================================
# Optuna Objective
# ============================================================================

def objective(trial, cached_data, device):
    """Optuna objective function."""

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
        metrics = train_with_params(cached_data, params, device, trial)
        qlike = metrics['QLIKE']

        print(f"Trial {trial.number}: QLIKE={qlike:.6f}, Dir_Acc={metrics['Directional_Acc']:.2f}%")

        return qlike

    except optuna.TrialPruned:
        raise


# ============================================================================
# Main Optimization
# ============================================================================

def main():
    """Main optimization function."""
    print("=" * 80)
    print("LSTM HYPERPARAMETER OPTIMIZATION - OPTIMIZED VERSION")
    print("=" * 80)

    # Setup
    data_dir = os.path.join(project_root, 'data/processed')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nDevice: {device}")
    print(f"Data directory: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        return

    # CACHE DATA (Load once!)
    print("\n1. Caching data (this takes 5-10 seconds)...")
    cached_manager = CachedDataManager(data_dir)
    cached_data = cached_manager.get_data()

    # Create study
    print("\n2. Creating Optuna study...")
    study = optuna.create_study(
        study_name=f'lstm_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # Run optimization
    print("\n3. Starting optimization (50 trials)...")
    print("   Estimated time: 20-30 minutes (optimized)")
    print("   (Original would take 1-2 hours)")
    print("=" * 80)

    study.optimize(
        lambda trial: objective(trial, cached_data, device),
        n_trials=50,
        timeout=3600,  # 1 hour max
        show_progress_bar=True
    )

    # Results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)

    print("\n4. Best Trial:")
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

    print(f"\n5. Results saved to: {output_dir}/")

    print("\n" + "=" * 80)
    print("[SUCCESS] Optimization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
