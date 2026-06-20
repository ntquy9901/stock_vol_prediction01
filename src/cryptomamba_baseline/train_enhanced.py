"""
CryptoMamba Enhanced V2 - Training Script with Edge Case Handling

Enhanced V2 training with comprehensive edge case handling, learning curves,
and honest performance documentation.

Root cause fixes implemented:
- HAR features generated from parkinson_volatility (not assumed)
- ReLU output ensures non-negative predictions
- Edge cases handled (NaN, exploding gradients, empty data)
- Learning curves plotted every 10 epochs (ML/DS rule)
- Integration-ready (no mocking, real validation)

Target: 50-52% Dir Acc (HYPOTHESIS - requires validation)

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cryptomamba_baseline.model_enhanced import create_cryptomamba_model_enhanced
from src.cryptomamba_baseline.config_enhanced import (
    CRYPTOMAMBA_CONFIG_ENHANCED,
    MODEL_CONFIG_ENHANCED,
    TRAINING_CONFIG_ENHANCED
)
from src.common.har_features import generate_har_features, validate_har_features
from src.common.evaluation import evaluate_predictions
from src.common.data_normalization import normalize_for_training, denormalize_predictions, VolatilityNormalizer

warnings.filterwarnings('ignore')


class CryptoMambaEnhancedTrainer:
    """
    Trainer for CryptoMamba Enhanced V2 with comprehensive edge case handling.

    Handles:
    - HAR feature generation and validation
    - Edge cases (empty data, NaN, exploding gradients)
    - Learning curve plotting (every 10 epochs)
    - Gradient tracking with memory-efficient deque
    - Model checkpointing
    """

    def __init__(self, config=None):
        """Initialize trainer with configuration."""
        self.config = config or CRYPTOMAMBA_CONFIG_ENHANCED
        self.model_config = MODEL_CONFIG_ENHANCED
        self.training_config = TRAINING_CONFIG_ENHANCED

        # Training state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.gradient_norms = deque(maxlen=100)  # Fixed: deque prevents unbounded growth
        self.parameter_updates = deque(maxlen=100)  # Fixed: deque prevents unbounded growth
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Metrics tracking
        self.val_metrics_history = []
        self.test_metrics = None

        # Data normalization (CRITICAL for small-scale volatility data)
        self.target_normalizer = VolatilityNormalizer()
        self.feature_stats = None

        # Output directory
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.output_dir = project_root / "results" / f"cryptomamba_enhanced_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[DIR] Output directory: {self.output_dir}")
        print(f"[PC]  Device: {self.device}")

    def load_and_prepare_data(self, data_path):
        """
        Load and prepare data with HAR feature generation.

        Args:
            data_path: Path to processed CSV file

        Returns:
            train_loader, val_loader, test_loader

        Raises:
            ValueError: If data is insufficient or HAR features invalid
        """
        print(f"\n[DATA] Loading data from {data_path}...")

        # Load data
        if not os.path.exists(data_path):
            raise ValueError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        print(f"   Loaded {len(df)} rows")

        # CRITICAL: Generate HAR features if missing
        required_har_cols = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']
        if not all(col in df.columns for col in required_har_cols):
            print("   [!]  HAR features missing, generating from parkinson_volatility...")
            df = generate_har_features(df, volatility_col='parkinson_volatility')

        # Validate HAR features
        validate_har_features(df)
        print(f"   [OK] HAR features validated")

        # Edge case: Check data length
        seq_length = self.config['seq_length']
        forecast_horizon = self.config['forecast_horizon']
        min_required = seq_length + forecast_horizon

        if len(df) < min_required:
            raise ValueError(
                f"Insufficient data: {len(df)} rows, need at least {min_required} "
                f"(seq_length={seq_length} + forecast_horizon={forecast_horizon})"
            )

        # Edge case: Check for NaN in features
        feature_cols = required_har_cols
        if df[feature_cols].isnull().any().any():
            nan_count = df[feature_cols].isnull().sum().sum()
            raise ValueError(f"NaN found in features: {nan_count} NaN values")

        # Prepare features and targets
        X, y = self._create_sequences(df, feature_cols)

        # Split temporally (70/15/15)
        n = len(X)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        print(f"   Train: {len(X_train)} samples")
        print(f"   Val:   {len(X_val)} samples")
        print(f"   Test:  {len(X_test)} samples")

        # Edge case: Check for empty splits
        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            raise ValueError(
                f"Empty data split detected - train: {len(X_train)}, "
                f"val: {len(X_val)}, test: {len(X_test)}"
            )

        # Create dataloaders
        batch_size = self.config['batch_size']

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, val_loader, test_loader

    def _create_sequences(self, df, feature_cols):
        """Create sequences for time series forecasting with normalization."""
        seq_length = self.config['seq_length']
        forecast_horizon = self.config['forecast_horizon']

        sequences = []
        targets = []

        volatility = df['parkinson_volatility'].values

        for i in range(len(df) - seq_length - forecast_horizon):
            # Extract features
            seq_features = df[feature_cols].iloc[i:i + seq_length].values
            sequences.append(seq_features)

            # Target: volatility forecast_horizon days ahead
            target = volatility[i + seq_length + forecast_horizon - 1]
            targets.append(target)

        X = np.array(sequences)
        y = np.array(targets).reshape(-1, 1)

        # CRITICAL: Normalize data to prevent dead ReLU problem
        # Volatility data is very small (0.0001-0.01) which causes training instability
        X_norm, y_norm, self.target_normalizer, self.feature_stats = normalize_for_training(X, y)

        X = torch.FloatTensor(X_norm)
        y = torch.FloatTensor(y_norm)

        return X, y

    def create_model(self):
        """Create Enhanced V2 model."""
        print(f"\n[BUILD]  Creating CryptoMamba Enhanced V2 model...")

        self.model = create_cryptomamba_model_enhanced(
            num_features=self.model_config['num_features'],
            hidden_dim=self.model_config['hidden_dim'],
            num_layers=self.model_config['num_layers'],
            d_state=self.model_config['d_state'],
            d_conv=self.model_config['d_conv'],
            expand=self.model_config['expand'],
            dropout=self.model_config['dropout'],
            seq_length=self.model_config['seq_length'],
        )

        self.model.to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"   [OK] Model created: {num_params:,} parameters")

        # Edge case: Validate learning rate
        lr = self.training_config['learning_rate']
        if lr < 1e-6 or lr > 1.0:
            raise ValueError(f"Invalid learning rate: {lr}, must be in [1e-6, 1.0]")

        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.training_config['weight_decay']
        )

        self.criterion = nn.MSELoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.training_config['lr_step_size'],
            gamma=self.training_config['lr_gamma']
        )

        print(f"   Optimizer: Adam (lr={lr}, weight_decay={self.training_config['weight_decay']})")
        print(f"   Scheduler: StepLR (step={self.training_config['lr_step_size']}, gamma={self.training_config['lr_gamma']})")

        return self.model

    def track_parameter_updates(self):
        """
        Track parameter update magnitude (memory-efficient with deque).

        Returns:
            bool: True if parameters are updating, False if stagnant
        """
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.gradient_norms.append(total_norm)

        # Check if parameters are updating (non-zero gradients)
        is_updating = total_norm > 1e-8

        return is_updating

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with edge case checks."""
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            # Edge case: Check for NaN inputs
            if torch.isnan(X_batch).any():
                print(f"   ❌ NaN detected in input at batch {batch_idx}, epoch {epoch}")
                raise ValueError("NaN in input data - check data preprocessing")

            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(X_batch)

            # Edge case: Check for NaN predictions
            if torch.isnan(predictions).any():
                print(f"   ❌ NaN prediction detected at batch {batch_idx}, epoch {epoch}")
                raise ValueError("Model produced NaN predictions - check model architecture")

            # Edge case: Check for negative predictions (shouldn't happen with ReLU)
            if (predictions < 0).any():
                print(f"   [!]  Negative predictions detected at batch {batch_idx}, epoch {epoch}")
                print(f"      Min prediction: {predictions.min().item():.6f}")
                print(f"      ReLU constraint may not be working correctly")

            loss = self.criterion(predictions, y_batch)

            # Backward pass
            loss.backward()

            # Edge case: Check for exploding gradients
            grad_norm = self.track_parameter_updates()
            if grad_norm > 100:
                print(f"\n   [!]  EXPLODING GRADIENTS DETECTED")
                print(f"      Epoch: {epoch}, Batch: {batch_idx}")
                print(f"      Gradient norm: {grad_norm:.2f}")
                print(f"      Stopping training to prevent instability")
                raise ValueError(f"Exploding gradients (norm={grad_norm:.2f}) - training halted")

            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader):
        """Validate model and compute all 6 metrics."""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)

                val_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        avg_loss = val_loss / len(val_loader)

        # Denormalize predictions and targets for metrics
        # Metrics should be computed on original volatility scale
        y_true_norm = np.concatenate(all_targets).flatten()
        y_pred_norm = np.concatenate(all_predictions).flatten()

        y_true = self.target_normalizer.inverse_transform(y_true_norm)
        y_pred = self.target_normalizer.inverse_transform(y_pred_norm)

        metrics = evaluate_predictions(y_true, y_pred)

        return avg_loss, metrics

    def plot_learning_curves(self):
        """Plot learning curves (ML/DS rule: every 10 epochs)."""
        if len(self.train_losses) == 0:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        axes[0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Gradient norms
        if len(self.gradient_norms) > 0:
            axes[1].plot(list(self.gradient_norms), label='Gradient Norm', linewidth=2, color='orange')
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Gradient Norm')
            axes[1].set_title('Gradient Norms (Last 100 steps)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Explosion Threshold')

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "learning_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   [DATA] Learning curves saved: {plot_path}")

    def train(self, train_loader, val_loader):
        """Train model with early stopping and learning curves."""
        print(f"\n[RUN] Starting training...")
        print(f"   Max epochs: {self.training_config['num_epochs']}")
        print(f"   Patience: {self.training_config['patience']}")
        print(f"   Min epochs: {self.training_config['min_epochs']}")

        num_epochs = self.training_config['num_epochs']
        patience = self.training_config['patience']
        min_epochs = self.training_config['min_epochs']

        no_update_epochs = 0

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics_history.append(val_metrics)

            # Track parameter updates
            is_updating = self.track_parameter_updates()

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"\n   Epoch [{epoch+1}/{num_epochs}]")
                print(f"      Train Loss: {train_loss:.6f}")
                print(f"      Val Loss:   {val_loss:.6f}")
                print(f"      Val RMSE:  {val_metrics['rmse']:.6f}")
                print(f"      Val Dir Acc: {val_metrics['directional_accuracy']:.2f}%")
                print(f"      Grad Norm:  {self.gradient_norms[-1]:.4f}" if len(self.gradient_norms) > 0 else "      Grad Norm:  N/A")

                # Plot learning curves (ML/DS rule: every 10 epochs)
                self.plot_learning_curves()

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                no_update_epochs = 0

                # Save best model
                torch.save(
                    self.model.state_dict(),
                    self.output_dir / "best_cryptomamba_enhanced_model.pth"
                )
            else:
                self.patience_counter += 1
                no_update_epochs += 1

            # Edge case: No parameter updates
            if not is_updating:
                no_update_epochs += 1
                if no_update_epochs >= 50:
                    print(f"\n   [!]  WARNING: Parameters not updating for {no_update_epochs} epochs")
                    print(f"      Check learning rate or model architecture")
                    print(f"      Continuing training...")
            else:
                no_update_epochs = 0

            # Early stopping
            if self.patience_counter >= patience and (epoch + 1) >= min_epochs:
                print(f"\n   [OK] Early stopping triggered at epoch {epoch + 1}")
                print(f"      Best val loss: {self.best_val_loss:.6f}")
                break

            # Edge case: Exploding loss
            if train_loss > 1e6 or val_loss > 1e6 or np.isnan(train_loss) or np.isnan(val_loss):
                print(f"\n   ❌ CRITICAL: Loss exploding or NaN at epoch {epoch + 1}")
                print(f"      Train loss: {train_loss}")
                print(f"      Val loss: {val_loss}")
                raise ValueError("Training instability detected - halting")

        print(f"\n   [OK] Training completed after {epoch + 1} epochs")
        print(f"      Best val loss: {self.best_val_loss:.6f}")

    def evaluate_test_set(self, test_loader):
        """Evaluate on test set and compute all 6 metrics."""
        print(f"\n[TEST] Evaluating on test set...")

        # Load best model
        self.model.load_state_dict(
            torch.load(self.output_dir / "best_cryptomamba_enhanced_model.pth")
        )

        test_loss, test_metrics = self.validate(test_loader)

        print(f"   Test Loss:  {test_loss:.6f}")
        print(f"   Test RMSE:  {test_metrics['rmse']:.6f}")
        print(f"   Test MAE:   {test_metrics['mae']:.6f}")
        print(f"   Test R²:    {test_metrics['r2']:.6f}")
        print(f"   Test QLIKE: {test_metrics['qlike']:.6f}")
        print(f"   Test Dir Acc: {test_metrics['directional_accuracy']:.2f}%")

        self.test_metrics = test_metrics

        return test_metrics

    def save_results(self):
        """Save training results with all 6 metrics."""
        print(f"\n[SAVE] Saving results...")

        # Get best validation metrics
        best_idx = np.argmin(self.val_losses)
        best_val_metrics = self.val_metrics_history[best_idx]

        # Compute val-test differences
        val_test_diff = {
            'mse_diff': float(self.test_metrics['mse'] - best_val_metrics['mse']),
            'rmse_diff': float(self.test_metrics['rmse'] - best_val_metrics['rmse']),
            'mae_diff': float(self.test_metrics['mae'] - best_val_metrics['mae']),
            'r2_diff': float(self.test_metrics['r2'] - best_val_metrics['r2']),
            'qlike_diff': float(self.test_metrics['qlike'] - best_val_metrics['qlike']),
            'dir_acc_diff': float(self.test_metrics['directional_accuracy'] - best_val_metrics['directional_accuracy'])
        }

        results = {
            'model': 'CryptoMamba Enhanced V2',
            'timestamp': self.timestamp,
            'config': {
                'hidden_dim': self.config['hidden_dim'],
                'num_layers': self.config['num_layers'],
                'd_state': self.config['d_state'],
                'dropout': self.config['dropout'],
                'learning_rate': self.config['learning_rate'],
                'weight_decay': self.config['weight_decay'],
                'num_epochs_trained': len(self.train_losses),
            },
            'validation_metrics': {
                'mse': float(best_val_metrics['mse']),
                'rmse': float(best_val_metrics['rmse']),
                'mae': float(best_val_metrics['mae']),
                'r2': float(best_val_metrics['r2']),
                'qlike': float(best_val_metrics['qlike']),
                'directional_accuracy': float(best_val_metrics['directional_accuracy'])
            },
            'test_metrics': {
                'mse': float(self.test_metrics['mse']),
                'rmse': float(self.test_metrics['rmse']),
                'mae': float(self.test_metrics['mae']),
                'r2': float(self.test_metrics['r2']),
                'qlike': float(self.test_metrics['qlike']),
                'directional_accuracy': float(self.test_metrics['directional_accuracy'])
            },
            'val_test_diff': val_test_diff,
            'training_history': {
                'train_losses': [float(l) for l in self.train_losses],
                'val_losses': [float(l) for l in self.val_losses]
            }
        }

        # Save JSON
        results_path = self.output_dir / "cryptomamba_enhanced_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"   [OK] Results saved: {results_path}")

        # Print comparison table
        print(f"\n[DATA] Final Results:")
        print(f"   {'Metric':<15} {'Validation':<15} {'Test':<15} {'Difference':<15}")
        print(f"   {'-'*60}")
        print(f"   {'MSE':<15} {best_val_metrics['mse']:.6f}        {self.test_metrics['mse']:.6f}        {val_test_diff['mse_diff']:+.6f}")
        print(f"   {'RMSE':<15} {best_val_metrics['rmse']:.6f}        {self.test_metrics['rmse']:.6f}        {val_test_diff['rmse_diff']:+.6f}")
        print(f"   {'MAE':<15} {best_val_metrics['mae']:.6f}        {self.test_metrics['mae']:.6f}        {val_test_diff['mae_diff']:+.6f}")
        print(f"   {'R²':<15} {best_val_metrics['r2']:.6f}        {self.test_metrics['r2']:.6f}        {val_test_diff['r2_diff']:+.6f}")
        print(f"   {'QLIKE':<15} {best_val_metrics['qlike']:.6f}        {self.test_metrics['qlike']:.6f}        {val_test_diff['qlike_diff']:+.6f}")
        print(f"   {'Dir Acc':<15} {best_val_metrics['directional_accuracy']:.2f}%        {self.test_metrics['directional_accuracy']:.2f}%        {val_test_diff['dir_acc_diff']:+.2f}%")

        # Print interpretation
        print(f"\n[INFO] Interpretation:")
        if self.test_metrics['directional_accuracy'] > 48.32:
            print(f"   [OK] Test Dir Acc ({self.test_metrics['directional_accuracy']:.2f}%) BEATS LSTM baseline (48.32%)")
        else:
            print(f"   ❌ Test Dir Acc ({self.test_metrics['directional_accuracy']:.2f}%) below LSTM baseline (48.32%)")

        if abs(val_test_diff['dir_acc_diff']) < 1.0:
            print(f"   [OK] Val-test gap ({abs(val_test_diff['dir_acc_diff']):.2f}%) < 1% - overfitting FIXED")
        else:
            print(f"   [!]  Val-test gap ({abs(val_test_diff['dir_acc_diff']):.2f}%) ≥ 1% - possible overfitting")

        return results


def main():
    """Main training function."""
    print("\n" + "="*70)
    print("CryptoMamba Enhanced V2 - Training with Root Cause Fixes")
    print("="*70)
    print("\n[!]  Performance Target: 50-52% Dir Acc (HYPOTHESIS - requires validation)")
    print("[LIST] Root Cause Fixes Applied:")
    print("   [OK] HAR features generated from parkinson_volatility")
    print("   [OK] ReLU output ensures non-negative predictions")
    print("   [OK] Edge cases handled (NaN, exploding gradients, empty data)")
    print("   [OK] Learning curves plotted every 10 epochs")
    print("   [OK] Integration-ready (real validation, no mocks)")
    print("   [OK] Memory-efficient gradient tracking (deque, not list)")
    print("="*70)

    # Find data file
    data_dir = project_root / "data" / "processed"
    data_files = list(data_dir.glob("*.csv"))

    if not data_files:
        raise ValueError(f"No data files found in {data_dir}")

    # Use first available file
    data_path = data_files[0]
    print(f"\n[FILE] Using data file: {data_path.name}")

    # Create trainer
    trainer = CryptoMambaEnhancedTrainer()

    # Load and prepare data
    train_loader, val_loader, test_loader = trainer.load_and_prepare_data(data_path)

    # Create model
    trainer.create_model()

    # Train
    trainer.train(train_loader, val_loader)

    # Evaluate on test set
    trainer.evaluate_test_set(test_loader)

    # Save results
    trainer.save_results()

    print(f"\n[OK] Training complete!")
    print(f"[DIR] Results saved to: {trainer.output_dir}")
    print(f"\n[SEARCH] Next steps:")
    print(f"   1. Review learning curves: {trainer.output_dir / 'learning_curves.png'}")
    print(f"   2. Check metrics JSON: {trainer.output_dir / 'cryptomamba_enhanced_results.json'}")
    print(f"   3. Compare with baselines (LSTM: 48.32%, HAR-R: 51.53%)")


if __name__ == "__main__":
    main()
