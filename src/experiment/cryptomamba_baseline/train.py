"""
CryptoMamba Training Script with 3-Way Temporal Split

Phase 1 Proof-of-Concept: Test if CryptoMamba beats LSTM (48% Dir Acc)

Uses:
- 70% train / 15% validation / 15% test temporal split
- All 6 mandatory metrics (MSE, RMSE, MAE, R², QLIKE, Dir Acc)
- Same training setup as LSTM models for fair comparison
- Project's existing evaluation framework

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import time
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cryptomamba_baseline.model import create_cryptomamba_model
from src.cryptomamba_baseline.dataset import CryptoMambaDataset
from src.common.temporal_split import TemporalSplitter
from src.common.evaluation import evaluate_predictions
from src.cryptomamba_baseline.config import CRYPTOMAMBA_CONFIG, TRAINING_CONFIG, MODEL_CONFIG


def train_cryptomamba_with_val(data_dir: str, output_dir: str = None):
    """
    Train CryptoMamba with 3-way temporal split (70/15/15).

    Args:
        data_dir: Directory containing processed CSV files
        output_dir: Output directory (default: results/cryptomamba_val_YYYY-MM-DD_HHMMSS)
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = f'results/cryptomamba_val_{timestamp}'

    print("=" * 80)
    print("CRYPTOMAMBA TRAINING - 3-WAY TEMPORAL SPLIT (70/15/15)")
    print("=" * 80)
    print("\nPhase 1 Proof-of-Concept: Test if CryptoMamba beats LSTM")
    print(f"LSTM Baseline: 48.01-48.32% Dir Acc")
    print(f"HAR-R Baseline: 51.53% Dir Acc")
    print(f"Target: Beat LSTM (>48%), approach HAR-R (>51%)")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Print configuration
    print("\n[CONFIGURATION]")
    print(f"  Hidden Dim: {CRYPTOMAMBA_CONFIG['hidden_dim']}")
    print(f"  Num Layers: {CRYPTOMAMBA_CONFIG['num_layers']}")
    print(f"  Dropout: {CRYPTOMAMBA_CONFIG['dropout']}")
    print(f"  Learning Rate: {CRYPTOMAMBA_CONFIG['learning_rate']}")
    print(f"  Sequence Length: {CRYPTOMAMBA_CONFIG['seq_length']}")
    print(f"  Use Volume: {CRYPTOMAMBA_CONFIG['use_volume']}")
    print(f"  Features: {MODEL_CONFIG['num_features']}")
    print(f"  Train/Val/Test Split: 70% / 15% / 15%")

    # Create dataset
    print("\n1. Creating CryptoMamba dataset...")
    dataset = CryptoMambaDataset(
        data_dir=data_dir,
        seq_length=CRYPTOMAMBA_CONFIG['seq_length'],
        forecast_horizon=CRYPTOMAMBA_CONFIG['forecast_horizon'],
        use_volume=CRYPTOMAMBA_CONFIG['use_volume'],
    )

    print(f"Total dataset size: {len(dataset)} sequences")

    # Create temporal splitter (70/15/15)
    print("\n2. Performing temporal split...")
    splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    split_info = splitter.get_info(len(dataset))

    print(f"Split plan:")
    print(f"  Train: {split_info['train_size']} sequences (70%)")
    print(f"  Val:   {split_info['val_size']} sequences (15%)")
    print(f"  Test:  {split_info['test_size']} sequences (15%)")

    # Create dataloaders
    print("\n3. Creating dataloaders...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_gpu = device.type == 'cuda'
    num_workers = 0  # Windows: use 0

    train_loader, val_loader, test_loader = splitter.create_dataloaders(
        dataset,
        batch_size=CRYPTOMAMBA_CONFIG['batch_size'],
        num_workers=num_workers,
        pin_memory=use_gpu
    )

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Initialize CryptoMamba model
    print("\n4. Initializing CryptoMamba model...")
    model = create_cryptomamba_model(**MODEL_CONFIG)

    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: CryptoMamba (simplified SSM for volatility)")

    # Training configuration
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop with validation
    print("\n5. Training with validation...")
    num_epochs = TRAINING_CONFIG['num_epochs']
    patience = TRAINING_CONFIG['patience']
    min_epochs = TRAINING_CONFIG['min_epochs']
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    print(f"Starting training: {num_epochs} epochs, min_epochs: {min_epochs}")
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

            # Gradient clipping
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

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping (only after min_epochs)
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
            }, os.path.join(output_dir, 'best_cryptomamba_model.pth'))
        else:
            epochs_without_improvement += 1

        # Print progress
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} "
              f"(Best Val: {best_val_loss:.6f} @ epoch {best_epoch+1})")

        # Early stopping (only after min_epochs)
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print(f"\nTraining completed: {epoch+1} epochs (best epoch: {best_epoch+1})")

    # Evaluate best model on validation set
    print("\n6. Evaluating best model on validation set...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_cryptomamba_model.pth'))
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

    # Compare with baselines
    print(f"\n{'='*80}")
    print("BASELINE COMPARISON")
    print(f"{'='*80}")
    print(f"LSTM Baseline: 48.01-48.32% Dir Acc")
    print(f"HAR-R Baseline: 51.53% Dir Acc")
    print(f"CryptoMamba:    {test_metrics['directional_accuracy']:.2f}% Dir Acc")

    if test_metrics['directional_accuracy'] > 48.32:
        print(f"✅ CryptoMamba BEATS LSTM baseline!")
    else:
        print(f"❌ CryptoMamba does NOT beat LSTM baseline")

    if test_metrics['directional_accuracy'] > 51.53:
        print(f"🎉 CryptoMamba BEATS HAR-R baseline!")
    else:
        print(f"⚠️  CryptoMamba approaches HAR-R baseline")

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    results = {
        'model': 'CryptoMamba with Validation (Phase 1 PoC)',
        'timestamp': timestamp,
        'phase': 'Phase 1 Proof-of-Concept',
        'features': f"HAR + volume ({MODEL_CONFIG['num_features']} features)",
        'split_method': 'Temporal (70/15/15)',
        'configuration': {
            'hidden_dim': CRYPTOMAMBA_CONFIG['hidden_dim'],
            'num_layers': CRYPTOMAMBA_CONFIG['num_layers'],
            'dropout': CRYPTOMAMBA_CONFIG['dropout'],
            'learning_rate': CRYPTOMAMBA_CONFIG['learning_rate'],
            'weight_decay': CRYPTOMAMBA_CONFIG['weight_decay'],
            'seq_length': CRYPTOMAMBA_CONFIG['seq_length'],
            'use_volume': CRYPTOMAMBA_CONFIG['use_volume'],
        },
        'best_epoch': best_epoch + 1,
        'best_val_loss': float(best_val_loss),
        'total_epochs': epoch + 1,
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
        'baseline_comparison': {
            'lstm_dir_acc': '48.01-48.32%',
            'harr_dir_acc': '51.53%',
            'cryptomamba_dir_acc': f"{test_metrics['directional_accuracy']:.2f}%",
            'beats_lstm': test_metrics['directional_accuracy'] > 48.32,
            'beats_harr': test_metrics['directional_accuracy'] > 51.53,
        }
    }

    with open(os.path.join(output_dir, 'cryptomamba_val_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*80}")

    return model, val_metrics, test_metrics


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    data_dir = 'data/processed'

    # Check if processed data exists
    if not os.path.exists(data_dir):
        print(f"[ERROR] Processed data directory not found: {data_dir}")
        print("Please run: python -m src.common.process_data")
        sys.exit(1)

    # Train CryptoMamba with validation
    print("\n" + "="*80)
    print("CRYPTOMAMBA PHASE 1 PROOF-OF-CONCEPT")
    print("="*80)
    print("\nObjective: Test if CryptoMamba beats LSTM baseline")
    print("LSTM Performance: 48.01-48.32% Dir Acc (all variants)")
    print("HAR-R Performance: 51.53% Dir Acc")
    print("CryptoMamba Target: >48% (beat LSTM), approach 51% (HAR-R)")
    print("="*80 + "\n")

    model, val_metrics, test_metrics = train_cryptomamba_with_val(data_dir)

    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print("="*80)
    print(f"Final Test Dir Acc: {test_metrics['directional_accuracy']:.2f}%")
    print(f"Target: >48% (beat LSTM)")
    print(f"Stretch: >51% (beat HAR-R)")

    if test_metrics['directional_accuracy'] > 48.32:
        print("Status: ✅ SUCCESS - Beats LSTM baseline!")
        print("Recommendation: Proceed to Phase 2 (HAR-Mamba integration)")
    elif test_metrics['directional_accuracy'] > 48.0:
        print("Status: ⚠️  PARTIAL - Approaches LSTM performance")
        print("Recommendation: Investigate hyperparameters or try Phase 2")
    else:
        print("Status: ❌ NEEDS INVESTIGATION - Below LSTM baseline")
        print("Recommendation: Check implementation, try different hyperparameters")

    print("="*80)
