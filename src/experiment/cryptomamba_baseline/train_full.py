"""
CryptoMamba Training Script (Full) - Full Architecture with mamba_ssm

Based on original CryptoMamba with proper hyperparameters and full selective scan:
- Learning rate: 0.01 (original)
- Max epochs: 1000 (original)
- Weight decay: 0.0005 (original)
- Hidden dim: 14 (original)
- Full selective state space via mamba_ssm
- Hierarchical structure: hidden_dims=[14, 1]
- ~136K parameters (vs 2,787 in V2)

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

from src.cryptomamba_baseline.model_full import create_cryptomamba_model_full, MAMBA_AVAILABLE
from src.cryptomamba_baseline.dataset import CryptoMambaDataset
from src.common.temporal_split import TemporalSplitter
from src.common.evaluation import evaluate_predictions
from src.cryptomamba_baseline.config_full import (
    CRYPTOMAMBA_CONFIG_FULL,
    TRAINING_CONFIG_FULL,
    MODEL_CONFIG_FULL
)


def train_cryptomamba_full(data_dir: str, output_dir: str = None, hierarchical: bool = True):
    """
    Train CryptoMamba Full with original hyperparameters and 3-way temporal split.

    Args:
        data_dir: Directory containing processed CSV files
        output_dir: Output directory (default: results/cryptomamba_full_YYYY-MM-DD_HHMMSS)
        hierarchical: Use hierarchical architecture (default: True)
    """
    if not MAMBA_AVAILABLE:
        print("=" * 80)
        print("❌ ERROR: mamba_ssm package not available")
        print("=" * 80)
        print("\nPlease install mamba_ssm:")
        print("  pip install mamba_ssm")
        print("\nRequirements:")
        print("  - CUDA toolkit 11.8 or higher")
        print("  - Compatible GPU (NVIDIA)")
        print("  - C++ compiler")
        print("\nInstallation guide:")
        print("  https://github.com/state-spaces/mamba#installation")
        print("=" * 80)
        sys.exit(1)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = f'results/cryptomamba_full_{timestamp}'

    print("=" * 80)
    print("CRYPTOMAMBA TRAINING FULL - FULL ARCHITECTURE WITH MAMBA_SSM (70/15/15)")
    print("=" * 80)
    print("\nPhase 1 Proof-of-Concept: Test if CryptoMamba beats LSTM")
    print(f"LSTM Baseline: 48.01-48.32% Dir Acc")
    print(f"HAR-R Baseline: 51.53% Dir Acc")
    print(f"CryptoMamba Full Target: Beat LSTM (>48%), approach HAR-R (>51%)")
    print("\nFull Architecture Features:")
    print("  - Full selective state space (via mamba_ssm)")
    print("  - Hierarchical structure: hidden_dims=[14, 1]")
    print("  - ~136K parameters (vs 2,787 in V2)")
    print("  - CUDA-accelerated selective scan")
    print(f"  - Architecture: {'Hierarchical [14, 1]' if hierarchical else 'Simplified [14]'}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Print configuration
    print("\n[CONFIGURATION - ORIGINAL HYPERPARAMETERS]")
    print(f"  Hidden Dim: {CRYPTOMAMBA_CONFIG_FULL['hidden_dim']} (original)")
    print(f"  Num Layers: {CRYPTOMAMBA_CONFIG_FULL['num_layers']} (hierarchical)")
    print(f"  Dropout: {CRYPTOMAMBA_CONFIG_FULL['dropout']} (original)")
    print(f"  Learning Rate: {CRYPTOMAMBA_CONFIG_FULL['learning_rate']} (original!)")
    print(f"  Weight Decay: {CRYPTOMAMBA_CONFIG_FULL['weight_decay']} (original!)")
    print(f"  Max Epochs: {CRYPTOMAMBA_CONFIG_FULL['num_epochs']} (original!)")
    print(f"  LR Scheduling: StepLR (step={CRYPTOMAMBA_CONFIG_FULL['lr_step_size']}, gamma={CRYPTOMAMBA_CONFIG_FULL['lr_gamma']})")
    print(f"  Sequence Length: {CRYPTOMAMBA_CONFIG_FULL['seq_length']}")
    print(f"  Use Volume: {CRYPTOMAMBA_CONFIG_FULL['use_volume']}")
    print(f"  Features: {MODEL_CONFIG_FULL['num_features']}")
    print(f"  Train/Val/Test Split: 70% / 15% / 15%")
    print(f"  Selective Scan: Full (mamba_ssm)")
    print(f"  Expected Parameters: ~136K")

    # Create dataset
    print("\n1. Creating CryptoMamba dataset...")
    dataset = CryptoMambaDataset(
        data_dir=data_dir,
        seq_length=CRYPTOMAMBA_CONFIG_FULL['seq_length'],
        forecast_horizon=CRYPTOMAMBA_CONFIG_FULL['forecast_horizon'],
        use_volume=CRYPTOMAMBA_CONFIG_FULL['use_volume'],
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

    if not use_gpu:
        print("\n⚠️  WARNING: No GPU detected. Training will be VERY slow.")
        print("   mamba_ssm requires GPU for optimal performance.")
        print("   Consider using a machine with CUDA-capable GPU.")

    train_loader, val_loader, test_loader = splitter.create_dataloaders(
        dataset,
        batch_size=CRYPTOMAMBA_CONFIG_FULL['batch_size'],
        num_workers=num_workers,
        pin_memory=use_gpu
    )

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Initialize CryptoMamba model Full
    print("\n4. Initializing CryptoMamba model Full (full architecture with mamba_ssm)...")

    model = create_cryptomamba_model_full(
        **MODEL_CONFIG_FULL,
        hierarchical=hierarchical
    )

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    if num_params < 50000:
        print(f"⚠️  WARNING: Low parameter count ({num_params:,})")
        print("   Expected ~136K for full CryptoMamba")
        print("   This might be simplified architecture")

    print(f"Architecture: {'Hierarchical [14, 1]' if hierarchical else 'Simplified [14]'}")
    print(f"Selective Scan: Full (via mamba_ssm)")

    # Training configuration (original hyperparameters)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG_FULL['learning_rate'],
        weight_decay=TRAINING_CONFIG_FULL['weight_decay']
    )

    # Learning rate scheduler (original: StepLR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=TRAINING_CONFIG_FULL['lr_step_size'],
        gamma=TRAINING_CONFIG_FULL['lr_gamma']
    )

    # Training loop with validation
    print("\n5. Training with validation (original hyperparameters, full selective scan)...")
    num_epochs = TRAINING_CONFIG_FULL['num_epochs']
    patience = TRAINING_CONFIG_FULL['patience']
    min_epochs = TRAINING_CONFIG_FULL['min_epochs']
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    print(f"Starting training: {num_epochs} epochs, min_epochs: {min_epochs}")
    print(f"Learning rate scheduling: StepLR (step={TRAINING_CONFIG_FULL['lr_step_size']}, gamma={TRAINING_CONFIG_FULL['lr_gamma']})")
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

        # Learning rate scheduling (StepLR - after each epoch)
        scheduler.step()

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
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0],
            }, os.path.join(output_dir, 'best_cryptomamba_full_model.pth'))
        else:
            epochs_without_improvement += 1

        # Print progress
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} "
              f"(Best Val: {best_val_loss:.6f} @ epoch {best_epoch+1}) - "
              f"LR: {current_lr:.6f}")

        # Early stopping (only after min_epochs)
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print(f"\nTraining completed: {epoch+1} epochs (best epoch: {best_epoch+1})")

    # Evaluate best model on validation set
    print("\n6. Evaluating best model on validation set...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_cryptomamba_full_model.pth'))
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
    print(f"CryptoMamba V2 (Simplified): 47.78% Dir Acc")
    print(f"CryptoMamba Full: {test_metrics['directional_accuracy']:.2f}% Dir Acc")

    if test_metrics['directional_accuracy'] > 48.32:
        print(f"✅ CryptoMamba Full BEATS LSTM baseline!")
    else:
        print(f"❌ CryptoMamba Full does NOT beat LSTM baseline")

    if test_metrics['directional_accuracy'] > 51.53:
        print(f"🎉 CryptoMamba Full BEATS HAR-R baseline!")
    else:
        print(f"⚠️  CryptoMamba Full approaches HAR-R baseline")

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    results = {
        'model': 'CryptoMamba Full (Full Architecture with mamba_ssm)',
        'timestamp': timestamp,
        'phase': 'Phase 1 Proof-of-Concept',
        'features': f"HAR ({MODEL_CONFIG_FULL['num_features']} features)",
        'split_method': 'Temporal (70/15/15)',
        'architecture': 'Hierarchical [14, 1]' if hierarchical else 'Simplified [14]',
        'selective_scan': 'Full (via mamba_ssm)',
        'configuration': {
            'hidden_dim': int(CRYPTOMAMBA_CONFIG_FULL['hidden_dim']),
            'num_layers': int(CRYPTOMAMBA_CONFIG_FULL['num_layers']),
            'dropout': float(CRYPTOMAMBA_CONFIG_FULL['dropout']),
            'learning_rate': float(CRYPTOMAMBA_CONFIG_FULL['learning_rate']),
            'weight_decay': float(CRYPTOMAMBA_CONFIG_FULL['weight_decay']),
            'max_epochs': int(CRYPTOMAMBA_CONFIG_FULL['num_epochs']),
            'lr_step_size': int(CRYPTOMAMBA_CONFIG_FULL['lr_step_size']),
            'lr_gamma': float(CRYPTOMAMBA_CONFIG_FULL['lr_gamma']),
            'seq_length': int(CRYPTOMAMBA_CONFIG_FULL['seq_length']),
            'use_volume': bool(CRYPTOMAMBA_CONFIG_FULL['use_volume']),
        },
        'model_info': {
            'total_parameters': num_params,
            'selective_scan': 'Full (mamba_ssm)',
            'architecture': 'Hierarchical' if hierarchical else 'Simplified',
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
            'cryptomamba_v2_dir_acc': '47.78%',
            'cryptomamba_full_dir_acc': f"{test_metrics['directional_accuracy']:.2f}%",
            'beats_lstm': bool(float(test_metrics['directional_accuracy']) > 48.32),
            'beats_harr': bool(float(test_metrics['directional_accuracy']) > 51.53),
        }
    }

    with open(os.path.join(output_dir, 'cryptomamba_full_results.json'), 'w') as f:
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

    # Check if mamba_ssm is available
    if not MAMBA_AVAILABLE:
        print("\n" + "="*80)
        print("❌ ERROR: mamba_ssm package not available")
        print("="*80)
        print("\nPlease install mamba_ssm before training:")
        print("  pip install mamba_ssm")
        print("\nRequirements:")
        print("  - CUDA toolkit 11.8 or higher")
        print("  - Compatible NVIDIA GPU")
        print("  - C++ compiler (gcc/g++ or MSVC)")
        print("\nInstallation guide:")
        print("  https://github.com/state-spaces/mamba#installation")
        print("="*80)
        sys.exit(1)

    # Train CryptoMamba Full
    print("\n" + "="*80)
    print("CRYPTOMAMBA FULL - FULL ARCHITECTURE WITH MAMBA_SSM")
    print("="*80)
    print("\nObjective: Test if CryptoMamba beats LSTM baseline (full selective scan)")
    print("LSTM Performance: 48.01-48.32% Dir Acc (all variants)")
    print("HAR-R Performance: 51.53% Dir Acc")
    print("CryptoMamba V2: 47.78% Dir Acc (simplified, 2,787 params)")
    print("CryptoMamba Full Target: >48% (beat LSTM), approach 51% (HAR-R)")
    print("\nFull Architecture Features:")
    print("  - Full selective state space (via mamba_ssm)")
    print("  - Hierarchical structure: hidden_dims=[14, 1]")
    print("  - ~136K parameters (vs 2,787 in V2)")
    print("  - CUDA-accelerated selective scan operations")
    print("="*80 + "\n")

    model, val_metrics, test_metrics = train_cryptomamba_full(data_dir, hierarchical=True)

    print("\n" + "="*80)
    print("PHASE 1 COMPLETE - FULL")
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
        print("Recommendation: Check implementation, verify CUDA setup")

    print("="*80)
