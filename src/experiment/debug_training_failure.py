"""
LSTM Training Failure - Root Cause Analysis

Debug script to identify why LSTM model is not learning.
This performs comprehensive checks on data, model, and training process.

Author: Stock Volatility Prediction Team
Date: 2026-06-17
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.lstm_baseline.dataset import PooledVolatilityDataset
from src.lstm_baseline.model import SimpleVolatilityLSTM
from src.common.evaluation import qlike_loss, directional_accuracy


def check_data_quality(data_dir: str):
    """
    Check 1: Verify data quality and statistics
    """
    print("\n" + "=" * 80)
    print("CHECK 1: DATA QUALITY")
    print("=" * 80)

    dataset = PooledVolatilityDataset(data_dir, seq_length=22, forecast_horizon=5)

    # Sample data to check ranges
    print("\n[1.1] Sampling data...")
    inputs_scaled = []
    targets_scaled = []
    targets_original = []

    for idx in range(min(1000, len(dataset))):
        X_seq, y_target = dataset[idx]

        # Get original target (before scaling)
        ticker, seq_idx = dataset.metadata[idx]
        file_path = os.path.join(data_dir, f"{ticker}_processed.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'parkinson_volatility' in df.columns:
                parkinson = df['parkinson_volatility'].values
                target_idx = seq_idx + 22 + 5 - 1
                if target_idx < len(parkinson):
                    targets_original.append(parkinson[target_idx])

        inputs_scaled.append(X_seq.numpy().flatten())
        targets_scaled.append(y_target.item())

    inputs_scaled = np.array(inputs_scaled)
    targets_scaled = np.array(targets_scaled)
    targets_original = np.array(targets_original)

    print(f"\n[1.2] Input Statistics (Scaled):")
    print(f"   Mean: {np.mean(inputs_scaled):.6f}")
    print(f"   Std: {np.std(inputs_scaled):.6f}")
    print(f"   Min: {np.min(inputs_scaled):.6f}")
    print(f"   Max: {np.max(inputs_scaled):.6f}")

    print(f"\n[1.3] Target Statistics (Scaled):")
    print(f"   Mean: {np.mean(targets_scaled):.6f}")
    print(f"   Std: {np.std(targets_scaled):.6f}")
    print(f"   Min: {np.min(targets_scaled):.6f}")
    print(f"   Max: {np.max(targets_scaled):.6f}")

    print(f"\n[1.4] Target Statistics (Original Scale):")
    print(f"   Mean: {np.mean(targets_original):.6f}")
    print(f"   Std: {np.std(targets_original):.6f}")
    print(f"   Min: {np.min(targets_original):.6f}")
    print(f"   Max: {np.max(targets_original):.6f}")

    # Check for data issues
    print(f"\n[1.5] Data Quality Checks:")

    issues = []

    # Check if targets are constant
    if np.std(targets_original) < 1e-6:
        issues.append("CRITICAL: Targets are nearly constant!")
        print(f"   ❌ {issues[-1]}")
    else:
        print(f"   ✅ Targets have variance: {np.std(targets_original):.6f}")

    # Check if targets are reasonable range
    if np.max(targets_original) > 1.0 or np.min(targets_original) < 0:
        issues.append("WARNING: Parkinson vol outside [0,1] range")
        print(f"   ⚠️  {issues[-1]}")
    else:
        print(f"   ✅ Parkinson vol in reasonable range")

    # Check for NaN/Inf
    if np.isnan(targets_scaled).any() or np.isinf(targets_scaled).any():
        issues.append("CRITICAL: Scaled targets contain NaN/Inf!")
        print(f"   ❌ {issues[-1]}")
    else:
        print(f"   ✅ No NaN/Inf in scaled targets")

    return dataset, issues


def check_model_architecture():
    """
    Check 2: Verify model architecture and capacity
    """
    print("\n" + "=" * 80)
    print("CHECK 2: MODEL ARCHITECTURE")
    print("=" * 80)

    model = SimpleVolatilityLSTM(hidden_size=32)

    print(f"\n[2.1] Model Configuration:")
    print(f"   Input size: 1 (Parkinson volatility)")
    print(f"   Hidden size: 32")
    print(f"   Num layers: 1")
    print(f"   Output size: 1")

    print(f"\n[2.2] Parameter Count:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total: {total_params:,}")

    print(f"\n[2.3] Layer-by-Layer Breakdown:")
    for name, param in model.named_parameters():
        print(f"   {name}: {param.shape}")

    # Test forward pass
    print(f"\n[2.4] Forward Pass Test:")
    test_input = torch.randn(64, 22, 1)
    test_output = model(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {test_output.shape}")
    print(f"   Output sample: {test_output[:3].flatten().detach().numpy()}")

    # Check if output is reasonable
    if torch.any(test_output < 0):
        print(f"   ❌ WARNING: Model outputs negative values (ReLU should prevent this)")
    else:
        print(f"   ✅ Model outputs are non-negative")

    issues = []
    if total_params < 10000:
        issues.append("Model capacity may be too low (4.5K params)")

    return model, issues


def check_training_setup():
    """
    Check 3: Verify training configuration
    """
    print("\n" + "=" * 80)
    print("CHECK 3: TRAINING CONFIGURATION")
    print("=" * 80)

    print(f"\n[3.1] Current Configuration:")
    print(f"   Loss: MSE")
    print(f"   Optimizer: Adam")
    print(f"   Learning Rate: 0.001")
    print(f"   Batch Size: 64")
    print(f"   Epochs: 30")

    print(f"\n[3.2] Potential Issues:")

    issues = []

    # Learning rate check
    lr = 0.001
    if lr < 0.005:
        issues.append("Learning rate 0.001 may be too low for scaled data")
        print(f"   ⚠️  {issues[-1]}")
        print(f"      Recommendation: Try 0.01 or 0.005")

    # Model capacity check
    hidden_size = 32
    if hidden_size < 64:
        issues.append("Hidden size 32 may be insufficient")
        print(f"   ⚠️  {issues[-1]}")
        print(f"      Recommendation: Try 128 or 256")

    # Batch size check
    batch_size = 64
    print(f"   ✅ Batch size {batch_size} is reasonable")

    return issues


def simulate_one_epoch(dataset, model):
    """
    Check 4: Simulate one training epoch to see if model learns
    """
    print("\n" + "=" * 80)
    print("CHECK 4: SIMULATE ONE EPOCH")
    print("=" * 80)

    from torch.utils.data import DataLoader, random_split

    # Small subset for quick test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, _ = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n[4.1] Training on {len(train_dataset)} samples...")

    model.train()
    epoch_loss = 0.0
    sample_losses = []

    for i, (X_batch, y_batch) in enumerate(train_loader):
        if i >= 10:  # Only 10 batches for quick test
            break

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        sample_losses.append(loss.item())

        if i == 0:
            print(f"   First batch loss: {loss.item():.6f}")
            print(f"   Predictions sample: {predictions[:3].flatten().detach().cpu().numpy()}")
            print(f"   Targets sample: {y_batch[:3].flatten().detach().cpu().numpy()}")

    avg_loss = epoch_loss / min(10, len(train_loader))
    print(f"\n[4.2] Average Loss (10 batches): {avg_loss:.6f}")

    issues = []

    # Check if loss is reasonable
    if avg_loss > 1.0:
        issues.append("Loss is very high - possible scaling issue")
        print(f"   ❌ {issues[-1]}")
    elif avg_loss < 0.01:
        issues.append("Loss is very low - possible data leakage")
        print(f"   ❌ {issues[-1]}")
    else:
        print(f"   ✅ Loss is in reasonable range")

    # Check gradient flow
    print(f"\n[4.3] Gradient Check:")
    model.train()
    X_batch, y_batch = next(iter(train_loader))
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    optimizer.zero_grad()
    predictions = model(X_batch)
    loss = criterion(predictions, y_batch)
    loss.backward()

    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm < 1e-7:
                print(f"   ❌ {name}: gradient near zero ({grad_norm:.2e})")
                issues.append(f"Gradient vanishing in {name}")

    if len(grad_norms) > 0:
        avg_grad = np.mean(grad_norms)
        print(f"   Average gradient norm: {avg_grad:.6f}")

        if avg_grad < 1e-5:
            issues.append("Gradients are very small - vanishing gradient problem")
            print(f"   ⚠️  {issues[-1]}")
        else:
            print(f"   ✅ Gradients are flowing")

    return issues


def check_evaluation_metrics(dataset, model):
    """
    Check 5: Verify evaluation metrics are calculated correctly
    """
    print("\n" + "=" * 80)
    print("CHECK 5: EVALUATION METRICS")
    print("=" * 80)

    from torch.utils.data import DataLoader

    # Load a sample
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    predictions_scaled = []
    actuals_scaled = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            if len(predictions_scaled) >= 1000:  # Sample 1000
                break

            X_batch = X_batch.to(device)
            pred_scaled = model(X_batch).cpu().numpy().flatten()
            actual_scaled = y_batch.numpy().flatten()

            predictions_scaled.extend(pred_scaled)
            actuals_scaled.extend(actual_scaled)

    predictions_scaled = np.array(predictions_scaled)
    actuals_scaled = np.array(actuals_scaled)

    # Inverse transform
    predictions = dataset.target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    actuals = dataset.target_scaler.inverse_transform(actuals_scaled.reshape(-1, 1)).flatten()

    print(f"\n[5.1] Scaled Predictions:")
    print(f"   Mean: {predictions_scaled.mean():.6f}")
    print(f"   Std: {predictions_scaled.std():.6f}")
    print(f"   Range: [{predictions_scaled.min():.6f}, {predictions_scaled.max():.6f}]")

    print(f"\n[5.2] Scaled Actuals:")
    print(f"   Mean: {actuals_scaled.mean():.6f}")
    print(f"   Std: {actuals_scaled.std():.6f}")
    print(f"   Range: [{actuals_scaled.min():.6f}, {actuals_scaled.max():.6f}]")

    print(f"\n[5.3] Inverse-Transformed Predictions:")
    print(f"   Mean: {predictions.mean():.6f}")
    print(f"   Std: {predictions.std():.6f}")
    print(f"   Range: [{predictions.min():.6f}, {predictions.max():.6f}]")

    print(f"\n[5.4] Inverse-Transformed Actuals:")
    print(f"   Mean: {actuals.mean():.6f}")
    print(f"   Std: {actuals.std():.6f}")
    print(f"   Range: [{actuals.min():.6f}, {actuals.max():.6f}]")

    # Calculate metrics
    qlike = qlike_loss(actuals, predictions)
    dir_acc = directional_accuracy(actuals, predictions)

    print(f"\n[5.5] Metrics:")
    print(f"   QLIKE: {qlike:.6f}")
    print(f"   Directional Accuracy: {dir_acc:.2%}")

    issues = []

    # Check if model is just predicting mean
    pred_mean = predictions.mean()
    actual_mean = actuals.mean()

    if abs(pred_mean - actual_mean) < 0.01 * actual_mean:
        issues.append("Model is predicting close to mean - no learning")
        print(f"   ❌ {issues[-1]}")
        print(f"      Prediction mean: {pred_mean:.6f}")
        print(f"      Actual mean: {actual_mean:.6f}")

    if qlike > 0.5:
        issues.append("QLIKE is very high - model not learning")
        print(f"   ❌ {issues[-1]}")

    if dir_acc < 0.55:
        issues.append("Directional accuracy near random - model useless")
        print(f"   ❌ {issues[-1]}")

    return issues


def main():
    """Main execution - Run all checks."""
    print("\n" + "=" * 80)
    print("LSTM TRAINING FAILURE - ROOT CAUSE ANALYSIS")
    print("=" * 80)

    # Get project root correctly (from src/lstm_baseline/ go up 2 levels)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, 'data/processed')

    all_issues = []

    # Check 1: Data quality
    dataset, data_issues = check_data_quality(data_dir)
    all_issues.extend(data_issues)

    # Check 2: Model architecture
    model, model_issues = check_model_architecture()
    all_issues.extend(model_issues)

    # Check 3: Training setup
    training_issues = check_training_setup()
    all_issues.extend(training_issues)

    # Check 4: Simulate training
    training_sim_issues = simulate_one_epoch(dataset, model)
    all_issues.extend(training_sim_issues)

    # Check 5: Evaluation metrics
    eval_issues = check_evaluation_metrics(dataset, model)
    all_issues.extend(eval_issues)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - ALL ISSUES FOUND")
    print("=" * 80)

    if all_issues:
        print(f"\nTotal Issues: {len(all_issues)}")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        print("\n1. IMMEDIATE FIXES:")
        print("   - Double target scaler fit (bug in dataset.py line 58)")
        print("   - Increase hidden size: 32 -> 128")
        print("   - Increase learning rate: 0.001 -> 0.01")

        print("\n2. ADD DEBUGGING:")
        print("   - Add prediction/actual statistics after inverse-transform")
        print("   - Add gradient norm monitoring")
        print("   - Add sample predictions visualization")

        print("\n3. VERIFY FIX:")
        print("   - Re-train with fixes")
        print("   - Check QLIKE < 0.20")
        print("   - Check Directional Accuracy > 55%")
    else:
        print("\n✅ No critical issues found in checks")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
