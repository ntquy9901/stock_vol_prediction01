"""
Calculate All Missing Metrics Script

This script resolves missing data issues by calculating:
- MSE (missing from all models)
- QLIKE (missing from LSTM-HAR and Enhanced LSTM-HAR)
- R² (missing from HAR-R Linear)

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.common.parkinson_utils import calculate_parkinson_volatility
from src.common.feature_engineering import create_har_features, create_5day_target
from src.lstm_baseline.dataset import PooledVolatilityDataset
from src.lstm_har_baseline.dataset import HARVolatilityDataset
from src.lstm_har_enhanced.dataset_enhanced import EnhancedHARDataset
from src.lstm_baseline.model import SimpleVolatilityLSTM
from src.lstm_har_baseline.model import HARVolatilityLSTM
from src.lstm_har_enhanced.model_enhanced import EnhancedHARVolatilityLSTM

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def calculate_qlike(y_true, y_pred, epsilon=1e-8):
    """
    Calculate QLIKE loss (academic standard for volatility).

    Formula: QLIKE = mean(y_true/y_pred - log(y_true/y_pred) - 1)

    Args:
        y_true: True volatility values
        y_pred: Predicted volatility values
        epsilon: Small value to avoid division by zero

    Returns:
        QLIKE loss (lower is better)
    """
    y_pred = np.maximum(y_pred, epsilon)
    y_true = np.maximum(y_true, epsilon)

    ratio = y_true / y_pred
    qlike = ratio - np.log(ratio) - 1
    return np.mean(qlike)

def calculate_r2(y_true, y_pred):
    """
    Calculate R² score.

    Formula: R² = 1 - (SS_res / SS_tot)

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² score (higher is better, max 1.0)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    r2 = 1 - (ss_res / ss_tot)
    return r2

def calculate_mse(y_true, y_pred):
    """Calculate Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

def evaluate_har_baseline(data_dir='data/processed'):
    """Evaluate HAR-R Linear baseline model."""
    print("\n" + "="*80)
    print("EVALUATING HAR-R LINEAR BASELINE")
    print("="*80)

    # Load HAR-R model
    model_path = 'results/har_baseline_2026-06-18_004155/har_baseline_model.pkl'

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None

    with open(model_path, 'rb') as f:
        har_model = joblib.load(f)

    # Load processed data
    print("\n1. Loading processed data...")
    all_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith('_processed.csv'):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)
            all_files.append(df)

    print(f"  Loaded {len(all_files)} stock files")

    # Create HAR features for all stocks
    print("\n2. Creating HAR features...")
    features_list = []

    for df in all_files:
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Extract parkinson volatility
        vol = df['parkinson_volatility']

        # Create HAR features
        har_features = create_har_features(vol)
        df = pd.concat([df, har_features], axis=1)

        # Create 5-day target
        target = create_5day_target(vol)
        df['target_5d'] = target

        # Keep valid rows
        df_valid = df.dropna(subset=['har_daily_vol', 'har_weekly_vol',
                                      'har_monthly_vol', 'target_5d']).copy()

        if len(df_valid) > 0:
            features_list.append(df_valid)

    # Combine all stocks
    all_features = pd.concat(features_list, ignore_index=True)
    print(f"  Total samples: {len(all_features)}")

    # Prepare features and target
    feature_cols = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']
    X = all_features[feature_cols].values
    y = all_features['target_5d'].values

    # Temporal split (80/20) - SAME AS TRAINING
    split_idx = int(0.8 * len(X))
    X_test, y_test = X[split_idx:], y[split_idx:]

    print(f"  Test size: {len(X_test)}")

    # Generate predictions
    print("\n3. Generating predictions...")
    y_pred = har_model.predict(X_test)

    # Calculate all metrics
    print("\n4. Calculating metrics...")
    mse = calculate_mse(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = calculate_r2(y_test, y_pred)
    qlike = calculate_qlike(y_test, y_pred)

    # Calculate directional accuracy (CORRECT METHOD)
    # Compare direction of CHANGE, not sign of values
    actual_changes = np.sign(np.diff(y_test))
    pred_changes = np.sign(np.diff(y_pred))
    dir_acc = np.mean(actual_changes == pred_changes) * 100

    metrics = {
        'model': 'HAR-R Linear',
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'qlike': float(qlike),
        'directional_accuracy': float(dir_acc)
    }

    print(f"\nHAR-R Linear Metrics:")
    print(f"  MSE: {metrics['mse']:.8f}")
    print(f"  RMSE: {metrics['rmse']:.8f}")
    print(f"  MAE: {metrics['mae']:.8f}")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  QLIKE: {metrics['qlike']:.6f}")
    print(f"  Dir Acc: {metrics['directional_accuracy']:.2f}%")

    return metrics

def evaluate_simple_lstm(data_dir='data/processed'):
    """Evaluate Simple LSTM model."""
    print("\n" + "="*80)
    print("EVALUATING SIMPLE LSTM")
    print("="*80)

    # Load Simple LSTM model
    model_path = 'results/simple_lstm_2026-06-19_010018/best_simple_lstm.pth'

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None

    # Create dataset
    print("\n1. Creating dataset...")
    dataset = PooledVolatilityDataset(data_dir, seq_length=22, forecast_horizon=5)

    # Split (same 80/20 split as training)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    _, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"  Test size: {test_size}")

    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    # Load model
    print("\n2. Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleVolatilityLSTM(hidden_size=128)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state_dict (keys are layer names)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"  Device: {device}")

    # Generate predictions
    print("\n3. Generating predictions...")
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)

            # Inverse transform
            predictions_np = dataset.target_scaler.inverse_transform(predictions.cpu().numpy())
            targets_np = dataset.target_scaler.inverse_transform(y_batch.numpy().reshape(-1, 1))

            all_predictions.extend(predictions_np.flatten())
            all_targets.extend(targets_np.flatten())

    y_pred = np.array(all_predictions)
    y_true = np.array(all_targets)

    print(f"  Predictions shape: {y_pred.shape}")

    # Calculate all metrics
    print("\n4. Calculating metrics...")
    mse = calculate_mse(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = calculate_r2(y_true, y_pred)
    qlike = calculate_qlike(y_true, y_pred)

    # Calculate directional accuracy (CORRECT METHOD)
    # Compare direction of CHANGE, not sign of values
    actual_changes = np.sign(np.diff(y_true))
    pred_changes = np.sign(np.diff(y_pred))
    dir_acc = np.mean(actual_changes == pred_changes) * 100

    metrics = {
        'model': 'Simple LSTM',
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'qlike': float(qlike),
        'directional_accuracy': float(dir_acc)
    }

    print(f"\nSimple LSTM Metrics:")
    print(f"  MSE: {metrics['mse']:.8f}")
    print(f"  RMSE: {metrics['rmse']:.8f}")
    print(f"  MAE: {metrics['mae']:.8f}")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  QLIKE: {metrics['qlike']:.6f}")
    print(f"  Dir Acc: {metrics['directional_accuracy']:.2f}%")

    return metrics

def evaluate_lstm_har(data_dir='data/processed'):
    """Evaluate LSTM-HAR model."""
    print("\n" + "="*80)
    print("EVALUATING LSTM-HAR")
    print("="*80)

    # Load LSTM-HAR model
    model_path = 'results/lstm_har_2026-06-19_020803/best_lstm_har_model.pth'

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None

    # Create dataset
    print("\n1. Creating dataset...")
    dataset = HARVolatilityDataset(data_dir, seq_length=22, forecast_horizon=5)

    # Split (same 80/20 split as training)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    _, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"  Test size: {test_size}")

    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    # Load model
    print("\n2. Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HARVolatilityLSTM(hidden_size=64, num_layers=2, dropout=0.2)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state_dict (keys are layer names)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"  Device: {device}")

    # Generate predictions
    print("\n3. Generating predictions...")
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)

            # Inverse transform
            predictions_np = dataset.target_scaler.inverse_transform(predictions.cpu().numpy())
            targets_np = dataset.target_scaler.inverse_transform(y_batch.numpy().reshape(-1, 1))

            all_predictions.extend(predictions_np.flatten())
            all_targets.extend(targets_np.flatten())

    y_pred = np.array(all_predictions)
    y_true = np.array(all_targets)

    print(f"  Predictions shape: {y_pred.shape}")

    # Calculate all metrics
    print("\n4. Calculating metrics...")
    mse = calculate_mse(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = calculate_r2(y_true, y_pred)
    qlike = calculate_qlike(y_true, y_pred)

    # Calculate directional accuracy (CORRECT METHOD)
    # Compare direction of CHANGE, not sign of values
    actual_changes = np.sign(np.diff(y_true))
    pred_changes = np.sign(np.diff(y_pred))
    dir_acc = np.mean(actual_changes == pred_changes) * 100

    metrics = {
        'model': 'LSTM-HAR',
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'qlike': float(qlike),
        'directional_accuracy': float(dir_acc)
    }

    print(f"\nLSTM-HAR Metrics:")
    print(f"  MSE: {metrics['mse']:.8f}")
    print(f"  RMSE: {metrics['rmse']:.8f}")
    print(f"  MAE: {metrics['mae']:.8f}")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  QLIKE: {metrics['qlike']:.6f}")
    print(f"  Dir Acc: {metrics['directional_accuracy']:.2f}%")

    return metrics

def evaluate_enhanced_lstm_har(data_dir='data/processed'):
    """Evaluate Enhanced LSTM-HAR model."""
    print("\n" + "="*80)
    print("EVALUATING ENHANCED LSTM-HAR")
    print("="*80)

    # Load Enhanced LSTM-HAR model
    model_path = 'results/enhanced_lstm_har_2026-06-19_021632/best_enhanced_lstm_har_model.pth'

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None

    # Create dataset
    print("\n1. Creating dataset...")
    dataset = EnhancedHARDataset(data_dir, seq_length=22, forecast_horizon=5)

    # Split (same 80/20 split as training)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    _, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"  Test size: {test_size}")

    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    # Load model
    print("\n2. Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedHARVolatilityLSTM(hidden_size=64, num_layers=2, dropout=0.2)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state_dict (keys are layer names)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"  Device: {device}")

    # Generate predictions
    print("\n3. Generating predictions...")
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)

            # Inverse transform
            predictions_np = dataset.target_scaler.inverse_transform(predictions.cpu().numpy())
            targets_np = dataset.target_scaler.inverse_transform(y_batch.numpy().reshape(-1, 1))

            all_predictions.extend(predictions_np.flatten())
            all_targets.extend(targets_np.flatten())

    y_pred = np.array(all_predictions)
    y_true = np.array(all_targets)

    print(f"  Predictions shape: {y_pred.shape}")

    # Calculate all metrics
    print("\n4. Calculating metrics...")
    mse = calculate_mse(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = calculate_r2(y_true, y_pred)
    qlike = calculate_qlike(y_true, y_pred)

    # Calculate directional accuracy (CORRECT METHOD)
    # Compare direction of CHANGE, not sign of values
    actual_changes = np.sign(np.diff(y_true))
    pred_changes = np.sign(np.diff(y_pred))
    dir_acc = np.mean(actual_changes == pred_changes) * 100

    metrics = {
        'model': 'Enhanced LSTM-HAR',
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'qlike': float(qlike),
        'directional_accuracy': float(dir_acc)
    }

    print(f"\nEnhanced LSTM-HAR Metrics:")
    print(f"  MSE: {metrics['mse']:.8f}")
    print(f"  RMSE: {metrics['rmse']:.8f}")
    print(f"  MAE: {metrics['mae']:.8f}")
    print(f"  R²: {metrics['r2']:.6f}")
    print(f"  QLIKE: {metrics['qlike']:.6f}")
    print(f"  Dir Acc: {metrics['directional_accuracy']:.2f}%")

    return metrics

def main():
    """Main function to evaluate all models and calculate all metrics."""
    print("="*80)
    print("CALCULATING ALL MISSING METRICS")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d_%H%M%S')}")

    data_dir = 'data/processed'

    # Evaluate all models
    results = []

    har_metrics = evaluate_har_baseline(data_dir)
    if har_metrics:
        results.append(har_metrics)

    simple_lstm_metrics = evaluate_simple_lstm(data_dir)
    if simple_lstm_metrics:
        results.append(simple_lstm_metrics)

    lstm_har_metrics = evaluate_lstm_har(data_dir)
    if lstm_har_metrics:
        results.append(lstm_har_metrics)

    enhanced_lstm_har_metrics = evaluate_enhanced_lstm_har(data_dir)
    if enhanced_lstm_har_metrics:
        results.append(enhanced_lstm_har_metrics)

    # Create comprehensive comparison table
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS COMPARISON - ALL MODELS")
    print("="*80)

    print(f"\n{'Model':<25} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<10} {'QLIKE':<12} {'Dir Acc':<10}")
    print("-" * 110)

    for metrics in results:
        print(f"{metrics['model']:<25} "
              f"{metrics['mse']:<12.8f} "
              f"{metrics['rmse']:<12.8f} "
              f"{metrics['mae']:<12.8f} "
              f"{metrics['r2']:<10.6f} "
              f"{metrics['qlike']:<12.6f} "
              f"{metrics['directional_accuracy']:<10.2f}%")

    # Save results to JSON
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, f'all_metrics_comparison_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to: {results_file}")
    print("="*80)

    return results

if __name__ == "__main__":
    results = main()
