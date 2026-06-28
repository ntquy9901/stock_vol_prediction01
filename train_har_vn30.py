"""
Train HAR-R Linear Baseline on VN30 Stocks Only
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.common.parkinson_utils import calculate_parkinson_volatility
from src.common.feature_engineering import create_har_features, create_5day_target
from src.common.evaluation import evaluate_predictions

print('='*80)
print('HAR-R LINEAR BASELINE - VN30 ONLY')
print('='*80)

# Data directory
data_dir = 'data/processed/vn30_only'
output_dir = 'results/har_baseline_vn30_2026-06-20'

os.makedirs(output_dir, exist_ok=True)
print(f'Results will be saved to: {output_dir}')

# Load VN30 processed data
print('\n1. Loading VN30 processed data...')
all_files = []
for filename in os.listdir(data_dir):
    if filename.endswith('_processed.csv'):
        file_path = os.path.join(data_dir, filename)
        df = pd.read_csv(file_path)
        all_files.append(df)

print(f'  Loaded {len(all_files)} VN30 stock files')

# Create HAR features for all VN30 stocks
print('\n2. Creating HAR features for VN30 stocks...')
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

# Combine all VN30 stocks
all_features = pd.concat(features_list, ignore_index=True)
print(f'  Total VN30 samples: {len(all_features)}')

# Prepare features and target
feature_cols = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']
X = all_features[feature_cols].values
y = all_features['target_5d'].values

print(f'  Feature shape: {X.shape}')
print(f'  Target shape: {y.shape}')

# Display sample statistics
print(f'\n  VN30 Data Statistics:')
print(f'    X mean: {X.mean():.6f}, std: {X.std():.6f}')
print(f'    y mean: {y.mean():.6f}, std: {y.std():.6f}')

# Temporal train/test split (80/20) - CHRONOLOGICAL
print('\n3. Splitting VN30 data chronologically (temporal split)...')
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f'  Train size: {len(X_train)}')
print(f'  Test size: {len(X_test)}')

# Train HAR-R Linear Regression model
print('\n4. Training HAR-R Linear Regression on VN30...')
model = LinearRegression()

import time
start_time = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time

print(f'  Training time: {train_time:.3f} seconds')
print(f'  Coefficients: {model.coef_}')
print(f'  Intercept: {model.intercept_:.6f}')

# Feature importance (coefficients)
print('\n5. VN30 Feature Importance:')
for i, col in enumerate(feature_cols):
    print(f'  {col:20s}: {model.coef_[i]:10.6f}')

# Make predictions on test set
print('\n6. Evaluating on VN30 test set...')
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics using src/common/evaluation
print('\n7. VN30 Test Results:')
print('-'*80)

metrics = evaluate_predictions(y_test, y_pred_test)

print(f'\nVN30-ONLY PERFORMANCE:')
for metric_name, value in metrics.items():
    if metric_name == 'Directional_Acc':
        print(f'  {metric_name}: {value:.2f}%')
    else:
        print(f'  {metric_name}: {value:.6f}')

# Save results to CSV
results_df = pd.DataFrame([metrics])
results_df.to_csv(os.path.join(output_dir, 'test_metrics.csv'), index=False)

# Save trained model
joblib.dump(model, os.path.join(output_dir, 'har_baseline_vn30_model.pkl'))
print(f'\nModel saved to: {output_dir}/har_baseline_vn30_model.pkl')

# Save training info (model coefficients, etc.)
total_samples = int(len(X))
info = {
    'model': 'HAR-R Linear (VN30-Only)',
    'dataset': 'VN30 stocks only (26/28 stocks)',
    'features': feature_cols,
    'target': 'target_5d',
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test)),
    'total_samples': total_samples,
    'vn30_stocks': len(all_files),
    'training_time_seconds': float(train_time),
    'coefficients': dict(zip(feature_cols, model.coef_.tolist())),
    'intercept': float(model.intercept_),
    'data_source': 'Parkinson volatility from VN30 processed CSV files',
    'test_metrics': {
        'mse': float(metrics['mse']),
        'rmse': float(metrics['rmse']),
        'mae': float(metrics['mae']),
        'r2': float(metrics['r2']),
        'qlike': float(metrics['qlike']),
        'directional_accuracy': float(metrics['directional_accuracy'])
    }
}

with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
    json.dump(info, f, indent=2)

print('\n' + '='*80)
print('HAR-R BASELINE (VN30-ONLY) TRAINING COMPLETE!')
print(f'Total time: {train_time:.3f} seconds')
print(f'Results saved to: {output_dir}/')
print('='*80)

print('\nVN30 MODEL SUMMARY:')
print(f'  Dataset: {info["vn30_stocks"]} VN30 stocks')
print(f'  Samples: {info["total_samples"]:,} total ({info["train_size"]:,} train, {info["test_size"]:,} test)')
print(f'  Dir Acc: {info["test_metrics"]["directional_accuracy"]:.2f}%')
print(f'  RMSE: {info["test_metrics"]["rmse"]:.6f}')
print(f'  R²: {info["test_metrics"]["r2"]:.6f}')

# Success criteria check
print('\nSUCCESS CRITERIA CHECK:')
dir_acc = info["test_metrics"]["directional_accuracy"]
rmse = info["test_metrics"]["rmse"]

print(f'  RMSE Target (<0.20): {"PASS" if rmse < 0.20 else "FAIL"} - Actual: {rmse:.6f}')
print(f'  Dir Acc Target (>55%): {"PASS" if dir_acc > 55 else "FAIL"} - Actual: {dir_acc:.2f}%')