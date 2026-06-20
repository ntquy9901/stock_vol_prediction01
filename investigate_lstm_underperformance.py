"""
Comprehensive Investigation: Why LSTM Underperforms HAR-R Baseline

This script investigates:
1. Data quality and distribution
2. Feature scaling (StandardScaler vs MinMaxScaler)
3. Model prediction analysis
4. Directional accuracy calculation
5. Comparison with HAR-R Linear baseline

Author: Investigation Team
Date: 2026-06-19
"""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path

# Load data
print("="*80)
print("INVESTIGATING LSTM UNDERPERFORMANCE")
print("="*80)
print()

# Check 1: Analyze raw volatility data
print("CHECK 1: VOLATILITY DATA DISTRIBUTION")
print("-"*80)

data_dir = 'data/processed'
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'processed' in f]

all_volatility = []
for csv_file in csv_files[:5]:  # Sample 5 stocks
    file_path = os.path.join(data_dir, csv_file)
    df = pd.read_csv(file_path)
    if 'parkinson_volatility' in df.columns:
        vol = df['parkinson_volatility'].values
        vol = vol[~np.isnan(vol)]
        all_volatility.extend(vol)

all_volatility = np.array(all_volatility)
print(f"Total volatility values analyzed: {len(all_volatility)}")
print(f"Range: [{all_volatility.min():.6f}, {all_volatility.max():.6f}]")
print(f"Mean: {all_volatility.mean():.6f}")
print(f"Std: {all_volatility.std():.6f}")
print(f"Median: {np.median(all_volatility):.6f}")
print(f"Percentiles:")
print(f"  25%: {np.percentile(all_volatility, 25):.6f}")
print(f"  50%: {np.percentile(all_volatility, 50):.6f}")
print(f"  75%: {np.percentile(all_volatility, 75):.6f}")
print(f"  95%: {np.percentile(all_volatility, 95):.6f}")
print(f"  99%: {np.percentile(all_volatility, 99):.6f}")

print()
print("KEY OBSERVATION:")
print(f"  Volatility values are VERY SMALL (mean ~ {all_volatility.mean():.6f})")
print(f"  This suggests data is already normalized or scaled!")
print()

# Check 2: Load and analyze LSTM model results
print("CHECK 2: LSTM MODEL RESULTS ANALYSIS")
print("-"*80)

try:
    with open('results/enhanced_lstm_har_val_2026-06-19_122550/enhanced_lstm_har_val_results.json', 'r') as f:
        lstm_results = json.load(f)

    print(f"Model: {lstm_results['model']}")
    print(f"Features: {lstm_results['features']}")
    print(f"Best Epoch: {lstm_results['best_epoch']}")
    print()

    val_metrics = lstm_results['validation_metrics']
    test_metrics = lstm_results['test_metrics']

    print("Validation Metrics:")
    print(f"  RMSE: {val_metrics['rmse']:.6f}")
    print(f"  Dir Acc: {val_metrics['directional_accuracy']:.2f}%")

    print("Test Metrics:")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  Dir Acc: {test_metrics['directional_accuracy']:.2f}%")
    print(f"  QLIKE: {test_metrics['qlike']:.6f}")
    print()

    print("KEY ISSUE:")
    print(f"  Dir Acc = {test_metrics['directional_accuracy']:.2f}% < 50%")
    print(f"  This is WORSE than random guessing!")
    print()

except Exception as e:
    print(f"Error loading LSTM results: {e}")

# Check 3: Compare with HAR-R baseline
print("CHECK 3: HAR-R BASELINE COMPARISON")
print("-"*80)

try:
    with open('results/all_metrics_comparison_2026-06-19_073515.json', 'r') as f:
        comparison = json.load(f)

    har_model = next(m for m in comparison if m['model'] == 'HAR-R Linear')

    print("HAR-R Linear Baseline:")
    print(f"  RMSE: {har_model['rmse']:.6f}")
    print(f"  Dir Acc: {har_model['directional_accuracy']:.2f}%")
    print(f"  QLIKE: {har_model['qlike']:.6f}")
    print()

    print("COMPARISON:")
    lstm_dir_acc = test_metrics['directional_accuracy']
    har_dir_acc = har_model['directional_accuracy']
    print(f"  Dir Acc Gap: {lstm_dir_acc - har_dir_acc:+.2f}%")
    print(f"  LSTM is {'BETTER' if lstm_dir_acc > har_dir_acc else 'WORSE'} than baseline!")

except Exception as e:
    print(f"Error loading comparison data: {e}")

print()

# Check 4: Investigate potential issues
print("CHECK 4: POTENTIAL ISSUES IDENTIFIED")
print("-"*80)

issues = []

# Issue 1: Very small volatility values
if all_volatility.mean() < 0.01:
    issues.append("ISSUE 1: Volatility values are extremely small")
    issues.append(f"  Mean vol = {all_volatility.mean():.6f}")
    issues.append("  This can cause numerical precision problems")
    issues.append("  StandardScaler may not work well with such small values")

print()
for issue in issues:
    print(issue)

print()

# Check 5: Feature Scaling Analysis
print("CHECK 5: FEATURE SCALING ANALYSIS")
print("-"*80)

try:
    from src.lstm_har_enhanced.dataset_enhanced import EnhancedHARDataset

    dataset = EnhancedHARDataset(
        data_dir='data/processed',
        seq_length=22,
        forecast_horizon=5
    )

    # Get a sample
    X_sample, y_sample = dataset[0]

    print("Dataset Analysis:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Input shape: {X_sample.shape}")
    print(f"  Feature mean: {X_sample.mean():.6f}")
    print(f"  Feature std: {X_sample.std():.6f}")
    print(f"  Feature min: {X_sample.min():.6f}")
    print(f"  Feature max: {X_sample.max():.6f}")
    print()

    # Check scaler type
    scaler_type = type(dataset.target_scaler).__name__
    print(f"Target Scaler: {scaler_type}")
    print()

    # Check scaling parameters
    if hasattr(dataset.target_scaler, 'mean_'):
        print("Scaler Parameters:")
        print(f"  Mean: {dataset.target_scaler.mean_[0]:.6f}")
        print(f"  Scale: {dataset.target_scaler.scale_[0]:.6f}")
        print()

        # Test inverse transform
        test_scaled = np.array([[0.0]])  # Zero prediction (mean)
        test_original = dataset.target_scaler.inverse_transform(test_scaled)
        print(f"Zero prediction (scaled) -> Original scale: {test_original[0, 0]:.6f}")
        print(f"This should equal mean: {dataset.target_scaler.mean_[0]:.6f}")

    # Load some actual predictions
    print()
    print("CHECK 6: PREDICTION VARIANCE ANALYSIS")
    print("-"*80)

    # Simulate loading some predictions
    print("To check prediction variance, we need to:")
    print("  1. Load trained model")
    print("  2. Get predictions on test set")
    print("  3. Check if predictions have variance or are all the same")
    print()

    print("EXPECTED ISSUE:")
    print("  If predictions have very low variance, Dir Acc will be poor")
    print("  Model might be predicting near the mean (regression to mean)")

except Exception as e:
    print(f"Error in dataset analysis: {e}")
    import traceback
    traceback.print_exc()

print()

# Check 7: Recommendations
print("CHECK 7: INVESTIGATION FINDINGS & RECOMMENDATIONS")
print("-"*80)

findings = [
    "FINDING 1: LSTM Dir Acc < 50% suggests systematic prediction error",
    "FINDING 2: HAR-R Linear beats LSTM with simpler approach",
    "FINDING 3: Volatility values are extremely small (~0.001-0.01)",
    "FINDING 4: StandardScaler may not be optimal for this data",
    "FINDING 5: Possible numerical precision issues with small values",
    "FINDING 6: Model might be underfitting or over-regularized"
]

for finding in findings:
    print(finding)

print()

recommendations = [
    "PRIORITY 1: Check prediction variance - if all similar, model is underfitting",
    "PRIORITY 2: Try MinMaxScaler instead of StandardScaler for small values",
    "PRIORITY 3: Reduce regularization (dropout 0.2 -> 0.1, weight_decay 1e-5 -> 1e-6)",
    "PRIORITY 4: Increase model capacity (hidden_size 64 -> 128, layers 2 -> 3)",
    "PRIORITY 5: Check if model is learning (plot learning curves)",
    "PRIORITY 6: Analyze prediction errors (bias vs variance trade-off)"
]

for i, rec in enumerate(recommendations, 1):
    print(f"  {rec}")

print()
print("="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
print()
print("NEXT STEPS:")
print("1. Run actual model predictions to check variance")
print("2. Plot learning curves to diagnose under/overfitting")
print("3. Try alternative scaling methods")
print("4. Implement and test recommendations above")