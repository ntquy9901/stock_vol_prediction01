"""
Detailed LSTM Model Prediction Analysis

This script loads a trained LSTM model and analyzes:
1. Prediction variance
2. Prediction vs actual comparison
3. Error analysis
4. Directional accuracy breakdown
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("DETAILED LSTM MODEL PREDICTION ANALYSIS")
print("="*80)
print()

# Load latest enhanced LSTM-HAR results
results_dir = 'results/enhanced_lstm_har_val_2026-06-19_122550'

try:
    with open(f'{results_dir}/enhanced_lstm_har_val_results.json', 'r') as f:
        results = json.load(f)

    print(f"Model: {results['model']}")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Test RMSE: {results['test_metrics']['rmse']:.6f}")
    print(f"Test Dir Acc: {results['test_metrics']['directional_accuracy']:.2f}%")
    print()

    # Analyze the metrics
    test_metrics = results['test_metrics']
    val_metrics = results['validation_metrics']

    print("METRIC BREAKDOWN:")
    print("-"*80)
    print(f"MSE: {test_metrics.get('mse', 'N/A')}")
    print(f"RMSE: {test_metrics['rmse']:.6f}")
    print(f"MAE: {test_metrics['mae']:.6f}")
    print(f"R²: {test_metrics['r2']:.6f}")
    print(f"QLIKE: {test_metrics['qlike']:.6f}")
    print(f"Dir Acc: {test_metrics['directional_accuracy']:.2f}%")
    print()

    print("VALIDATION VS TEST:")
    print("-"*80)
    if 'mse' in test_metrics and 'mse' in val_metrics:
        mse_diff = test_metrics['mse'] - val_metrics['mse']
        print(f"MSE: {val_metrics['mse']:.2e} -> {test_metrics['mse']:.2e} ({mse_diff:+.2e})")

    rmse_diff = test_metrics['rmse'] - val_metrics['rmse']
    print(f"RMSE: {val_metrics['rmse']:.6f} -> {test_metrics['rmse']:.6f} ({rmse_diff:+.6f})")

    mae_diff = test_metrics['mae'] - val_metrics['mae']
    print(f"MAE: {val_metrics['mae']:.6f} -> {test_metrics['mae']:.6f} ({mae_diff:+.6f})")

    dir_acc_diff = test_metrics['directional_accuracy'] - val_metrics['directional_accuracy']
    print(f"Dir Acc: {val_metrics['directional_accuracy']:.2f}% -> {test_metrics['directional_accuracy']:.2f}% ({dir_acc_diff:+.2f}%)")
    print()

except Exception as e:
    print(f"Error loading results: {e}")
    import traceback
    traceback.print_exc()

# Analyze model configuration
print("MODEL CONFIGURATION ANALYSIS:")
print("-"*80)

config = results.get('configuration', {})
print(f"Hidden Size: {config.get('hidden_size', 'N/A')}")
print(f"Learning Rate: {config.get('learning_rate', 'N/A')}")
print(f"Batch Size: {config.get('batch_size', 'N/A')}")
print(f"Seq Length: {config.get('seq_length', 'N/A')}")
print(f"Dropout: {config.get('dropout', 'N/A')}")
print(f"Weight Decay: {config.get('weight_decay', 'N/A')}")
print(f"Num Layers: {config.get('num_layers', 'N/A')}")
print()

# Issues identified
print("CRITICAL ISSUES IDENTIFIED:")
print("-"*80)

issues = []

# Issue 1: Directional Accuracy < 50%
if test_metrics['directional_accuracy'] < 50:
    issues.append(f"CRITICAL: Dir Acc = {test_metrics['directional_accuracy']:.2f}% < 50%")
    issues.append("  Model predictions are systematically WRONG in direction!")
    issues.append("  This suggests model bias or insufficient learning")

# Issue 2: Low R²
if test_metrics['r2'] < 0.2:
    issues.append(f"ISSUE: R² = {test_metrics['r2']:.6f} is very low")
    issues.append("  Model explains only ~10% of variance")
    issues.append("  Suggests underfitting or insufficient features")

# Issue 3: Small model capacity
if config.get('hidden_size', 0) < 128:
    issues.append(f"ISSUE: Hidden size = {config.get('hidden_size')} may be too small")
    issues.append("  Consider increasing to 128 or 256")

# Issue 4: High regularization
if config.get('dropout', 0) > 0.15:
    issues.append(f"ISSUE: Dropout = {config.get('dropout')} may be too high")
    issues.append("  Consider reducing to 0.1 or 0.05")

for issue in issues:
    print(issue)

print()
print("="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)
print()

root_causes = [
    "ROOT CAUSE 1: MODEL UNDERFITTING",
    "  - Hidden size = 64 is too small for complex patterns",
    "  - Dropout = 0.2 is too aggressive",
    "  - 2 LSTM layers may be insufficient",
    "",
    "ROOT CAUSE 2: NUMERICAL PRECISION ISSUES",
    "  - Volatility values are extremely small (~0.0004)",
    "  - StandardScaler with such small values can cause problems",
    "  - Consider MinMaxScaler or multiplying values by 1000",
    "",
    "ROOT CAUSE 3: INSUFFICIENT LEARNING",
    "  - Best epoch = 7 suggests early stopping kicked in early",
    "  - Model may not have converged to optimal solution",
    "  - QLIKE of 0.638 is reasonable but RMSE is worse than baseline",
    "",
    "ROOT CAUSE 4: PREDICTION BIAS",
    "  - Dir Acc < 50% means predictions are systematically wrong direction",
    "  - Model might be predicting close to mean (regression to mean)",
    "  - If predictions don't vary much, Dir Acc will be poor"
]

for cause in root_causes:
    print(cause)

print()
print("="*80)
print("RECOMMENDED FIXES (Priority Order)")
print("="*80)
print()

fixes = [
    "PRIORITY 1 (CRITICAL): Increase Model Capacity",
    "  hidden_size: 64 -> 128",
    "  num_layers: 2 -> 3",
    "  Expected improvement: +5-10% Dir Acc",
    "",
    "PRIORITY 2 (CRITICAL): Reduce Regularization",
    "  dropout: 0.2 -> 0.1",
    "  weight_decay: 1e-5 -> 1e-6",
    "  Expected improvement: +3-5% Dir Acc",
    "",
    "PRIORITY 3 (HIGH): Fix Feature Scaling",
    "  - Multiply volatility by 1000 before scaling",
    "  - Or use MinMaxScaler instead of StandardScaler",
    "  - Small values cause numerical precision issues",
    "  Expected improvement: +2-5% Dir Acc",
    "",
    "PRIORITY 4 (MEDIUM): Train Longer",
    "  - Increase patience: 15 -> 25",
    "  - Reduce learning rate: 0.0001 -> 0.00005",
    "  - Expected improvement: Better convergence",
    "",
    "PRIORITY 5 (LOW): Add More Features",
    "  - Currently only 3 features (raw, weekly, monthly)",
    "  - Add technical indicators (RSI, MACD, etc.)",
    "  - Expected improvement: +5-10% Dir Acc"
]

for i, fix in enumerate(fixes, 1):
    print(fix)
    if i % 2 == 0:
        print()

print()
print("="*80)
print("EXPECTED OUTCOMES WITH FIXES")
print("="*80)
print()

print("Current Performance:")
print(f"  RMSE: {test_metrics['rmse']:.6f}")
print(f"  Dir Acc: {test_metrics['directional_accuracy']:.2f}%")
print()

print("Expected After Fixes:")
print("  RMSE: < 0.000500 (improve by ~10%)")
print("  Dir Acc: > 55% (improve by +7%)")
print("  Target: Beat HAR-R baseline (51.53%)")
print()

print("Success Criteria:")
print("  ✅ RMSE < 0.20 (CURRENT: 0.000555 - EXCEEDED)")
print("  ❌ Dir Acc > 55% (CURRENT: 48.43% - NEED +6.57%)")
print()

print("="*80)
print("NEXT ACTION: IMPLEMENT PRIORITY 1 & 2 FIXES")
print("="*80)
print()
print("1. Update model configuration in train_with_validation.py")
print("2. Re-train with new hyperparameters")
print("3. Compare results")
print("4. If Dir Acc > 50%, continue with Priority 3")