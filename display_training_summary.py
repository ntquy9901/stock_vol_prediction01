"""
Display Training Summary for Simple LSTM and LSTM-HAR
"""
from pathlib import Path
import json

print("="*80)
print("VN30 MODEL TRAINING SUMMARY")
print("="*80)

# Simple LSTM Results
print("\n[1] Simple LSTM (1-Layer, 128 hidden, Raw Volatility)")
print("-"*80)
simple_results = Path('results/simple_lstm_vn30_2026-06-20/training_results.json')
if simple_results.exists():
    with open(simple_results, 'r') as f:
        results = json.load(f)

    metrics = results['test_metrics']
    comparison = results['comparison_with_baseline']

    print("Status: [OK] Training Complete")
    print(f"  Dir Acc:     {metrics['directional_accuracy']:.2f}%")
    print(f"  RMSE:        {metrics['rmse']:.6f}")
    print(f"  R²:          {metrics['r2']:.6f}")
    print(f"  QLIKE:       {metrics['qlike']:.6f}")
    print()
    print("  vs HAR-R Baseline:")
    print(f"    Dir Acc:  {comparison['har_r_dir_acc']:.2f}% -> {metrics['directional_accuracy']:.2f}% (+" + f"{comparison['dir_acc_improvement']:.2f}%)")
    print(f"    RMSE:     {comparison['har_r_rmse']:.6f} -> {metrics['rmse']:.6f} (" + f"{comparison['rmse_difference']:+.6f})")
    print(f"    R²:       {comparison['har_r_r2']:.3f} -> {metrics['r2']:.3f} (+" + f"{comparison['r2_improvement_percent']:.1f}%)")
    print()
    print("  Training:")
    print(f"    Best Epoch: {results['training']['best_epoch']}")
    print(f"    Total Epochs: {results['training']['total_epochs']}")
    print(f"    Early Stopped: {results['training']['early_stopped']}")
else:
    print("Status: [...] Training results not found")

# LSTM-HAR Results
print("\n[2] LSTM-HAR (2-Layer, 64 hidden, HAR Features)")
print("-"*80)
lstm_har_results = Path('results/lstm_har_vn30_2026-06-20/training_results.json')
if lstm_har_results.exists():
    with open(lstm_har_results, 'r') as f:
        results = json.load(f)

    metrics = results['test_metrics']
    comparison = results.get('comparison_with_baseline', {})

    print("Status: [OK] Training Complete")
    print(f"  Dir Acc:     {metrics['directional_accuracy']:.2f}%")
    print(f"  RMSE:        {metrics['rmse']:.6f}")
    print(f"  R²:          {metrics['r2']:.6f}")
    print(f"  QLIKE:       {metrics['qlike']:.6f}")
    print()
    if comparison:
        print("  vs HAR-R Baseline:")
        print(f"    Dir Acc:  {comparison['har_r_dir_acc']:.2f}% -> {metrics['directional_accuracy']:.2f}% (+" + f"{comparison['dir_acc_improvement']:.2f}%)")
        print(f"    RMSE:     {comparison['har_r_rmse']:.6f} -> {metrics['rmse']:.6f} (" + f"{comparison['rmse_difference']:+.6f})")
    print()
    print("  Training:")
    print(f"    Best Epoch: {results['training']['best_epoch']}")
    print(f"    Total Epochs: {results['training']['total_epochs']}")
    print(f"    Early Stopped: {results['training']['early_stopped']}")
else:
    print("Status: [...] Training in progress")

# Enhanced LSTM-HAR (for comparison)
print("\n[3] Enhanced LSTM-HAR (3-Layer, 128 hidden, HAR Features) - Reference")
print("-"*80)
enhanced_results = Path('results/enhanced_lstm_har_vn30_2026-06-20/training_results.json')
if enhanced_results.exists():
    with open(enhanced_results, 'r') as f:
        results = json.load(f)

    metrics = results['test_metrics']
    comparison = results['comparison_with_baseline']

    print("Status: [OK] Training Complete")
    print(f"  Dir Acc:     {metrics['directional_accuracy']:.2f}%")
    print(f"  RMSE:        {metrics['rmse']:.6f}")
    print(f"  R²:          {metrics['r2']:.6f}")
    print(f"  QLIKE:       {metrics['qlike']:.6f}")
    print()
    print("  vs HAR-R Baseline:")
    print(f"    Dir Acc:  {comparison['har_r_dir_acc']:.2f}% -> {metrics['directional_accuracy']:.2f}% (+" + f"{comparison['dir_acc_improvement']:.2f}%)")
    print(f"    RMSE:     {comparison['har_r_rmse']:.6f} -> {metrics['rmse']:.6f} (" + f"{comparison['rmse_difference']:+.6f})")
    print(f"    R²:       {comparison['har_r_r2']:.3f} -> {metrics['r2']:.3f} (+" + f"{comparison['r2_improvement_percent']:.1f}%)")
else:
    print("Status: [X] Not found")

print("\n" + "="*80)
print("MODEL RANKING (by Directional Accuracy)")
print("="*80)

models = []
if simple_results.exists():
    with open(simple_results, 'r') as f:
        r = json.load(f)
    models.append(('Simple LSTM', r['test_metrics']['directional_accuracy']))

if lstm_har_results.exists():
    with open(lstm_har_results, 'r') as f:
        r = json.load(f)
    models.append(('LSTM-HAR (2L)', r['test_metrics']['directional_accuracy']))

if enhanced_results.exists():
    with open(enhanced_results, 'r') as f:
        r = json.load(f)
    models.append(('Enhanced LSTM-HAR', r['test_metrics']['directional_accuracy']))

# Always add HAR-R
models.append(('HAR-R Linear', 51.53))

# Sort by Dir Acc descending
models.sort(key=lambda x: x[1], reverse=True)

for i, (name, acc) in enumerate(models, 1):
    status = "[OK]" if i == 1 else ("[2nd]" if i == 2 else "[3rd]" if i == 3 else "[X]")
    print(f"{status} {i}. {name:20s} - Dir Acc: {acc:.2f}%")

print("\n" + "="*80)
print("REPORT FILES")
print("="*80)
print("SIMPLE_LSTM_VN30_REPORT.md   - Simple LSTM detailed report [UPDATED]")
print("LSTM_HAR_VN30_REPORT.md     - LSTM-HAR detailed report [UPDATED]")
print("VN30_PERFORMANCE_REPORT.md   - Complete comparison report")
print("="*80)
