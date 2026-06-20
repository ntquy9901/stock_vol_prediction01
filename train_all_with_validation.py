"""
Train All Models with 3-Way Temporal Split

This script trains all baseline models (HAR-R, Simple LSTM, LSTM-HAR, Enhanced LSTM-HAR)
using proper 3-way temporal split (70/15/15) to fix data leakage issue.

Models:
1. HAR-R Linear (already uses temporal split)
2. Simple LSTM (fixed to use temporal split)
3. LSTM-HAR (fixed to use temporal split)
4. Enhanced LSTM-HAR (fixed to use temporal split)

All models will have:
- Train metrics (70% data)
- Validation metrics (15% data) - for early stopping
- Test metrics (15% data) - final evaluation
- Val vs Test comparison - detect overfitting

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def train_all_models_with_validation(data_dir='data/processed'):
    """
    Train all models with 3-way temporal split evaluation.

    This script orchestrates training of all 4 baseline models:
    - HAR-R Linear (already correct)
    - Simple LSTM (new temporal split)
    - LSTM-HAR (new temporal split)
    - Enhanced LSTM-HAR (new temporal split)

    Each model will produce:
    - Validation metrics (early stopping)
    - Test metrics (final evaluation)
    - Val vs Test comparison
    """
    print("="*80)
    print("TRAINING ALL MODELS - 3-WAY TEMPORAL SPLIT (70/15/15)")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Model 1: HAR-R Linear (already uses temporal split, but needs val/test split)
    print("\n" + "="*80)
    print("MODEL 1/4: HAR-R LINEAR")
    print("="*80)
    try:
        # HAR-R already trained, just load results
        # For now, skip as it needs to be updated to 3-way split
        print("SKIP - HAR-R needs to be updated to 3-way split first")
        print("Use: python src/har_baseline/train.py (already temporal, but 2-way)")
    except Exception as e:
        print(f"ERROR training HAR-R: {e}")

    # Model 2: Simple LSTM (NEW - with temporal split)
    print("\n" + "="*80)
    print("MODEL 2/4: SIMPLE LSTM (WITH VALIDATION)")
    print("="*80)
    try:
        from src.lstm_baseline.train_with_validation import train_simple_lstm_with_val

        model, val_metrics, test_metrics = train_simple_lstm_with_val(data_dir)

        results['simple_lstm'] = {
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'val_test_diff': {
                'rmse_diff': test_metrics['rmse'] - val_metrics['rmse'],
                'mae_diff': test_metrics['mae'] - val_metrics['mae'],
                'r2_diff': test_metrics['r2'] - val_metrics['r2'],
                'dir_acc_diff': test_metrics['directional_accuracy'] - val_metrics['directional_accuracy']
            }
        }

        print(f"[OK] Simple LSTM trained successfully")
        print(f"  Val RMSE: {val_metrics['rmse']:.6f}")
        print(f"  Test RMSE: {test_metrics['rmse']:.6f}")
        print(f"  RMSE Diff: {test_metrics['rmse'] - val_metrics['rmse']:+.6f}")

    except Exception as e:
        print(f"[ERROR] training Simple LSTM: {e}")
        import traceback
        traceback.print_exc()

    # Model 3: LSTM-HAR (NEW - with temporal split)
    print("\n" + "="*80)
    print("MODEL 3/4: LSTM-HAR (WITH VALIDATION)")
    print("="*80)
    try:
        from src.lstm_har_baseline.train_with_validation import train_lstm_har_with_val

        model, val_metrics, test_metrics = train_lstm_har_with_val(data_dir)

        results['lstm_har'] = {
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'val_test_diff': {
                'rmse_diff': test_metrics['rmse'] - val_metrics['rmse'],
                'mae_diff': test_metrics['mae'] - val_metrics['mae'],
                'r2_diff': test_metrics['r2'] - val_metrics['r2'],
                'dir_acc_diff': test_metrics['directional_accuracy'] - val_metrics['directional_accuracy']
            }
        }

        print(f"[OK] LSTM-HAR trained successfully")
        print(f"  Val RMSE: {val_metrics['rmse']:.6f}")
        print(f"  Test RMSE: {test_metrics['rmse']:.6f}")
        print(f"  RMSE Diff: {test_metrics['rmse'] - val_metrics['rmse']:+.6f}")

    except Exception as e:
        print(f"[ERROR] training LSTM-HAR: {e}")
        import traceback
        traceback.print_exc()

    # Model 4: Enhanced LSTM-HAR (NEW - with temporal split)
    print("\n" + "="*80)
    print("MODEL 4/4: ENHANCED LSTM-HAR (WITH VALIDATION)")
    print("="*80)
    try:
        from src.lstm_har_enhanced.train_with_validation import train_enhanced_lstm_har_with_val

        model, val_metrics, test_metrics = train_enhanced_lstm_har_with_val(data_dir)

        results['enhanced_lstm_har'] = {
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'val_test_diff': {
                'rmse_diff': test_metrics['rmse'] - val_metrics['rmse'],
                'mae_diff': test_metrics['mae'] - val_metrics['mae'],
                'r2_diff': test_metrics['r2'] - val_metrics['r2'],
                'dir_acc_diff': test_metrics['directional_accuracy'] - val_metrics['directional_accuracy']
            }
        }

        print(f"[OK] Enhanced LSTM-HAR trained successfully")
        print(f"  Val RMSE: {val_metrics['rmse']:.6f}")
        print(f"  Test RMSE: {test_metrics['rmse']:.6f}")
        print(f"  RMSE Diff: {test_metrics['rmse'] - val_metrics['rmse']:+.6f}")

    except Exception as e:
        print(f"[ERROR] training Enhanced LSTM-HAR: {e}")
        import traceback
        traceback.print_exc()

    # Save comprehensive results
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS - ALL MODELS")
    print("="*80)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    for model_name, model_results in results.items():
        val_m = model_results['validation_metrics']
        test_m = model_results['test_metrics']
        diff = model_results['val_test_diff']

        print(f"\n{model_name.upper()}:")
        print(f"  Validation RMSE: {val_m['rmse']:.6f}")
        print(f"  Test RMSE:       {test_m['rmse']:.6f}")
        print(f"  RMSE Difference:  {diff['rmse_diff']:+.6f}")
        print(f"  Validation Dir Acc: {val_m['directional_accuracy']:.2f}%")
        print(f"  Test Dir Acc:       {test_m['directional_accuracy']:.2f}%")
        print(f"  Dir Acc Difference: {diff['dir_acc_diff']:+.2f}%")

    # Save to JSON
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, f'all_models_val_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to: {results_file}")
    print("="*80)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total models trained: {len(results)}")
    print(f"Split method: 3-way temporal (70/15/15)")
    print(f"Data leakage: FIXED (no random split)")
    print(f"Validation: ENABLED (early stopping)")
    print(f"Metrics: Val + Test + Comparison")
    print("="*80)

    return results


if __name__ == "__main__":
    data_dir = 'data/processed'
    results = train_all_models_with_validation(data_dir)

    print("\nTRAINING COMPLETE!")
    print("\nNext steps:")
    print("1. Review results in results/all_models_val_results_*.json")
    print("2. Compare val vs test metrics for each model")
    print("3. Check for overfitting (large val/test gaps)")
    print("4. Update documentation with new findings")
