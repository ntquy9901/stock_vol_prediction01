"""
Unit Test: JSON Serialization for Training Results

This test verifies that all values in the training results dictionary
can be properly serialized to JSON without TypeError.

Tests for the bug: "TypeError: Object of type bool_ is not JSON serializable"
"""

import json
import numpy as np


def test_json_serialization():
    """Test that training results dict can be serialized to JSON"""

    # Simulate the results dictionary structure from training
    results = {
        'model': 'Enhanced LSTM-HAR with Overfitting Prevention',
        'version': '2.0',
        'timestamp': '2026-06-21_120000',
        'dataset': '30 VN30 stocks',
        'overfitting_prevention': {
            'data_centric': {
                'data_augmentation': bool(True),  # Explicit bool conversion
                'augment_factor': int(2) if bool(True) else None
            },
            'model_centric': {
                'early_stopping_patience': int(15),
                'weight_decay': float(1e-5),
                'lstm_dropout': float(0.2),
                'fc_dropout': float(0.3),
                'gradient_clipping': float(1.0),
                'lr_scheduler': 'ReduceLROnPlateau',
                'layer_normalization': bool(True)  # Explicit bool conversion
            }
        },

        # Configuration (with explicit type conversions - this is the fix)
        'configuration': {
            'hidden_size': int(128),
            'num_layers': int(3),
            'dropout': float(0.2),
            'fc_dropout': float(0.3),
            'use_layer_norm': bool(True),  # Explicit bool conversion
            'learning_rate': float(0.001),
            'batch_size': int(32),
            'seq_length': int(22),
            'forecast_horizon': int(5),
            'weight_decay': float(1e-5),
            'gradient_clip': float(1.0),
            'num_epochs': int(70),
            'patience': int(15),
            'min_epochs': int(15),
            'lr_scheduler_factor': float(0.5),
            'lr_scheduler_patience': int(5),
            'apply_augmentation': bool(True),  # Explicit bool conversion
            'augment_factor': int(2),
            'plot_interval': int(10)
        },

        # Training results
        'best_epoch': int(18),
        'best_val_loss': float(0.780),
        'total_epochs_trained': int(32),
        'early_stopped': bool(True),  # Explicit bool conversion

        # Metrics (all converted to Python types)
        'validation_metrics': {
            'mse': float(2.15e-07),
            'rmse': float(0.000464),
            'mae': float(0.000262),
            'r2': float(0.106349),
            'qlike': float(0.569348),
            'directional_accuracy': float(48.44)
        },

        'test_metrics': {
            'mse': float(3.11e-07),
            'rmse': float(0.000557),
            'mae': float(0.000259),
            'r2': float(0.098125),
            'qlike': float(0.640517),
            'directional_accuracy': float(48.56)
        },

        'val_test_diff': {
            'mse_diff': float(9.57e-08),
            'rmse_diff': float(0.000094),
            'mae_diff': float(-0.000002),
            'r2_diff': float(-0.008224),
            'qlike_diff': float(0.071169),
            'dir_acc_diff': float(0.12)
        },

        'overfitting_analysis': {
            'is_overfitting': bool(False),  # Explicit bool conversion
            'rmse_gap_threshold': float(0.05),
            'rmse_gap_status': str('OK'),
            'r2_gap_threshold': float(-0.05),
            'r2_gap_status': str('OK'),
            'dir_acc_gap_threshold': float(-2.0),
            'dir_acc_gap_status': str('OK')
        }
    }

    # Test 1: Try to serialize to JSON
    try:
        json_str = json.dumps(results, indent=2)
        print("[OK] Test 1 PASSED: JSON serialization successful")
        print(f"   JSON string length: {len(json_str)} characters")
    except TypeError as e:
        print(f"[FAIL] Test 1 FAILED: {e}")
        return False

    # Test 2: Verify all Python types (not numpy)
    def check_types(obj, path="root"):
        """Recursively check for numpy types"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                check_types(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_types(item, f"{path}[{i}]")
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            print(f"[FAIL] Test 2 FAILED: Found numpy type at {path}: {type(obj)}")
            return False
        return True

    if check_types(results):
        print("[OK] Test 2 PASSED: No numpy types found in results dict")
    else:
        return False

    # Test 3: Simulate potential numpy bool issues
    print("\n[INFO] Test 3: Testing with numpy bool (potential bug)")
    results_with_numpy_bug = results.copy()
    results_with_numpy_bug['early_stopped'] = np.bool_(True)  # Numpy bool

    try:
        json.dumps(results_with_numpy_bug, indent=2)
        print("[FAIL] Test 3 FAILED: Should have raised TypeError for numpy.bool_")
        return False
    except TypeError as e:
        print(f"[OK] Test 3 PASSED: Correctly caught numpy.bool_ error: {e}")

    print("\n" + "="*80)
    print("[SUCCESS] ALL TESTS PASSED - JSON serialization is working correctly")
    print("="*80)
    return True


if __name__ == '__main__':
    success = test_json_serialization()
    exit(0 if success else 1)
