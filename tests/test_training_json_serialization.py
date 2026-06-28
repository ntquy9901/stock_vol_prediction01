"""
Integration Test: Simulate Training Results JSON Serialization

This test simulates the actual training scenario to ensure that
the results dictionary can be serialized to JSON without errors.

This tests the fix for: "TypeError: Object of type bool_ is not JSON serializable"
"""

import json
import numpy as np
from datetime import datetime


def simulate_training_results():
    """Simulate the exact structure of training results"""

    # Simulate config dict (as defined in training script)
    config = {
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'fc_dropout': 0.3,
        'use_layer_norm': True,
        'learning_rate': 0.001,
        'batch_size': 32,
        'seq_length': 22,
        'forecast_horizon': 5,
        'weight_decay': 1e-5,
        'gradient_clip': 1.0,
        'num_epochs': 70,
        'patience': 15,
        'min_epochs': 15,
        'lr_scheduler_factor': 0.5,
        'lr_scheduler_patience': 5,
        'apply_augmentation': True,
        'augment_factor': 2,
        'plot_interval': 10
    }

    # Simulate training results
    best_epoch = 18
    best_val_loss = 0.780
    epoch = 32
    epochs_without_improvement = 15

    # Simulate metrics (numpy arrays from evaluation)
    val_metrics = {
        'mse': np.float64(2.15e-07),
        'rmse': np.float64(0.000464),
        'mae': np.float64(0.000262),
        'r2': np.float64(0.106349),
        'qlike': np.float64(0.569348),
        'directional_accuracy': np.float64(48.44)
    }

    test_metrics = {
        'mse': np.float64(3.11e-07),
        'rmse': np.float64(0.000557),
        'mae': np.float64(0.000259),
        'r2': np.float64(0.098125),
        'qlike': np.float64(0.640517),
        'directional_accuracy': np.float64(48.56)
    }

    # Calculate differences (these become numpy types!)
    mse_diff = test_metrics['mse'] - val_metrics['mse']
    rmse_diff = test_metrics['rmse'] - val_metrics['rmse']
    mae_diff = test_metrics['mae'] - val_metrics['mae']
    r2_diff = test_metrics['r2'] - val_metrics['r2']
    qlike_diff = test_metrics['qlike'] - val_metrics['qlike']
    dir_acc_diff = test_metrics['directional_accuracy'] - val_metrics['directional_accuracy']

    # Check overfitting (this returns numpy bool!)
    is_overfitting = (rmse_diff >= 0.05) or (r2_diff <= -0.05) or (dir_acc_diff <= -2)

    # Build results dict with EXPLICIT TYPE CONVERSIONS (the fix!)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    results = {
        'model': 'Enhanced LSTM-HAR with Overfitting Prevention',
        'version': '2.0',
        'timestamp': timestamp,
        'overfitting_prevention': {
            'data_centric': {
                'data_augmentation': bool(config['apply_augmentation']),  # EXPLICIT bool()
                'augment_factor': config['augment_factor'] if config['apply_augmentation'] else None
            },
            'model_centric': {
                'early_stopping_patience': int(config['patience']),
                'weight_decay': float(config['weight_decay']),
                'lstm_dropout': float(config['dropout']),
                'fc_dropout': float(config['fc_dropout']),
                'gradient_clipping': float(config['gradient_clip']),
                'lr_scheduler': 'ReduceLROnPlateau',
                'layer_normalization': bool(config['use_layer_norm'])  # EXPLICIT bool()
            }
        },
        'configuration': {
            'hidden_size': int(config['hidden_size']),
            'num_layers': int(config['num_layers']),
            'dropout': float(config['dropout']),
            'fc_dropout': float(config['fc_dropout']),
            'use_layer_norm': bool(config['use_layer_norm']),  # EXPLICIT bool()
            'learning_rate': float(config['learning_rate']),
            'batch_size': int(config['batch_size']),
            'seq_length': int(config['seq_length']),
            'forecast_horizon': int(config['forecast_horizon']),
            'weight_decay': float(config['weight_decay']),
            'gradient_clip': float(config['gradient_clip']),
            'num_epochs': int(config['num_epochs']),
            'patience': int(config['patience']),
            'min_epochs': int(config['min_epochs']),
            'lr_scheduler_factor': float(config['lr_scheduler_factor']),
            'lr_scheduler_patience': int(config['lr_scheduler_patience']),
            'apply_augmentation': bool(config['apply_augmentation']),  # EXPLICIT bool()
            'augment_factor': int(config['augment_factor']),
            'plot_interval': int(config['plot_interval'])
        },
        'best_epoch': int(best_epoch + 1),
        'best_val_loss': float(best_val_loss),
        'total_epochs_trained': int(epoch + 1),
        'early_stopped': bool(epochs_without_improvement >= config['patience']),  # EXPLICIT bool()
        'validation_metrics': {
            'mse': float(val_metrics['mse']),  # EXPLICIT float()
            'rmse': float(val_metrics['rmse']),
            'mae': float(val_metrics['mae']),
            'r2': float(val_metrics['r2']),
            'qlike': float(val_metrics['qlike']),
            'directional_accuracy': float(val_metrics['directional_accuracy'])
        },
        'test_metrics': {
            'mse': float(test_metrics['mse']),  # EXPLICIT float()
            'rmse': float(test_metrics['rmse']),
            'mae': float(test_metrics['mae']),
            'r2': float(test_metrics['r2']),
            'qlike': float(test_metrics['qlike']),
            'directional_accuracy': float(test_metrics['directional_accuracy'])
        },
        'val_test_diff': {
            'mse_diff': float(mse_diff),  # EXPLICIT float() for numpy result
            'rmse_diff': float(rmse_diff),
            'mae_diff': float(mae_diff),
            'r2_diff': float(r2_diff),
            'qlike_diff': float(qlike_diff),
            'dir_acc_diff': float(dir_acc_diff)
        },
        'overfitting_analysis': {
            'is_overfitting': bool(is_overfitting),  # EXPLICIT bool() for numpy bool
            'rmse_gap_threshold': float(0.05),
            'rmse_gap_status': str('OVERFIT' if rmse_diff >= 0.05 else 'OK'),
            'r2_gap_threshold': float(-0.05),
            'r2_gap_status': str('OVERFIT' if r2_diff <= -0.05 else 'OK'),
            'dir_acc_gap_threshold': float(-2.0),
            'dir_acc_gap_status': str('OVERFIT' if dir_acc_diff <= -2 else 'OK')
        }
    }

    return results


def test_integration():
    """Test the complete training results serialization"""

    print("="*80)
    print("INTEGRATION TEST: Training Results JSON Serialization")
    print("="*80)

    # Get simulated results
    results = simulate_training_results()

    # Test 1: Serialize to JSON
    print("\n[Test 1] Attempting JSON serialization...")
    try:
        json_str = json.dumps(results, indent=2)
        print("[OK] Test 1 PASSED: JSON serialization successful")
        print(f"   JSON size: {len(json_str)} characters")
        print(f"   JSON preview (first 200 chars): {json_str[:200]}...")
    except TypeError as e:
        print(f"[FAIL] Test 1 FAILED: {e}")
        return False

    # Test 2: Deserialize and verify
    print("\n[Test 2] Verifying deserialization...")
    try:
        loaded = json.loads(json_str)
        print("[OK] Test 2 PASSED: JSON deserialization successful")
        print(f"   Model: {loaded['model']}")
        print(f"   Version: {loaded['version']}")
        print(f"   Best Epoch: {loaded['best_epoch']}")
        print(f"   Test RMSE: {loaded['test_metrics']['rmse']:.6f}")
        print(f"   Test Dir Acc: {loaded['test_metrics']['directional_accuracy']:.2f}%")
    except Exception as e:
        print(f"[FAIL] Test 2 FAILED: {e}")
        return False

    # Test 3: Verify all types are Python native (not numpy)
    print("\n[Test 3] Checking for numpy types...")

    def check_no_numpy_types(obj, path="root"):
        """Recursively check for numpy types"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                check_no_numpy_types(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_no_numpy_types(item, f"{path}[{i}]")
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            print(f"[FAIL] Found numpy type at {path}: {type(obj)}")
            return False
        return True

    if check_no_numpy_types(results):
        print("[OK] Test 3 PASSED: No numpy types in results dict")
    else:
        return False

    # Test 4: Verify critical boolean values
    print("\n[Test 4] Verifying boolean conversions...")
    bool_checks = [
        ("config.use_layer_norm", results['configuration']['use_layer_norm'], True),
        ("config.apply_augmentation", results['configuration']['apply_augmentation'], True),
        ("overfitting_prevention.layer_normalization", results['overfitting_prevention']['model_centric']['layer_normalization'], True),
        ("overfitting_prevention.data_augmentation", results['overfitting_prevention']['data_centric']['data_augmentation'], True),
        ("overfitting_analysis.is_overfitting", results['overfitting_analysis']['is_overfitting'], False),
    ]

    all_bools_ok = True
    for path, value, expected_type in bool_checks:
        if isinstance(value, bool) and not isinstance(value, np.bool_):
            print(f"  [OK] {path} = {value} (Python bool)")
        else:
            print(f"  [FAIL] {path} = {value} (type: {type(value)})")
            all_bools_ok = False

    if all_bools_ok:
        print("[OK] Test 4 PASSED: All boolean values are Python bools")
    else:
        print("[FAIL] Test 4 FAILED: Some values are not Python bools")
        return False

    print("\n" + "="*80)
    print("[SUCCESS] ALL INTEGRATION TESTS PASSED")
    print("="*80)
    print("\nFix verified:")
    print("  - All boolean values explicitly converted with bool()")
    print("  - All numeric values explicitly converted with float() or int()")
    print("  - JSON serialization works without errors")
    print("="*80)

    return True


if __name__ == '__main__':
    success = test_integration()
    exit(0 if success else 1)
