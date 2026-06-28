"""
Check training metrics from saved model

This script loads the best model and evaluates on test set
to verify metrics are correct.
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from lstm_gat_hybrid.model_parallel import create_parallel_lstm_gat_model
from lstm_gat_hybrid.config import LSTMGATConfig
from lstm_gat_hybrid.dataset_with_graph_method import create_multi_stock_dataloaders_with_graph_method
from common.evaluation import evaluate_predictions


def check_metrics(results_dir):
    """
    Check metrics from training results

    Args:
        results_dir: Directory containing training results
    """
    results_path = Path(results_dir)

    # Load training results
    results_file = results_path / 'training_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)

        print("=" * 80)
        print("TRAINING RESULTS FROM JSON")
        print("=" * 80)
        print(f"Epochs trained: {results['training_summary']['num_epochs_trained']}")
        print(f"Best epoch: {results['training_summary']['best_epoch']}")
        print(f"Best val loss: {results['training_summary']['best_val_loss']:.6f}")

        print("\nTest Metrics:")
        test_metrics = results['test_metrics']
        print(f"  MSE: {test_metrics['mse']:.8f}")
        print(f"  RMSE: {test_metrics['rmse']:.6f}")
        print(f"  MAE: {test_metrics['mae']:.6f}")
        print(f"  R²: {test_metrics['r2']:.6f}")
        print(f"  QLIKE: {test_metrics['qlike']:.6f}")
        print(f"  Dir Acc: {test_metrics['directional_accuracy']:.2f}%")

        # Check if metrics are reasonable
        print("\n" + "=" * 80)
        print("METRICS VALIDATION")
        print("=" * 80)

        # Check RMSE (should be 0.001 - 0.01 for volatility)
        if test_metrics['rmse'] < 0.01:
            print(f"[OK] RMSE {test_metrics['rmse']:.6f} is in PHYSICAL scale (good!)")
        else:
            print(f"[ERROR] RMSE {test_metrics['rmse']:.6f} is too high - possible normalized scale issue")

        # Check QLIKE (should be 0.5 - 1.0 for volatility)
        if 0.5 < test_metrics['qlike'] < 1.0:
            print(f"[OK] QLIKE {test_metrics['qlike']:.6f} is reasonable")
        else:
            print(f"[ERROR] QLIKE {test_metrics['qlike']:.6f} is outside normal range")

        # Check Dir Acc (should be >50%, ideally >65%)
        if test_metrics['directional_accuracy'] > 50:
            print(f"[OK] Dir Acc {test_metrics['directional_accuracy']:.2f}% is better than random")
        else:
            print(f"[WARNING] Dir Acc {test_metrics['directional_accuracy']:.2f}% is worse than random")

        # Check R² (should be positive)
        if test_metrics['r2'] > 0:
            print(f"[OK] R² {test_metrics['r2']:.6f} is positive (model explains variance)")
        else:
            print(f"[ERROR] R² {test_metrics['r2']:.6f} is negative (model worse than mean)")

        print("\n" + "=" * 80)
        print("COMPARISON WITH BASELINES")
        print("=" * 80)
        print(f"LSTM-HAR Enhanced (document): 67.90% Dir Acc")
        print(f"This model: {test_metrics['directional_accuracy']:.2f}% Dir Acc")

        if test_metrics['directional_accuracy'] > 67.90:
            print(f"[SUCCESS] Beats LSTM-HAR Enhanced by {test_metrics['directional_accuracy'] - 67.90:.2f}%")
        else:
            print(f"[INFO] Gap to LSTM-HAR Enhanced: {67.90 - test_metrics['directional_accuracy']:.2f}%")

    else:
        print(f"[ERROR] No training_results.json found in {results_dir}")
        print("Training might not be complete yet")

        # Check if model exists
        model_file = results_path / 'best_parallel_model.pth'
        if model_file.exists():
            print(f"\n[OK] Found best_parallel_model.pth - running evaluation...")

            # Load model
            config = LSTMGATConfig()
            config.num_stocks = 30

            model = create_parallel_lstm_gat_model(config)
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
            model.eval()

            print(f"[OK] Model loaded successfully")

            # Create test dataloader
            print("\nCreating test dataloader...")
            _, _, test_loader, datasets = create_multi_stock_dataloaders_with_graph_method(
                data_dir='data/processed',
                seq_length=22,
                forecast_horizon=5,
                graph_method='knn',
                batch_size=11,
                num_workers=0
            )

            # Evaluate on test set
            print("\nEvaluating on test set...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for x, adj_matrix, y, _ in test_loader:
                    x = x.to(device)
                    adj_matrix = adj_matrix.to(device)
                    y = y.to(device)

                    predictions = model(x, adj_matrix)
                    predictions_flat = predictions.reshape(y.shape[0] * y.shape[1])
                    y_flat = y.reshape(y.shape[0] * y.shape[1])

                    all_predictions.extend(predictions_flat.cpu().numpy())
                    all_targets.extend(y_flat.cpu().numpy())

            # Extract actual dataset (handle Subset wrapper)
            actual_dataset = datasets[2]
            if hasattr(datasets[2], 'dataset'):
                actual_dataset = datasets[2].dataset
                print(f"[DEBUG] Extracted original dataset from Subset")

            # Inverse transform
            if hasattr(actual_dataset, 'target_normalizers'):
                print(f"[DEBUG] Applying inverse transform...")

                all_predictions = np.array(all_predictions).flatten()
                all_targets = np.array(all_targets).flatten()

                all_predictions_denorm = np.zeros_like(all_predictions)
                all_targets_denorm = np.zeros_like(all_targets)

                for i in range(len(all_predictions)):
                    stock_idx = i % len(actual_dataset.stock_names)
                    stock_name = actual_dataset.stock_names[stock_idx]

                    if stock_name in actual_dataset.target_normalizers:
                        all_predictions_denorm[i] = \
                            actual_dataset.target_normalizers[stock_name].inverse_transform(
                                all_predictions[i:i+1].reshape(1, -1)
                            ).flatten()[0]
                        all_targets_denorm[i] = \
                            actual_dataset.target_normalizers[stock_name].inverse_transform(
                                all_targets[i:i+1].reshape(1, -1)
                            ).flatten()[0]
                    else:
                        all_predictions_denorm[i] = all_predictions[i]
                        all_targets_denorm[i] = all_targets[i]

                all_predictions = all_predictions_denorm.flatten()
                all_targets = all_targets_denorm.flatten()

                print(f"[DEBUG] Prediction range: [{all_predictions.min():.6f}, {all_predictions.max():.6f}]")
                print(f"[DEBUG] Target range: [{all_targets.min():.6f}, {all_targets.max():.6f}]")
            else:
                print(f"[WARNING] No inverse transform applied!")

            # Calculate metrics
            metrics = evaluate_predictions(all_targets, all_predictions)

            print("\n" + "=" * 80)
            print("EVALUATION RESULTS")
            print("=" * 80)
            print(f"Test MSE: {metrics['mse']:.8f}")
            print(f"Test RMSE: {metrics['rmse']:.6f}")
            print(f"Test MAE: {metrics['mae']:.6f}")
            print(f"Test R²: {metrics['r2']:.6f}")
            print(f"Test QLIKE: {metrics['qlike']:.6f}")
            print(f"Test Dir Acc: {metrics['directional_accuracy']:.2f}%")
        else:
            print(f"[ERROR] No model file found in {results_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Check training metrics')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Results directory to check')

    args = parser.parse_args()

    check_metrics(args.results_dir)
