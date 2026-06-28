"""
Evaluate TimesNet checkpoint and compare with baselines
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from src.timesnet_baseline.model import create_timesnet_model
from src.timesnet_baseline.config import TimesNetConfig
from src.timesnet_baseline.dataset import create_timesnet_dataloaders
from src.common.evaluation import evaluate_predictions

def evaluate_checkpoint(checkpoint_path):
    """Evaluate a TimesNet checkpoint"""

    print("="*80)
    print("TIMESNET CHECKPOINT EVALUATION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    config = TimesNetConfig()
    device = torch.device('cpu')  # Use CPU for evaluation

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, datasets = create_timesnet_dataloaders(
        data_dir='data/processed',
        seq_length=config.seq_len,
        forecast_horizon=config.forecast_horizon,
        batch_size=config.batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        num_workers=0,
        normalize=True
    )

    train_dataset, val_dataset, test_dataset = datasets

    # Create model
    print("Creating model...")
    model = create_timesnet_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Evaluate on test set
    print("\nEvaluating on test set...")

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_har, x_temporal, y in test_loader:
            x_har = x_har.to(device)
            x_temporal = x_temporal.to(device)

            predictions = model(x_har, x_temporal)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y.numpy())

    # Convert to numpy and denormalize
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Denormalize
    if hasattr(test_dataset, 'dataset'):
        denorm_func = test_dataset.dataset.denormalize_predictions
    else:
        denorm_func = test_dataset.denormalize_predictions

    all_predictions_denorm = denorm_func(all_predictions)
    all_targets_denorm = denorm_func(all_targets)

    # Calculate 6 mandatory metrics
    metrics = evaluate_predictions(all_targets_denorm, all_predictions_denorm)

    # Print results
    print("\n" + "="*80)
    print("TIMESNET TEST RESULTS")
    print("="*80)

    print(f"\nTest Metrics:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
    print(f"  QLIKE: {metrics['qlike']:.6f}")
    print(f"  Dir Acc: {metrics['directional_accuracy']:.2f}%")

    # Compare with LSTM-HAR Enhanced
    print(f"\n{'='*80}")
    print("COMPARISON WITH BASELINES")
    print(f"{'='*80}")

    baselines = {
        'LSTM Baseline': 48.32,
        'HAR-R Linear': 51.53,
        'LSTM-HAR Enhanced': 67.90
    }

    print(f"\n{'Model':<20} {'Dir Acc':>12} {'vs TimesNet':>15}")
    print("-" * 50)

    for model_name, acc in baselines.items():
        diff = metrics['directional_accuracy'] - acc
        comparison = f"{diff:+.2f}%"
        print(f"{model_name:<20} {acc:>10.2f}% {comparison:>15}")

    print(f"{'TimesNet':<20} {metrics['directional_accuracy']:>10.2f}% {'(current)':>15}")

    # Evaluate model quality
    print(f"\n{'='*80}")
    print("MODEL EVALUATION")
    print(f"{'='*80}")

    dir_acc = metrics['directional_accuracy']

    if dir_acc >= 67.90:
        print(f"\n[SUCCESS] TimesNet beats LSTM-HAR Enhanced!")
        print(f"   Improvement: {dir_acc - 67.90:.2f}%")
        print(f"   Recommendation: Proceed to hyperparameter tuning")
    elif dir_acc >= 65.0:
        print(f"\n[GOOD] Competitive with LSTM-HAR Enhanced")
        print(f"   Gap: {67.90 - dir_acc:.2f}%")
        print(f"   Recommendation: Try feature engineering or ensemble")
    elif dir_acc >= 55.0:
        print(f"\n[MODERATE] Better than random but below target")
        print(f"   Gap to LSTM-HAR: {67.90 - dir_acc:.2f}%")
        print(f"   Recommendation: Investigate hyperparameters or data issues")
    else:
        print(f"\n[POOR] Below acceptable threshold")
        print(f"   Dir Acc: {dir_acc:.2f}% (worse than random guessing: 50%)")
        print(f"   Recommendation: Debug model architecture or training process")

    # Save results
    results = {
        'model': 'TimesNet Checkpoint Evaluation',
        'checkpoint_path': str(checkpoint_path),
        'evaluated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_metrics': {
            'mse': float(metrics['mse']),
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae']),
            'r2': float(metrics['r2']),
            'qlike': float(metrics['qlike']),
            'directional_accuracy': float(metrics['directional_accuracy'])
        },
        'comparison_with_baselines': {
            'lstm_baseline_diff': float(dir_acc - 48.32),
            'har_r_diff': float(dir_acc - 51.53),
            'lstm_har_enhanced_diff': float(dir_acc - 67.90)
        }
    }

    results_path = checkpoint_path.parent / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return metrics


if __name__ == '__main__':
    # Find most recent checkpoint
    checkpoint_dirs = sorted(Path('results').glob('timesnet_baseline_*'), key=lambda x: x.name)

    if not checkpoint_dirs:
        print("No TimesNet checkpoints found!")
        exit(1)

    # Find the most recent checkpoint with model file
    for checkpoint_dir in reversed(checkpoint_dirs):
        checkpoint_file = checkpoint_dir / 'best_timesnet_model.pth'
        if checkpoint_file.exists():
            print(f"Found checkpoint: {checkpoint_file}")
            metrics = evaluate_checkpoint(checkpoint_file)
            break
    else:
        print("No valid checkpoint files found!")
        exit(1)
