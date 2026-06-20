"""
HAR-R Baseline Training Script

This module trains HAR-R (Heterogeneous Autoregressive) baseline model
for 5-day ahead volatility forecasting using linear regression.

Uses shared utilities from src/common:
    - parkinson_utils: Parkinson volatility calculation
    - feature_engineering: HAR feature creation
    - evaluation: Model metrics calculation

Model Specification:
    target_5d = β₀ + β₁*har_daily_vol + β₂*har_weekly_vol + β₃*har_monthly_vol

Data Flow:
    Raw OHLCV → Parkinson Volatility → HAR Features → Linear Regression → Predictions

Author: Stock Volatility Prediction Team
Date: 2026-06-18
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import joblib
import json

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.common.parkinson_utils import calculate_parkinson_volatility
from src.common.feature_engineering import create_har_features, create_5day_target
from src.common.evaluation import evaluate_predictions


def train_har_baseline(data_dir: str, output_dir: str = None):
    """
    Train HAR-R baseline on pooled dataset using Parkinson volatility.

    Args:
        data_dir: Directory containing processed CSV files with parkinson_volatility
        output_dir: Output directory (default: results/har_baseline_YYYY-MM-DD_HHMMSS)
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = f'results/har_baseline_{timestamp}'

    print("=" * 80)
    print("HAR-R BASELINE TRAINING")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Load processed data (already contains parkinson_volatility)
    print("\n1. Loading processed data (Parkinson volatility)...")
    all_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith('_processed.csv'):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)
            all_files.append(df)

    print(f"  Loaded {len(all_files)} stock files")

    # Create HAR features for all stocks
    print("\n2. Creating HAR features using src/common utilities...")
    features_list = []

    for df in all_files:
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Extract parkinson volatility (already calculated from raw OHLCV)
        vol = df['parkinson_volatility']

        # Create HAR features using src/common/feature_engineering
        har_features = create_har_features(vol)
        df = pd.concat([df, har_features], axis=1)

        # Create 5-day target using src/common/feature_engineering
        target = create_5day_target(vol)
        df['target_5d'] = target

        # Keep valid rows (no NaN in features or target)
        df_valid = df.dropna(subset=['har_daily_vol', 'har_weekly_vol',
                                      'har_monthly_vol', 'target_5d']).copy()

        if len(df_valid) > 0:
            features_list.append(df_valid)

    # Combine all stocks into pooled dataset
    all_features = pd.concat(features_list, ignore_index=True)
    print(f"  Total samples: {len(all_features)}")

    # Prepare features and target
    feature_cols = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']
    X = all_features[feature_cols].values
    y = all_features['target_5d'].values

    print(f"  Feature shape: {X.shape}")
    print(f"  Target shape: {y.shape}")

    # Display sample statistics
    print(f"\n  Data Statistics:")
    print(f"    X mean: {X.mean():.6f}, std: {X.std():.6f}")
    print(f"    y mean: {y.mean():.6f}, std: {y.std():.6f}")

    # Temporal train/test split (80/20) - CHRONOLOGICAL
    print("\n3. Splitting data chronologically (temporal split)...")
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Train size: {len(X_train)}")
    print(f"  Test size: {len(X_test)}")

    # Train HAR-R Linear Regression model
    print("\n4. Training HAR-R Linear Regression...")
    model = LinearRegression()

    import time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"  Training time: {train_time:.3f} seconds")
    print(f"  Coefficients: {model.coef_}")
    print(f"  Intercept: {model.intercept_:.6f}")

    # Feature importance (coefficients)
    print("\n5. Feature Importance:")
    for i, col in enumerate(feature_cols):
        print(f"  {col:20s}: {model.coef_[i]:10.6f}")

    # Make predictions on test set
    print("\n6. Evaluating on test set...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics using src/common/evaluation
    print("\n7. Test Results:")
    print("-" * 80)

    metrics = evaluate_predictions(y_test, y_pred_test)

    for metric_name, value in metrics.items():
        if metric_name == 'Directional_Acc':
            print(f"{metric_name}: {value:.2f}%")
        else:
            print(f"{metric_name}: {value:.6f}")

    # Save results to CSV
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(os.path.join(output_dir, 'test_metrics.csv'), index=False)

    # Save trained model
    joblib.dump(model, os.path.join(output_dir, 'har_baseline_model.pkl'))
    print(f"\nModel saved to: {output_dir}/har_baseline_model.pkl")

    # Save training info (model coefficients, etc.)
    info = {
        'model_type': 'HAR-R Linear Regression',
        'features': feature_cols,
        'target': 'target_5d',
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'training_time_seconds': float(train_time),
        'coefficients': dict(zip(feature_cols, model.coef_.tolist())),
        'intercept': float(model.intercept_),
        'data_source': 'Parkinson volatility from processed CSV files'
    }

    with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 80)
    print("HAR-R Baseline Training Complete!")
    print(f"Total time: {train_time:.3f} seconds")
    print(f"Results saved to: {output_dir}/")
    print("=" * 80)

    return model, metrics


if __name__ == "__main__":
    """Main execution - Can be run as module or directly."""
    print("\n" + "=" * 80)
    print("HAR-R BASELINE - LINEAR REGRESSION WITH PARKINSON VOLATILITY")
    print("=" * 80)

    # Check if processed data exists
    data_dir = os.path.join(project_root, 'data/processed')

    if not os.path.exists(data_dir):
        print(f"[ERROR] Processed data directory not found: {data_dir}")
        print("\nPlease process raw OHLCV data first:")
        print("  python -m src.common.process_parkinson_pipeline")
        sys.exit(1)

    # Train HAR baseline
    model, metrics = train_har_baseline(data_dir)

    print("\n[SUCCESS] HAR-R baseline training completed successfully!")

    # Show how to run from command line
    print("\n" + "=" * 80)
    print("USAGE:")
    print("  From project root: python -m src.har_baseline.train")
    print("  From this directory: python train.py")
    print("=" * 80)
