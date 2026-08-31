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
from sklearn.linear_model import LinearRegression
from datetime import datetime
import joblib
import json

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.common.feature_engineering import create_har_features, create_5day_target
from src.common.evaluation import evaluate_predictions


FEATURE_COLS = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']


def load_har_train_test_split(data_dir: str, train_ratio: float = 0.8):
    """
    Load processed CSVs and build a PER-TICKER chronological train/test split.

    For each ticker: sort by date, compute HAR features + the 5-day-ahead target
    (both causal), drop warm-up/trailing NaN rows, then cut at ``train_ratio`` of
    THAT ticker's own valid rows. The early portion of every ticker goes to train
    and the late portion of every ticker goes to test.

    This replaces the earlier single global cut over a ticker-concatenated array,
    which -- because tickers were concatenated whole in arbitrary os.listdir order
    -- put some tickers entirely in train and others entirely in test instead of
    performing a temporal split (CLAUDE.md Section 3.A). The pattern here mirrors
    HARVolatilityDataset._load_all_data (src/lstm_har_baseline/dataset.py).

    The rows stay ticker-blocked (one ticker's contiguous block after another),
    so within each block consecutive array positions are consecutive trading days
    for that ticker; evaluate_predictions()'s np.diff-based directional accuracy is
    therefore already correct without an n_stocks argument (only the few
    ticker-boundary transitions produce a spurious diff, negligible at this data
    volume).

    Args:
        data_dir: Directory containing ``*_processed.csv`` files.
        train_ratio: Fraction of each ticker's valid rows used for training.

    Returns:
        Tuple ``(X_train, y_train, X_test, y_test, train_meta, test_meta)`` where
        the ``*_meta`` DataFrames carry ``ticker`` and ``date`` per row.
    """
    train_parts, test_parts = [], []

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('_processed.csv'):
            continue

        ticker = filename.replace('_processed.csv', '')
        df = pd.read_csv(os.path.join(data_dir, filename))

        # Sort chronologically (per-ticker) before any feature/target creation.
        df = df.sort_values('date').reset_index(drop=True)

        vol = df['parkinson_variance']
        har_features = create_har_features(vol)
        df = pd.concat([df, har_features], axis=1)
        df['target_5d'] = create_5day_target(vol)
        df['ticker'] = ticker

        # Keep valid rows (no NaN in features or target).
        df_valid = df.dropna(subset=FEATURE_COLS + ['target_5d']).copy()
        if len(df_valid) == 0:
            continue

        # Per-ticker chronological cut point.
        split_idx = int(train_ratio * len(df_valid))
        train_parts.append(df_valid.iloc[:split_idx])
        test_parts.append(df_valid.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['target_5d'].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df['target_5d'].values
    train_meta = train_df[['ticker', 'date']].reset_index(drop=True)
    test_meta = test_df[['ticker', 'date']].reset_index(drop=True)

    return X_train, y_train, X_test, y_test, train_meta, test_meta


def train_har_baseline(data_dir: str, output_dir: str = None):
    """
    Train HAR-R baseline on pooled dataset using Parkinson volatility.

    Args:
        data_dir: Directory containing processed CSV files with parkinson_variance
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

    # Load processed data and build a per-ticker chronological train/test split.
    # (See load_har_train_test_split for why a single global cut was wrong.)
    print("\n1-3. Loading data, creating HAR features, and splitting per ticker...")
    feature_cols = FEATURE_COLS
    X_train, y_train, X_test, y_test, train_meta, test_meta = (
        load_har_train_test_split(data_dir)
    )

    n_tickers = train_meta['ticker'].nunique()
    print(f"  Tickers: {n_tickers}")
    print(f"  Total samples: {len(X_train) + len(X_test)}")
    print(f"  Feature shape (train): {X_train.shape}")

    print("\n  Data Statistics (train):")
    print(f"    X mean: {X_train.mean():.6f}, std: {X_train.std():.6f}")
    print(f"    y mean: {y_train.mean():.6f}, std: {y_train.std():.6f}")

    print("\n  Per-ticker chronological split (80/20 of each ticker's own rows):")
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
