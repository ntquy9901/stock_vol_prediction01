"""
Process VN100 Raw Data → Add HAR Features → Ready for Training

Input: data/raw/vn100_enhanced/*.csv (100 stocks)
Output: data/processed/vn100_only/*.csv (with HAR features)

Steps:
1. Load raw CSV files
2. Generate HAR features (daily, weekly, monthly volatility)
3. Remove outliers (optional)
4. Save to processed directory
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

from src.common.har_features import generate_har_features

print("="*80)
print("PROCESSING VN100 DATA")
print("="*80)

# Configuration
INPUT_DIR = Path("data/raw/vn100_enhanced")
OUTPUT_DIR = Path("data/processed/vn100_only")
REMOVE_OUTLIERS = True
N_STD = 3.0

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n[Configuration]")
print(f"  Input:  {INPUT_DIR}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Remove outliers: {REMOVE_OUTLIERS} (n_std={N_STD})")

# Get all CSV files
csv_files = sorted(INPUT_DIR.glob('*.csv'))
print(f"\n[Processing] Found {len(csv_files)} CSV files")

if len(csv_files) == 0:
    print(f"\n[ERROR] No CSV files found in {INPUT_DIR}")
    sys.exit(1)

# Process each stock
processed_count = 0
skipped_count = 0
total_outliers_removed = 0

for csv_file in csv_files:
    stock_name = csv_file.stem  # Remove .csv extension

    try:
        # Load raw data
        df = pd.read_csv(csv_file)

        # Validate required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  [SKIP] {stock_name}: Missing columns {missing_cols}")
            skipped_count += 1
            continue

        # Check if already has parkinson_volatility
        if 'parkinson_volatility' in df.columns:
            print(f"  [INFO] {stock_name}: Already has parkinson_volatility")
        else:
            # Calculate Parkinson volatility
            df['hl_pct'] = np.log(df['high'] / df['low'])
            df['parkinson_volatility'] = (df['hl_pct'] ** 2) / (4 * np.log(2))

            # Add date column back if it was renamed
            df['ohlcv_date'] = df['date']

        # Remove outliers
        if REMOVE_OUTLIERS:
            original_len = len(df)

            # Calculate z-scores
            vol_values = df['parkinson_volatility'].values
            if np.std(vol_values) > 0:
                z_scores = np.abs((vol_values - np.mean(vol_values)) / np.std(vol_values))
                outlier_mask = z_scores < N_STD
                df_clean = df[outlier_mask].copy()
            else:
                df_clean = df.copy()

            outliers_removed = original_len - len(df_clean)
            total_outliers_removed += outliers_removed

            if len(df_clean) < 30:
                print(f"  [SKIP] {stock_name}: Insufficient data after outlier removal ({len(df_clean)} rows)")
                skipped_count += 1
                continue

            df = df_clean

        # Generate HAR features
        try:
            df_har = generate_har_features(df)
            print(f"  [OK] {stock_name}: {len(df)} rows, {outliers_removed if REMOVE_OUTLIERS else 0} outliers removed")
        except Exception as e:
            print(f"  [SKIP] {stock_name}: HAR generation failed - {e}")
            skipped_count += 1
            continue

        # Save processed data
        output_file = OUTPUT_DIR / f"{stock_name}_processed.csv"
        df_har.to_csv(output_file, index=False)
        processed_count += 1

    except Exception as e:
        print(f"  [ERROR] {stock_name}: {e}")
        skipped_count += 1
        continue

print(f"\n{'='*80}")
print(f"PROCESSING COMPLETE")
print(f"{'='*80}")
print(f"  Successfully processed: {processed_count} stocks")
print(f"  Skipped: {skipped_count} stocks")
print(f"  Total outliers removed: {total_outliers_removed}")
print(f"  Output directory: {OUTPUT_DIR}")

if processed_count < 20:
    print(f"\n[WARNING] Only {processed_count} stocks processed - need at least 20 for training")

print(f"\n[Next Step]")
print(f"  Run training:")
print(f"    python src/lstm_gat_hybrid/train_parallel_enhanced.py --graph_method knn")
print(f"    (Need to update script to use data/processed/vn100_only)")
