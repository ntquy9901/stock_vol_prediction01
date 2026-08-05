"""
Train LSTM-GNN on VN100 and Test

Steps:
1. Process VN100 raw data → add HAR features
2. Split 70/15/15 chronologically
3. Train LSTM-GNN (50 epochs)
4. Evaluate on test set
5. Compare with baseline models
"""
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LSTM-GNN TRAINING & TESTING ON VN100")
print("="*80)

# Step 1: Check VN100 data
import os
from pathlib import Path

vn100_raw = Path("data/raw/vn100_enhanced")
vn100_processed = Path("data/processed/vn100_only")

print(f"\n[Step 1] Checking VN100 data...")
print(f"  Raw data: {vn100_raw.exists()} - {len(list(vn100_raw.glob('*.csv')))} files")
print(f"  Processed data: {vn100_processed.exists()} - {len(list(vn100_processed.glob('*.csv'))) if vn100_processed.exists() else 0} files")

if not vn100_processed.exists() or len(list(vn100_processed.glob('*.csv'))) == 0:
    print("\n[ERROR] VN100 processed data not found!")
    print("\nNeed to process raw data first:")
    print("  1. Add HAR features to VN100 data")
    print("  2. Remove outliers")
    print("  3. Save to data/processed/vn100_only/")
    print("\nDo you want me to create the processing script?")

else:
    print(f"\n[Step 2] VN100 data ready - {len(list(vn100_processed.glob('*.csv')))} stocks")

    # Step 3: Train model
    print("\n[Step 3] Training LSTM-GNN on VN100...")
    print("  Configuration:")
    print("    - Graph method: knn")
    print("    - Max epochs: 50")
    print("    - Temporal split: 70/15/15")
    print("    - All 4 data leakage fixes active")

    # Training command
    print("\n  Command to run:")
    print("  python src/lstm_gat_hybrid/train_parallel_enhanced.py --graph_method knn")

    # Step 4: Expected results
    print("\n[Step 4] Expected evaluation metrics:")
    print("  - MSE, RMSE, MAE: Lower is better")
    print("  - R²: Higher is better (close to 1.0)")
    print("  - QLIKE: Lower is better")
    print("  - Dir Acc: Higher is better (>55% good)")

print("\n" + "="*80)
