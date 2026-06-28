"""
Test to verify HAR features leakage fix

This test verifies that:
1. HAR features are computed separately for each split
2. No HAR features in training data contain statistics from val/test data
3. Performance drops to realistic range (54-56% Dir Acc)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

print("="*80)
print("HAR FEATURES LEAKAGE FIX VERIFICATION TEST")
print("="*80)

# Test 1: Verify HAR features are computed separately
print("\n[Test 1] Verifying HAR features are computed separately...")
from src.lstm_gat_hybrid.dataset_with_graph_method import (
    _load_raw_stock_data,
    _split_raw_data_by_date,
    _generate_har_for_split
)

# Load raw data
stock_data_raw = _load_raw_stock_data(
    data_dir='data/processed/vn30_only',
    remove_outliers=False
)

# Split raw data
train_raw, val_raw, test_raw, train_end_idx, val_end_idx, min_length = \
    _split_raw_data_by_date(stock_data_raw, 0.7, 0.15, 0.15)

# Generate HAR features separately
train_har = _generate_har_for_split(train_raw, 'train')
val_har = _generate_har_for_split(val_raw, 'val')
test_har = _generate_har_for_split(test_raw, 'test')

# Verify: Check if monthly means are different
print("\n[Test 1] Checking if monthly means are computed separately...")

# Get a sample stock
sample_stock = list(train_har.keys())[0]

# Check last row of training data
train_last_monthly = train_har[sample_stock]['har_monthly_vol'].iloc[-1]
print(f"  Training last monthly mean: {train_last_monthly:.6f}")

# Check first row of validation data
val_first_monthly = val_har[sample_stock]['har_monthly_vol'].iloc[0]
print(f"  Validation first monthly mean: {val_first_monthly:.6f}")

# These should be different because they're computed on different data
if abs(train_last_monthly - val_first_monthly) > 1e-6:
    print("  [PASS] Monthly means are different (computed separately)")
else:
    print("  [FAIL] Monthly means are the same (potential leakage)")

# Test 2: Verify no overlap in date ranges
print("\n[Test 2] Verifying no overlap in date ranges...")

# Get date ranges
if 'date' in train_raw[sample_stock].columns:
    train_dates = set(train_raw[sample_stock]['date'].astype(str))
    val_dates = set(val_raw[sample_stock]['date'].astype(str))
    test_dates = set(test_raw[sample_stock]['date'].astype(str))

    # Check for overlaps
    train_val_overlap = train_dates & val_dates
    val_test_overlap = val_dates & test_dates
    train_test_overlap = train_dates & test_dates

    if len(train_val_overlap) == 0 and len(val_test_overlap) == 0 and len(train_test_overlap) == 0:
        print("  [PASS] No date overlap between splits")
    else:
        print(f"  [FAIL] Found date overlaps:")
        if train_val_overlap:
            print(f"    Train-Val: {len(train_val_overlap)} dates")
        if val_test_overlap:
            print(f"    Val-Test: {len(val_test_overlap)} dates")
        if train_test_overlap:
            print(f"    Train-Test: {len(train_test_overlap)} dates")
else:
    print("  [SKIP] No 'date' column found")

# Test 3: Create dataset using fixed function
print("\n[Test 3] Creating dataset with fixed function...")
from src.lstm_gat_hybrid.dataset_with_graph_method import create_multi_stock_dataloaders_with_graph_method_fixed

try:
    train_loader, val_loader, test_loader, datasets = create_multi_stock_dataloaders_with_graph_method_fixed(
        data_dir='data/processed/vn30_only',
        seq_length=22,
        forecast_horizon=5,
        graph_method='knn',
        k_neighbors=8,
        batch_size=32,
        normalize=True,
        remove_outliers=False,
        data_augmentation=False
    )

    train_dataset, val_dataset, test_dataset = datasets

    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val:   {len(val_dataset)} sequences")
    print(f"  Test:  {len(test_dataset)} sequences")
    print("  [PASS] Dataset created successfully with fixed function")

except Exception as e:
    print(f"  [FAIL] Error creating dataset: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("HAR FEATURES LEAKAGE FIX VERIFICATION COMPLETE")
print("="*80)
print("\nNext step: Run quick_test to verify Dir Acc drops to 54-56%")
print("Command: python src/lstm_gat_hybrid/train_parallel_enhanced.py --quick_test --graph_method knn")
