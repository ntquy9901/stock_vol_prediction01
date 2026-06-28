# HAR Features Data Leakage Fix

**Date:** 2026-06-22
**Status:** CRITICAL BUG FIX
**Impact:** +3-5% Dir Acc inflation

---

## Problem Identified

The current implementation generates HAR features on the FULL dataset before temporal split, causing rolling window means to leak future information into training sequences.

### Current (WRONG) Flow:

```python
# Step 1: Load all stock data
stock_data = load_all_data()  # 100% of data

# Step 2: Generate HAR features on 100% of data
stock_data_with_har = generate_har_features(stock_data)
# har_monthly_vol at row i uses data from rows [i-22:i]

# Step 3: Create sequences from pre-computed HAR features
sequences = create_sequences(stock_data_with_har)

# Step 4: Temporal split of sequences (70/15/15)
train_sequences = sequences[:890]    # Uses HAR from [0:912]
val_sequences = sequences[890:1059]   # Uses HAR from [890:1081]
test_sequences = sequences[1059:]     # Uses HAR from [1059:1246]
```

### The Leakage:

```
Training sequence at index 889 (last training sequence):
  Data window: [889:911]  (22 rows)
  HAR monthly at row 911: mean(rows [889-22:911]) = mean([867:911])

But row 891 is in validation set!
Row 891's HAR monthly: mean([869:891]) includes row 891
```

When sequence 889 is created, it contains HAR features from row 911, which was computed using data up to row 911. This includes rows that are in the validation set!

**Impact:** Training sequences see statistical patterns (monthly mean) from future (validation/test) data.

---

## Solution: Split-First Approach

### New (CORRECT) Flow:

```python
# Step 1: Load all stock data
stock_data = load_all_data()

# Step 2: Find minimum length
min_length = min(len(df) for df in stock_data.values())

# Step 3: Calculate temporal split POINTS (by DATE index, not sequence index)
train_end_idx = int(min_length * 0.7)        # e.g., 872
val_end_idx = int(min_length * 0.85)         # e.g., 1059

# Step 4: Split RAW data FIRST (by date index)
train_raw = {stock: df.iloc[:train_end_idx] for stock, df in stock_data.items()}
val_raw = {stock: df.iloc[train_end_idx:val_end_idx] for stock, df in stock_data.items()}
test_raw = {stock: df.iloc[val_end_idx:] for stock, df in stock_data.items()}

# Step 5: Generate HAR features SEPARATELY for each split
train_har = {stock: generate_har_features(df) for stock, df in train_raw.items()}
val_har = {stock: generate_har_features(df) for stock, df in val_raw.items()}
test_har = {stock: generate_har_features(df) for stock, df in test_raw.items()}

# Step 6: Create sequences from each split's HAR data
train_dataset = create_sequences_from_har(train_har, seq_length=22, ...)
val_dataset = create_sequences_from_har(val_har, seq_length=22, ...)
test_dataset = create_sequences_from_har(test_har, seq_length=22, ...)

# Step 7: Fit normalizers on TRAINING HAR data only
train_dataset.fit_normalizers()

# Step 8: Copy normalizers to val/test
val_dataset.normalizers = train_dataset.normalizers
test_dataset.normalizers = train_dataset.normalizers
```

### Why This Works:

```
Training data: rows [0:872]
  HAR monthly at row 871: mean([849:871]) ← ALL in training set!

Validation data: rows [872:1059]
  HAR monthly at row 873: mean([851:873])
  Row 873 is in validation set, but rolling mean uses [851:873]
  Row 851 is in training set ← this is OK (only uses PAST data)

Test data: rows [1059:]
  HAR monthly at row 1060: mean([1038:1060])
  Row 1060 is in test set, rolling mean uses [1038:1060]
  Row 1038 is in validation set ← this is OK (only uses PAST data)
```

Key insight: **Rolling means can use data from previous splits**, but they cannot use data from future splits.

---

## Implementation Strategy

### Option A: Refactor `create_multi_stock_dataloaders_with_graph_method`
- Complexity: High (need to change entire data flow)
- Risk: Medium (breaking changes to existing code)
- Benefit: Clean architecture, proper temporal isolation

### Option B: Create New Function `create_multi_stock_dataloaders_with_graph_method_fixed`
- Complexity: Medium (can keep old code as backup)
- Risk: Low (non-breaking)
- Benefit: Easy to test, can compare old vs new

### Option C: Create New Dataset Class `MultiStockDatasetWithGraphMethodSplitFirst`
- Complexity: Medium (new class, but can reuse existing code)
- Risk: Low (non-breaking)
- Benefit: Clean separation, can test both approaches

---

## Recommendation

**Use Option B:** Create new function with split-first approach.

Steps:
1. Create helper function `_load_and_split_raw_data()`
2. Create helper function `_generate_har_for_split()`
3. Create helper function `_create_dataset_from_har()`
4. Implement new `create_multi_stock_dataloaders_with_graph_method_fixed()`
5. Test: Compare Dir Acc (should drop 3-5% more)
6. Update training script to use new function
7. Document migration path

---

## Expected Performance Change

| Metric | Current (HAR leak) | After Fix | Change |
|--------|-------------------|-----------|--------|
| **Dir Acc** | 66.59% | **54-56%** | -10-12% |
| **R²** | ~0.65 | **0.55-0.60** | -10-15% |
| **RMSE** | 0.002672 | **~0.001800** | Improvement |

---

## Testing Plan

1. **Unit Test:** Verify HAR features are computed separately for each split
2. **Integration Test:** Verify no data leakage using temporal isolation test
3. **Performance Test:** Run quick_test, verify Dir Acc ~54-56%
4. **Comparison Test:** Compare with baseline (should still beat Enhanced LSTM-HAR)

---

## Files to Modify

1. `src/lstm_gat_hybrid/dataset_with_graph_method.py`
   - Add new function: `create_multi_stock_dataloaders_with_graph_method_fixed()`
   - Add helper functions for split-first approach

2. `src/lstm_gat_hybrid/train_parallel_enhanced.py`
   - Update to use new function
   - Add verification flag to check for leakage

3. `tests/lstm_gat_hybrid/test_dataset_data_leakage.py`
   - Add test for HAR features isolation

---

## Timeline

- Design: 30 minutes
- Implementation: 1-2 hours
- Testing: 30 minutes
- Documentation: 30 minutes

**Total:** 3-4 hours

---

**Last Updated:** 2026-06-22
**Status:** Design complete, ready for implementation
