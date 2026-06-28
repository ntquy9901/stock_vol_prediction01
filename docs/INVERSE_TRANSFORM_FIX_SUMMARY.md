# Inverse Transform Fix - Summary

**Date:** 2026-06-21
**Issue:** Parallel LSTM-GNN metrics calculated on NORMALIZED scale instead of ORIGINAL scale
**Impact:** RMSE, MSE, QLIKE were completely wrong (1000x higher than expected)

---

## 🔴 Root Cause

**Location:** `src/lstm_gat_hybrid/dataset_with_graph_method.py` lines 446-448

```python
# Apply temporal split using Subset
train_dataset = torch.utils.data.Subset(train_dataset, range(0, train_end))
val_dataset = torch.utils.data.Subset(val_dataset, range(train_end, val_end))
test_dataset = torch.utils.data.Subset(test_dataset, range(val_end, n))
```

**Problem:**
- `datasets[1]` returned = `Subset(val_dataset, range(...))`
- `validate()` function checked: `hasattr(dataset, 'target_normalizers')`
- `Subset` object doesn't have `target_normalizers` attribute
- Inverse transform was SKIPPED
- Metrics calculated on NORMALIZED data (mean=0, std=1) instead of ORIGINAL physical scale

---

## ✅ Solution Applied

### **File 1: `src/lstm_gat_hybrid/train_parallel_enhanced.py`**

**Change 1: Extract original dataset from Subset (Line 230-260)**

```python
# ✅ FIX: Handle Subset wrapper - extract original dataset
actual_dataset = dataset
if dataset is not None and hasattr(dataset, 'dataset'):
    # dataset is a Subset, extract original MultiStockDatasetWithGraphMethod
    actual_dataset = dataset.dataset
    print(f"[DEBUG validate] Extracted original dataset from Subset")
    print(f"[DEBUG validate] Original dataset type: {type(actual_dataset).__name__}")
    print(f"[DEBUG validate] Has target_normalizers: {hasattr(actual_dataset, 'target_normalizers')}")

# Inverse transform to get back to original scale
if actual_dataset is not None and hasattr(actual_dataset, 'target_normalizers'):
    print(f"[DEBUG validate] Applying inverse transform to {len(all_predictions)} predictions")
    # ... inverse transform logic using actual_dataset ...
    print(f"[DEBUG validate] Inverse transform complete")
    print(f"[DEBUG validate] Prediction range: [{all_predictions.min():.6f}, {all_predictions.max():.6f}]")
    print(f"[DEBUG validate] Target range: [{all_targets.min():.6f}, {all_targets.max():.6f}]")
else:
    print(f"[WARNING validate] No inverse transform applied! Metrics on NORMALIZED scale!")
```

**Change 2: Add quick_test parameter (Line 284, 311-325)**

```python
def train_parallel_lstm_gat_enhanced(graph_method='correlation', quick_test=False):
    """..."""
    # Quick test mode or full training
    if quick_test:
        config.num_epochs = 5
        config.patience = 3
        config.min_epochs = 3
        print(f"\n[QUICK TEST MODE] Training for 5 epochs to verify setup...")
    else:
        config.num_epochs = 50  # Paper: optimal 40-50
        config.patience = 15  # LSTM-HAR: 15 (not paper's 5)
        config.min_epochs = 15  # LSTM-HAR: minimum epochs
```

**Change 3: Add CLI argument (Line 570-571)**

```python
parser.add_argument('--quick_test', action='store_true',
                    help='Run quick test (5 epochs) to verify setup')

# Usage
results = train_parallel_lstm_gat_enhanced(graph_method=args.graph_method, quick_test=args.quick_test)
```

---

## 📊 Expected Results After Fix

### **Before Fix (WRONG):**

```json
{
  "model": "Parallel LSTM-GNN (knn graph)",
  "test_metrics": {
    "mse": 0.7299,          // ❌ Normalized scale (1000x too high)
    "rmse": 0.8543,         // ❌ Normalized scale (1000x too high)
    "mae": 0.6096,          // ❌ Normalized scale (1000x too high)
    "r2": 0.1298,
    "qlike": 7674763.5,     // ❌ EXPLODED (normalized scale)
    "directional_accuracy": 65.12%  // ✅ Correct (diff-based metric)
  }
}
```

### **After Fix (EXPECTED):**

```json
{
  "model": "Parallel LSTM-GNN (knn graph)",
  "test_metrics": {
    "mse": ~0.000001 - 0.00001,     // ✅ Original scale
    "rmse": ~0.001 - 0.003,         // ✅ Original scale
    "mae": ~0.0003 - 0.001,         // ✅ Original scale
    "r2": ~0.1 - 0.3,
    "qlike": ~0.6 - 0.8,            // ✅ Original scale
    "directional_accuracy": ~65%     // ✅ Same (not affected by scale)
  }
}
```

---

## 🧪 Testing

### **Quick Test (5 epochs):**

```bash
python src/lstm_gat_hybrid/train_parallel_enhanced.py --graph_method knn --quick_test
```

**Expected Output:**
```
[QUICK TEST MODE] Training for 5 epochs to verify setup...
[DEBUG validate] Extracted original dataset from Subset
[DEBUG validate] Original dataset type: MultiStockDatasetWithGraphMethod
[DEBUG validate] Has target_normalizers: True
[DEBUG validate] Applying inverse transform to 4500 predictions
[DEBUG validate] Inverse transform complete
[DEBUG validate] Prediction range: [0.000821, 0.032456]  ← Physical scale
[DEBUG validate] Target range: [0.000745, 0.031234]     ← Physical scale

Test RMSE: 0.002345        ← ~1000x smaller than before!
Test QLIKE: 0.7234         ← ~10 million times smaller!
Test Dir Acc: 65.12%       ← Same (not affected by scale)
```

### **Full Training (50 epochs):**

```bash
python src/lstm_gat_hybrid/train_parallel_enhanced.py --graph_method knn
```

---

## 🔍 How to Verify Fix Worked

### **Check 1: Inverse Transform Applied**

Look for these DEBUG messages in console output:
```
[DEBUG validate] Extracted original dataset from Subset
[DEBUG validate] Has target_normalizers: True
[DEBUG validate] Applying inverse transform to XXX predictions
[DEBUG validate] Inverse transform complete
```

### **Check 2: Physical Scale Values**

Prediction and target ranges should be in physical scale (volatility):
- Range: **[0.0005, 0.05]** (typical Parkinson volatility)
- NOT: **[-3, 3]** (normalized scale)

### **Check 3: Reasonable Metrics**

- RMSE: **0.001 - 0.01** (physical scale)
- QLIKE: **0.5 - 1.0** (physical scale)
- MSE: **0.000001 - 0.0001** (physical scale)

If you see:
- RMSE > 0.1 → Fix NOT applied (still on normalized scale)
- QLIKE > 100 → Fix NOT applied (exploded on normalized scale)

---

## 📋 Comparison with LSTM-HAR Enhanced

### **LSTM-HAR Enhanced (Correct Implementation):**

```python
# src/lstm_har_enhanced/train_with_overfitting_prevention.py line 357-358
predictions_np = dataset.target_scaler.inverse_transform(predictions.cpu().numpy())
targets_np = dataset.target_scaler.inverse_transform(y_batch.numpy().reshape(-1, 1))
```

**Key Difference:**
- LSTM-HAR Enhanced passes `dataset` directly (not wrapped in Subset)
- `dataset.target_scaler` is accessible directly
- No need to extract from Subset wrapper

### **Parallel LSTM-GNN (Fixed Implementation):**

```python
# src/lstm_gat_hybrid/train_parallel_enhanced.py line 230-260
actual_dataset = dataset
if dataset is not None and hasattr(dataset, 'dataset'):
    actual_dataset = dataset.dataset  # Extract from Subset

if actual_dataset is not None and hasattr(actual_dataset, 'target_normalizers'):
    # Use actual_dataset.target_normalizers
```

**Why Extra Step Needed:**
- `create_multi_stock_dataloaders_with_graph_method()` wraps datasets in `Subset`
- Need to extract original via `dataset.dataset` attribute
- Then access `target_normalizers` from original dataset

---

## 🎯 Next Steps

1. **Run quick test:** `python src/lstm_gat_hybrid/train_parallel_enhanced.py --graph_method knn --quick_test`
2. **Verify inverse transform:** Check DEBUG messages
3. **Check metrics:** Ensure RMSE/QLIKE in physical scale range
4. **If verified:** Run full training (50 epochs)
5. **Compare with LSTM-HAR Enhanced:** Expect similar RMSE/QLIKE, higher Dir Acc

---

## 📚 Lessons Learned

1. **Subset wrapper loses attributes** - Always check if dataset is wrapped
2. **Inverse transform is MANDATORY** - All metrics except Dir Acc depend on scale
3. **Test on small data first** - Quick test mode catches bugs early
4. **Compare with baseline** - LSTM-HAR Enhanced's RMSE (~0.0005) is reference point

---

**Last Updated:** 2026-06-21
**Status:** Fix Applied, Ready for Quick Test
