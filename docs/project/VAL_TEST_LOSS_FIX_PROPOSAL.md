# Fix Val/Test Loss Scale Mismatch - Implementation Plan

**Date:** 2026-06-27
**Priority:** HIGH
**Estimated Time:** 2-3 hours

---

## Problem Summary

**Symptom:**
- Val Loss: ~136,117 (normalized scale)
- Test MSE: 7.07e-06 (original scale)
- Test R²: 0.71 (good performance)
- Test Dir Acc: 67.58% (good performance)

**Root Cause:**
Val Loss và Test MSE được tính trên **khác scale**:
- Val Loss: MSE trên normalized targets (mean=0, std=1)
- Test MSE: MSE sau inverse transform (original scale)

**Why This Matters:**
1. Early stopping dựa trên Val Loss sai scale → có thể dừng quá sớm/quá muộn
2. Không thể so sánh trực tiếp Val Loss với Test MSE
3. Learning curves không phản ánh true model performance

---

## Solution Options

### Option 1: Inverse Transform trong Validation Loop (RECOMMENDED) ✅

**Phương án:** Áp dụng inverse transform cho validation predictions TRƯỚC khi tính loss.

**Ưu điểm:**
- ✅ Consistent với Test MSE (cùng original scale)
- ✅ Early stopping dựa trên actual prediction error
- ✅ So sánh được trực tiếp với LSTM-HAR Enhanced (67.90% Dir Acc)
- ✅ Learning curves có ý nghĩa thực tế

**Nhược điểm:**
- ⚠️ Cần lưu tất cả predictions trước khi tính loss (memory overhead)
- ⚠️ Validation loop phức tạp hơn một chút

**Implementation:**

```python
def validate(model, dataloader, criterion, device, dataset=None):
    """
    Validate with loss computed on ORIGINAL scale (consistent with test)

    FIXED: Inverse transform predictions BEFORE computing loss
    """
    model.eval()
    n_batches = 0

    all_predictions_norm = []
    all_targets_norm = []

    # Step 1: Collect all normalized predictions and targets
    with torch.no_grad():
        for x, adj_matrix, y, _ in dataloader:
            x = x.to(device)
            adj_matrix = adj_matrix.to(device)
            y = y.to(device)

            batch_size, num_stocks = y.shape

            optimizer.zero_grad()
            predictions = model(x, adj_matrix)
            predictions_flat = predictions.reshape(batch_size * num_stocks)
            y_flat = y.reshape(batch_size * num_stocks)

            # Collect normalized predictions and targets
            all_predictions_norm.extend(predictions_flat.cpu().numpy())
            all_targets_norm.extend(y_flat.cpu().numpy())

            n_batches += 1

    # Convert to numpy
    all_predictions_norm = np.array(all_predictions_norm).flatten()
    all_targets_norm = np.array(all_targets_norm).flatten()

    # Step 2: Extract dataset (handle Subset wrapper)
    actual_dataset = dataset
    if dataset is not None and hasattr(dataset, 'dataset'):
        actual_dataset = dataset.dataset

    # Step 3: Inverse transform to original scale
    if actual_dataset is not None and hasattr(actual_dataset, 'target_normalizers'):
        all_predictions_denorm = np.zeros_like(all_predictions_norm)
        all_targets_denorm = np.zeros_like(all_targets_norm)

        for i in range(len(all_predictions_norm)):
            stock_idx = i % len(actual_dataset.stock_names)
            stock_name = actual_dataset.stock_names[stock_idx]

            if stock_name in actual_dataset.target_normalizers:
                # Denormalize prediction
                all_predictions_denorm[i] = \
                    actual_dataset.target_normalizers[stock_name].inverse_transform(
                        all_predictions_norm[i:i+1].reshape(1, -1)
                    ).flatten()[0]
                # Denormalize target
                all_targets_denorm[i] = \
                    actual_dataset.target_normalizers[stock_name].inverse_transform(
                        all_targets_norm[i:i+1].reshape(1, -1)
                    ).flatten()[0]
            else:
                all_predictions_denorm[i] = all_predictions_norm[i]
                all_targets_denorm[i] = all_targets_norm[i]

        # Step 4: ✅ FIX: Compute loss on ORIGINAL scale (consistent with test)
        loss_tensor = criterion(
            torch.FloatTensor(all_predictions_denorm).to(device),
            torch.FloatTensor(all_targets_denorm).to(device)
        )
        avg_loss = loss_tensor.item()

        # Compute metrics on denormalized data
        metrics = evaluate_predictions(all_targets_denorm, all_predictions_denorm)
    else:
        # Fallback: Compute on normalized scale (WARNING: not consistent with test!)
        loss_tensor = criterion(
            torch.FloatTensor(all_predictions_norm).to(device),
            torch.FloatTensor(all_targets_norm).to(device)
        )
        avg_loss = loss_tensor.item()

        metrics = evaluate_predictions(all_targets_norm, all_predictions_norm)

    return avg_loss, metrics
```

---

### Option 2: Compute Test Metrics on Normalized Scale (NOT RECOMMENDED) ❌

**Phương án:** Compute Test metrics trên normalized scale thay vì original scale.

**Ưu điểm:**
- ✅ Đơn giản implement (không cần sửa validation loop)
- ✅ Consistent scale cho Val Loss và Test MSE

**Nhược điểm:**
- ❌ Test metrics kém interpretable (trên normalized scale)
- ❌ KHÔNG consistent với LSTM-HAR Enhanced approach
- ❌ Không thể so sánh trực tiếp với baseline (67.90% Dir Acc)

**Conclusion:** **KHÔNG dùng option này** vì chúng ta muốn so sánh với LSTM-HAR Enhanced.

---

## Implementation Steps (Option 1)

### Step 1: Add Debugging Prints (15 min)
```python
# Add to validate() function BEFORE inverse transform
print(f"[DEBUG validate] Before inverse transform:")
print(f"  predictions_norm range: [{all_predictions_norm.min():.6f}, {all_predictions_norm.max():.6f}]")
print(f"  targets_norm range: [{all_targets_norm.min():.6f}, {all_targets_norm.max():.6f}]")

# AFTER inverse transform
print(f"[DEBUG validate] After inverse transform:")
print(f"  predictions_denorm range: [{all_predictions_denorm.min():.6f}, {all_predictions_denorm.max():.6f}]")
print(f"  targets_denorm range: [{all_targets_denorm.min():.6f}, {all_targets_denorm.max():.6f}]")
```

### Step 2: Implement Inverse Transform in Validation (45 min)
- Copy code from inverse transform section (Line 242-274)
- Apply to ALL predictions before computing loss
- Compute loss on original scale

### Step 3: Test with Quick Test Mode (30 min)
```bash
cd D:\bmad-projects\stock_vol_prediction01
python -m src.lstm_gat_hybrid.train_parallel_enhanced --graph_method knn --quick_test
```

**Expected Output:**
```
Epoch | Train Loss | Val Loss    | Val Dir Acc | Val RMSE   | LR
---------------------------------------------------------------
1     | 0.001234   | 0.001456    | 65.23%      | 0.002345   | 0.005000
2     | 0.000987   | 0.001234    | 66.45%      | 0.002123   | 0.005000
...
```

**Success Criteria:**
- Val Loss và Test MSE cùng order of magnitude (~1e-6 to ~1e-3)
- Val Loss ≈ Train Loss (gap < 0.05)
- Val Dir Acc ≈ Test Dir Acc (gap < 5%)

### Step 4: Run Full Training (2 hours)
```bash
python -m src.lstm_gat_hybrid.train_parallel_enhanced --graph_method knn
```

**Expected Results:**
- Val Loss: ~1e-4 to ~1e-3 (original scale)
- Test MSE: ~1e-6 to ~1e-5 (original scale)
- Val/Test Dir Acc: 65-68% (similar to LSTM-HAR Enhanced)
- Overfitting gap (Val - Train) < 0.05

### Step 5: Compare with Baselines (30 min)
```python
# Expected comparison:
LSTM-HAR Enhanced:    67.90% Dir Acc
Parallel LSTM-GNN:     65-68% Dir Acc (target: >67.90%)
```

---

## Files to Modify

1. **src/lstm_gat_hybrid/train_parallel_enhanced.py**
   - Function: `validate()` (Line 199-281)
   - Change: Apply inverse transform BEFORE computing loss

2. **docs/project/VAL_TEST_LOSS_FIX_COMPLETED.md** (create)
   - Document: Before/after comparison
   - Include: Learning curves, metrics comparison

---

## Verification Checklist

- [ ] Debugging prints added to validate()
- [ ] Ran 1 epoch to check normalization status
- [ ] Confirmed: `y_flat.mean()` ≈ 0.0, `y_flat.std()` ≈ 1.0 (normalized)
- [ ] Implemented inverse transform in validation
- [ ] Tested with `--quick_test` mode (5 epochs)
- [ ] Confirmed: Val Loss and Test MSE same order of magnitude
- [ ] Ran full training (70 epochs)
- [ ] Compared results with LSTM-HAR Enhanced (67.90%)
- [ ] Updated documentation with before/after metrics

---

## Expected Impact

**Before Fix:**
```
Val Loss: 136,117 (normalized scale)
Test MSE: 7.07e-06 (original scale)
Gap: 10^11 times (!!)
```

**After Fix:**
```
Val Loss: ~1e-4 (original scale)
Test MSE: ~1e-6 (original scale)
Gap: ~100x (reasonable: test > val)
```

**Benefits:**
1. ✅ Early stopping dựa trên actual error
2. ✅ Learning curves có ý nghĩa
3. ✅ Comparable với LSTM-HAR Enhanced
4. ✅ Better model selection

---

## References

- **LSTM-HAR Enhanced approach:** `src/lstm_har_enhanced/train_with_overfitting_prevention.py`
- **Current validation code:** `src/lstm_gat_hybrid/train_parallel_enhanced.py` Line 199-281
- **Normalization utility:** `src/common/data_normalization.py`
- **Gemini code review:** User-provided feedback (2026-06-27)

---

**Last Updated:** 2026-06-27
**Status:** Ready for implementation
**Priority:** HIGH (fixing this will enable proper model comparison)
