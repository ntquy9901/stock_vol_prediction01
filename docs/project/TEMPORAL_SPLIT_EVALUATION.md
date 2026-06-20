# Temporal Split Evaluation Methodology - 3-Way Split (70/15/15)

**Date:** 2026-06-19
**Status:** ✅ Ready for Implementation
**Purpose:** Fix data leakage issue and enable proper validation during training

---

## Problem Statement

### Current Issue: Data Leakage in LSTM Models

**Existing methodology (WRONG):**
```python
# LSTM models use RANDOM split
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
```

**Problems:**
1. ❌ **Data Leakage:** Random split allows future data in training set
2. ❌ **Overestimated Metrics:** Model sees test patterns during training
3. ❌ **No Validation:** Cannot tune hyperparameters properly
4. ❌ **Unreliable Results:** Test performance not representative of real performance

---

## Solution: 3-Way Temporal Split

### New Methodology (CORRECT)

```
Train (70%):      2006-10-27 to 2020-03-23  (3,423 days = 13.7 years)
Validation (15%): 2020-03-24 to 2021-08-03  (734 days = 2.9 years)
Test (15%):       2021-08-04 to 2026-06-09  (733 days = 2.9 years)
```

### Benefits

| Benefit | Description |
|---------|-------------|
| **No Data Leakage** | Chronological split prevents future data in training |
| **Proper Validation** | Val set used for early stopping and model selection |
| **Reliable Test** | Test set completely unseen during training |
| **Overfitting Detection** | Compare val vs test metrics to detect overfitting |

---

## Implementation

### 1. New Utility Module

**File:** `src/common/temporal_split.py`

**Key Functions:**
```python
from src.common.temporal_split import TemporalSplitter

# Create splitter
splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

# Split dataset
train_loader, val_loader, test_loader = splitter.create_dataloaders(
    dataset, batch_size=32
)
```

**Features:**
- ✅ Chronological split (not random)
- ✅ Configurable ratios (default 70/15/15)
- ✅ Works with PyTorch Datasets
- ✅ Creates dataloaders directly
- ✅ Prints split information for documentation

---

### 2. Updated Training Script

**File:** `src/lstm_har_enhanced/train_with_validation.py`

**Key Improvements:**
```python
# OLD: 2-way random split
train_dataset, test_dataset = torch.utils.data.random_split(...)

# NEW: 3-way temporal split
splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
train_loader, val_loader, test_loader = splitter.create_dataloaders(dataset)
```

**Training Loop:**
```python
for epoch in range(num_epochs):
    # Train phase
    model.train()
    train_loss = train_epoch(model, train_loader)

    # Validation phase (NEW)
    model.eval()
    val_loss = validate_epoch(model, val_loader)

    # Early stopping based on validation loss (NEW)
    if val_loss < best_val_loss:
        save_model()
```

**Evaluation:**
```python
# Evaluate on validation set (NEW)
val_metrics = evaluate(model, val_loader)

# Evaluate on test set
test_metrics = evaluate(model, test_loader)

# Compare val vs test (NEW)
compare_metrics(val_metrics, test_metrics)
```

---

## Comparison: Old vs New Methodology

### Old Methodology (2-Way Random Split)

| Aspect | Train | Test | Issues |
|--------|-------|------|--------|
| **Split method** | Random 80% | Random 20% | Data leakage ❌ |
| **Time period** | Mixed | Mixed | Not chronological ❌ |
| **Validation** | None | None | No early stopping ❌ |
| **Metrics** | Test only | Test only | Overestimated ❌ |
| **Reliability** | Low | Low | Not representative ❌ |

**Example:**
```
Total data: 2006-2026 (4,890 days)

Random split:
- Train: Random 80% (mix of 2006-2026)
- Test: Random 20% (mix of 2006-2026)

Problem: Data from 2022-2026 can appear in BOTH train and test!
Result: Model learns future patterns = Data leakage
```

---

### New Methodology (3-Way Temporal Split)

| Aspect | Train | Validation | Test | Benefits |
|--------|-------|------------|------|----------|
| **Split method** | First 70% | Next 15% | Last 15% | Chronological ✅ |
| **Time period** | 2006-2020 | 2020-2021 | 2021-2026 | No leakage ✅ |
| **Validation** | Yes | Yes | Yes | Proper tuning ✅ |
| **Metrics** | Train, Val | Val | Test | Comprehensive ✅ |
| **Reliability** | High | High | High | Representative ✅ |

**Example:**
```
Total data: 2006-2026 (4,890 days)

Temporal split:
- Train: 2006-2020 (3,423 days = 70%)
- Val:   2020-2021 (734 days = 15%)
- Test:  2021-2026 (733 days = 15%)

Benefit: Strict chronological order = No data leakage
Result: Reliable metrics
```

---

## Metrics Comparison

### Expected Output

After training with new methodology, you'll see:

```
================================================================================
VALIDATION VS TEST COMPARISON
================================================================================
Metric          Validation       Test            Difference
--------------------------------------------------------------
RMSE            0.000603        0.000615        +0.000012
MAE             0.000303        0.000308        +0.000005
R²              0.136000        0.128000        -0.008000
Dir Acc         67.90%          66.50%          -1.40%

================================================================================
OVERFITTING CHECK
================================================================================
[OK] Test performance similar to validation - No significant overfitting
================================================================================
```

### Interpretation

**Good Results (No Overfitting):**
- Test RMSE ≈ Val RMSE (±10%)
- Test R² ≈ Val R² (±5%)
- Test Dir Acc ≈ Val Dir Acc (±2%)

**Warning Signs (Possible Overfitting):**
- Test RMSE >> Val RMSE (>10% degradation)
- Test R² << Val R² (>5% degradation)
- Test Dir Acc << Val Dir Acc (>2% degradation)

---

## Implementation Steps

### Step 1: Update All LSTM Models

**Models to update:**
1. ✅ Simple LSTM (`src/lstm_baseline/train.py`)
2. ✅ LSTM-HAR (`src/lstm_har_baseline/train.py`)
3. ✅ Enhanced LSTM-HAR (`src/lstm_har_enhanced/train_enhanced.py`)

**Changes:**
```python
# Replace random_split with temporal_split
from src.common.temporal_split import TemporalSplitter

splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
train_loader, val_loader, test_loader = splitter.create_dataloaders(
    dataset, batch_size=32
)
```

---

### Step 2: Update Training Loop

**Add validation phase:**
```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = train_epoch(model, train_loader)

    # Validation phase (NEW)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    # Early stopping based on validation loss (NEW)
    if val_loss < best_val_loss:
        save_model()
```

---

### Step 3: Add Val vs Test Comparison

**Evaluate on both sets:**
```python
# Validation metrics
val_metrics = evaluate(model, val_loader)

# Test metrics
test_metrics = evaluate(model, test_loader)

# Compare
print("Val vs Test Comparison:")
print(f"RMSE: {val_metrics['rmse']:.6f} vs {test_metrics['rmse']:.6f}")
print(f"Dir Acc: {val_metrics['dir_acc']:.2f}% vs {test_metrics['dir_acc']:.2f}%")
```

---

### Step 4: Document Results

**Save comprehensive JSON:**
```python
results = {
    'validation_metrics': {...},
    'test_metrics': {...},
    'val_test_diff': {
        'rmse_diff': test_metrics['rmse'] - val_metrics['rmse'],
        'dir_acc_diff': test_metrics['dir_acc'] - val_metrics['dir_acc']
    }
}
```

---

## Expected Impact

### On Metrics

**Before (Random Split - Overestimated):**
```
Enhanced LSTM-HAR:
- RMSE: 0.000603 (overestimated - saw test data during training)
- Dir Acc: 67.90% (overestimated - learned future patterns)
```

**After (Temporal Split - Realistic):**
```
Enhanced LSTM-HAR:
- Val RMSE: 0.000610 (realistic - early stopping on val)
- Test RMSE: 0.000620 (realistic - completely unseen data)
- Val Dir Acc: 67.50%
- Test Dir Acc: 66.00% (slight degradation = normal)
```

**Expected changes:**
- Test metrics may be slightly worse (no data leakage)
- Val metrics used for early stopping
- Gap between val and test indicates overfitting

---

### On Model Selection

**Before (No Validation):**
- Trained for fixed epochs
- No early stopping
- Test metrics biased (data leakage)

**After (With Validation):**
- Early stopping on validation loss
- Model selected based on validation performance
- Test metrics unbiased (completely unseen)

---

## Usage Examples

### Example 1: Train Model with Validation

```python
from src.lstm_har_enhanced.train_with_validation import train_enhanced_lstm_har_with_val

# Train with 3-way temporal split
model, val_metrics, test_metrics = train_enhanced_lstm_har_with_val('data/processed')

# Check val vs test gap
print(f"Val RMSE: {val_metrics['rmse']:.6f}")
print(f"Test RMSE: {test_metrics['rmse']:.6f}")
print(f"Difference: {test_metrics['rmse'] - val_metrics['rmse']:+.6f}")
```

---

### Example 2: Use TemporalSplitter Directly

```python
from src.common.temporal_split import TemporalSplitter

# Create splitter
splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

# Get split info before splitting
info = splitter.get_info(len(dataset))
print(f"Train: {info['train_size']}, Val: {info['val_size']}, Test: {info['test_size']}")

# Create dataloaders
train_loader, val_loader, test_loader = splitter.create_dataloaders(
    dataset, batch_size=32
)

# Use in training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = val_epoch(model, val_loader)  # NEW!
```

---

## Best Practices

### DO ✅

1. **Always use temporal split for time series**
   ```python
   splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
   ```

2. **Use validation set for early stopping**
   ```python
   if val_loss < best_val_loss:
       save_model()
   ```

3. **Compare val vs test metrics**
   ```python
   print(f"Val vs Test RMSE: {val_rmse:.6f} vs {test_rmse:.6f}")
   ```

4. **Document val vs test differences**
   ```python
   results['val_test_diff'] = {
       'rmse_diff': test_rmse - val_rmse,
       'dir_acc_diff': test_dir_acc - val_dir_acc
   }
   ```

---

### DON'T ❌

1. **Don't use random_split for time series**
   ```python
   # WRONG: Data leakage!
   train, test = torch.utils.data.random_split(dataset, [0.8, 0.2])
   ```

2. **Don't skip validation**
   ```python
   # WRONG: No early stopping
   for epoch in range(100):  # Fixed epochs
       train(model, data)
   ```

3. **Don't ignore val vs test gap**
   ```python
   # WRONG: Only report test metrics
   print(f"Test RMSE: {test_rmse:.6f}")
   # Should compare with validation!
   ```

---

## Files to Update

### Created Files
- ✅ `src/common/temporal_split.py` - Temporal split utilities
- ✅ `src/lstm_har_enhanced/train_with_validation.py` - Example implementation

### Files to Update
- ⏳ `src/lstm_baseline/train.py` - Update Simple LSTM training
- ⏳ `src/lstm_har_baseline/train.py` - Update LSTM-HAR training
- ⏳ `src/lstm_har_enhanced/train_enhanced.py` - Update Enhanced LSTM-HAR training

---

## Testing Plan

### Test 1: Verify No Data Leakage
```python
# Check dates don't overlap
assert train_end_date < val_start_date
assert val_end_date < test_start_date
```

### Test 2: Verify Val vs Test Gap
```python
# Test gap should be small (< 10% RMSE degradation)
rmse_gap = (test_rmse - val_rmse) / val_rmse
assert rmse_gap < 0.1, f"Large gap: {rmse_gap:.2%}"
```

### Test 3: Verify Early Stopping Works
```python
# Model should stop before max epochs if no improvement
assert best_epoch < num_epochs
```

---

## Summary

### Key Changes

| Change | Before | After |
|---------|--------|-------|
| **Split method** | Random (80/20) | Temporal (70/15/15) |
| **Validation** | None | Yes (15% for early stopping) |
| **Data leakage** | Yes ❌ | No ✅ |
| **Metrics** | Test only (biased) | Val + Test (unbiased) |
| **Reliability** | Low | High |

### Benefits

1. ✅ **No data leakage** - Chronological split prevents future data in training
2. ✅ **Proper validation** - Val set used for early stopping and model selection
3. ✅ **Reliable metrics** - Test set completely unseen during training
4. ✅ **Overfitting detection** - Compare val vs test to detect overfitting
5. ✅ **Better models** - Early stopping prevents overtraining

### Next Steps

1. Test `train_with_validation.py` on current data
2. Compare results with old methodology
3. Update all LSTM models to use temporal split
4. Document all results with val vs test comparison

---

**Status:** ✅ Ready for Implementation
**Priority:** HIGH - Fix critical data leakage issue
**Estimated Impact:** Test metrics may degrade slightly (realistic), but model selection improved

---

**Last Updated:** 2026-06-19
**Author:** Stock Volatility Prediction Team
