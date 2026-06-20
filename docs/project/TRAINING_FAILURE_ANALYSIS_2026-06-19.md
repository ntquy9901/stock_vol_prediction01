# LSTM-HAR Training Failure Analysis - CRITICAL BUGS DETECTED

**Date:** 2026-06-19  
**Training Run:** lstm_har_2026-06-19_001254  
**Status:** ❌ CATASTROPHIC FAILURE - Model Learned Nothing

---

## 📊 Training Results Summary

### Metrics Achieved

| Metric | Value | Expected | Status | Severity |
|--------|-------|----------|--------|----------|
| **Train Loss** | 0.973-0.974 (flat) | Decreasing | ❌ Flat | ⭐⭐⭐⭐ |
| **Val Loss** | 1.101623 (flat) | Decreasing | ❌ Flat | ⭐⭐⭐⭐⭐ |
| **RMSE** | 0.000666 | <0.000600 | ❌ 16% worse | ⭐⭐⭐ |
| **R²** | **-0.000026** | >0.20 | ❌ CATASTROPHIC | ⭐⭐⭐⭐⭐ |
| **Dir Acc** | **0.01%** | 48-55% | ❌ IMPOSSIBLE | ⭐⭐⭐⭐⭐ |

**Verdict:** Model is BROKEN -完全不工作!

---

## 🔍 Training Dynamics Analysis

### Loss Curve Behavior

**Epoch 1-16:**
```
Epoch 1:  Train=0.974319, Val=1.101623 (Best)
Epoch 2:  Train=0.974124, Val=1.101623 (No change)
Epoch 3:  Train=0.974588, Val=1.101623 (No change)
...
Epoch 16: Train=0.973853, Val=1.101623 (Still no change)
```

**Characteristics:**
- Train loss: Retails dao động nhẹ (0.973-0.975)
- Val loss: **HOÀN TOÀN NẰM NGANG** tại 1.101623
- Best val loss: Epoch 1 (không bao giờ improve)

**Interpretation:** Model output không improve suốt 16 epochs

---

### Learning Curve Visualization

**Expected:** Two decreasing curves (train ↓, val ↓)

**Actual:** Two flat horizontal lines = ════

```
Loss
  │
1.2│      ════════════════════  ← Val loss (FLAT)
  │      ════════════════════
1.0│  ════════════════════════  ← Train loss (FLAT)
  │  ══════════════════════════
0.8│_____________________________________
  │
  └───────────────────────────────────────> Epoch
    1    5    10   15
```

**This is WRONG!** Curves should DECREASE, not stay flat!

---

## ⚠️ Root Cause Analysis

### Issue 1: Learning Rate Too High ⭐⭐⭐⭐⭐ (90% Confidence)

**Symptoms:**
- Loss curves flat (oscillating around a point)
- Model cannot descend gradient
- Training "stuck"

**Mechanism:**
```
High LR → Large gradient steps → Overshoot minimum →
Oscillate around → Cannot settle → Flat loss curve
```

**Evidence:**
```python
learning_rate: 0.00209  # Current (Optuna-optimized)
# Expected: 0.001 - 0.0005 for this dataset
```

**Optuna Failure:**
- Optuna found 0.00209 as "best" on validation loss
- BUT validation loss was flat (not decreasing)
- Optuna confused by flat validation loss

---

### Issue 2: R² = -0.000026 (Near Zero) ⭐⭐⭐⭐⭐ (95% Confidence)

**What R² Means:**
```
R² = 1 - (SS_res / SS_tot)

SS_res = Σ(y_true - y_pred)²  # Residual sum of squares
SS_tot = Σ(y_true - y_mean)²  # Total sum of squares

R² = -0.000026 → SS_res ≈ SS_tot
→ Model prediction ≈ Using mean prediction
→ Model learned NOTHING
```

**Interpretation:**
- Model predictions are NO BETTER than predicting average
- Equivalent to: `y_pred = y_mean` for all inputs
- Model is useless

**Expected:** R² > 0.20 (model explains 20%+ variance)

---

### Issue 3: Directional Accuracy = 0.01% ⭐⭐⭐⭐⭐ (99% Confidence - BUG)

**Mathematical Impossibility:**
```
Random guessing: 50% Dir Acc (coin flip)
Our model: 0.01% Dir Acc

This is 5000× worse than random!
```

**Possible Causes:**

#### 3.1: Calculation Bug (Most Likely)
```python
# Current implementation
def directional_accuracy(y_true, y_pred):
    actual_changes = np.sign(np.diff(y_true))
    pred_changes = np.sign(np.diff(y_pred))
    accuracy = np.mean(actual_changes == pred_changes)
    return accuracy * 100

# POTENTIAL BUG: What if y_true or y_pred are:
# - All zeros? → np.sign(0) = 0 → All "equal"
# - All same value? → No changes → Division by zero
```

**Diagnostic Needed:**
```python
# ADD DEBUG CODE
print(f"y_true min: {y_true.min():.6f}, max: {y_true.max():.6f}")
print(f"y_pred min: {y_pred.min():.6f}, max: {y_pred.max():.6f}")
print(f"y_true std: {y_true.std():.6f}")
print(f"y_pred std: {y_pred.std():.6f}")

# Check if all values are the same
print(f"Unique y_true values: {len(np.unique(y_true))}")
print(f"Unique y_pred values: {len(np.unique(y_pred))}")
```

#### 3.2: Data Type Mismatch
- y_true might be scaled but y_pred is raw
- Or vice versa
- Scaling mismatch causes comparison failure

#### 3.3: Shape Mismatch
- y_true and y_pred might have different lengths
- np.diff() reduces length by 1
- If lengths don't match, comparison fails

---

### Issue 4: Train/Val Gap = 0.127 (12.7%) ⭐⭐⭐ (70% Confidence)

**Calculation:**
```
Gap = (Val Loss - Train Loss) / Train Loss
Gap = (1.101623 - 0.973853) / 0.973853
Gap = 0.127849 = 12.78%
```

**Interpretation:**
- Gap is moderate (not severe)
- Not overfitting (would be >20%)
- But not underfitting either
- **Problem:** Both train and val are flat, so gap is meaningless

---

## 🔬 Diagnostic Steps Required

### Priority 1: Debug Dir Acc = 0.01% ⭐⭐⭐⭐⭐ CRITICAL

**Add debug code to evaluation:**
```python
# In train.py, after getting predictions
print(f"\n[DEBUG] Predictions statistics:")
print(f"  y_true shape: {all_targets.shape}")
print(f"  y_pred shape: {all_predictions.shape}")
print(f"  y_true range: [{all_targets.min():.6f}, {all_targets.max():.6f}]")
print(f"  y_pred range: [{all_predictions.min():.6f}, {all_predictions.max():.6f}]")
print(f"  y_true mean: {all_targets.mean():.6f}")
print(f"  y_pred mean: {all_predictions.mean():.6f}")
print(f"  y_true std: {all_targets.std():.6f}")
print(f"  y_pred std: {all_predictions.std():.6f}")

# Check changes
actual_changes = np.sign(np.diff(all_targets))
pred_changes = np.sign(np.diff(all_predictions))
print(f"  Actual changes: {actual_changes[:20]}")
print(f"  Pred changes: {pred_changes[:20]}")
print(f"  Change agreement: {(actual_changes == pred_changes).sum()}/{len(actual_changes)}")
```

**Expected Output (if working):**
```
y_true range: [0.000100, 0.006800]
y_pred range: [0.000120, 0.006900]
Actual changes: [-1., 1., -1., 1., ...]
Change agreement: 5124/10243 ≈ 50%
```

**Bug Indicators:**
- All values are the same (no variation)
- Range is suspicious (e.g., all 0.000001)
- Mean = 0 or close to 0

---

### Priority 2: Test Learning Rate ⭐⭐⭐⭐⭐ CRITICAL

**Immediate test:** Reduce learning rate dramatically

```python
# In train.py, change learning rate
learning_rate = 0.0001  # Instead of 0.00209

# Or even lower
learning_rate = 0.00001
```

**Expected Result:**
- Train/val loss should start decreasing
- Model should start learning

---

### Priority 3: Check Data Scaling ⭐⭐⭐⭐ CRITICAL

**Verify scaling logic:**
```python
# In dataset.__getitem__, check:
# 1. Targets are RAW before scaling
# 2. Scaler is fitted on RAW targets
# 3. Inverse transform works correctly

# Test in Python
raw_target = 0.002  # Example raw target
scaled_target = target_scaler.transform([[raw_target]])[0, 0]
inverse_target = target_scaler.inverse_transform([[scaled_target]])[0, 0]

print(f"Raw: {raw_target:.6f}")
print(f"Scaled: {scaled_target:.6f}")
print(f"Inverse: {inverse_target:.6f}")
assert abs(raw_target - inverse_target) < 1e-6, "Scaling broken!"
```

---

## 🎯 Most Likely Root Causes

### Scenario 1: Learning Rate Too High (90%)

**Probability:** 90%

**Evidence:**
- Flat loss curves (characteristic of high LR)
- Optuna "best" LR was based on flat val loss
- Previous run with LR=0.001 achieved Dir Acc=41%

**Solution:**
```python
learning_rate = 0.0005  # Try much lower
# or
learning_rate = 0.0001  # Even lower
```

---

### Scenario 2: Directional Accuracy Bug (80%)

**Probability:** 80%

**Evidence:**
- 0.01% is mathematically impossible (should be 50% random)
- Previous run achieved 41% (not great, but possible)
- Calculation likely has bug

**Possible Bug:**
```python
# BUG: All predictions are nearly identical
if all(abs(p1 - p2) < 1e-6 for p1, p2 in zip(y_pred[:-1], y_pred[1:])):
    # All predictions are the same!
    pred_changes = np.sign(np.diff(y_pred))
    # pred_changes will all be 0 or NaN
    # Dir Acc = 0% or undefined
```

**Solution:**
```python
# Check prediction variance
pred_variance = np.var(all_predictions)
print(f"Prediction variance: {pred_variance:.10f}")

if pred_variance < 1e-10:
    print("ERROR: All predictions are identical!")
    print("Model outputs constant value - not learning!")
```

---

### Scenario 3: Model Too Simple (60%)

**Probability:** 60%

**Evidence:**
- LSTM with 64 hidden units may be too small
- HAR features (3) may not capture enough information
- Seq length 44 with limited data may confuse model

**Solution:**
```python
# Try larger model
hidden_size = 128  # Instead of 64
num_layers = 3      # Instead of 2
```

---

## 📊 Comparison with Previous Run

| Aspect | Previous Run | Current Run | Change | Impact |
|--------|--------------|-------------|--------|--------|
| **Config** | LR=0.001, Drop=0.2, Seq=22 | LR=0.00209, Drop=0.1, Seq=44 | Optuna "optimized" | ❌ Worse |
| **Train Loss** | Decreasing | Flat (0.973-0.975) | ❌ Stopped learning | ⭐⭐⭐⭐⭐ |
| **Val Loss** | Decreasing | Flat (1.101623) | ❌ Stopped learning | ⭐⭐⭐⭐⭐ |
| **RMSE** | 0.000575 | 0.000666 | ❌ +16% worse | ⭐⭐⭐ |
| **R²** | 0.122 | -0.000026 | ❌ Catastrophic | ⭐⭐⭐⭐⭐ |
| **Dir Acc** | 41.05% | 0.01% | ❌ Bug/Worse | ⭐⭐⭐⭐⭐ |

**Conclusion:** Optuna "optimization" made model **MUCH WORSE**! 😱

---

## 🔧 Immediate Actions Required

### Action 1: Roll Back Configuration ⭐⭐⭐⭐⭐ CRITICAL

**Revert to previous working config:**
```python
# Rollback
learning_rate: 0.001     # Was: 0.00209 (Optuna)
dropout: 0.2              # Was: 0.1 (manual)
seq_length: 22           # Was: 44 (Optuna)
patience: 10             # Was: 15 (manual)
weight_decay: 1e-5       # Was: 0.0 (Optuna)
```

**Rationale:**
- Previous config achieved Dir Acc = 41% (possible, though low)
- Optuna config achieved Dir Acc = 0.01% (impossible/broken)
- Optuna recommendations were WRONG

---

### Action 2: Debug Dir Acc Calculation ⭐⭐⭐⭐⭐ CRITICAL

**Add diagnostic code:**
```python
# After getting all_predictions, all_targets
print(f"\n[CRITICAL DEBUG]")
print(f"Predictions variance: {np.var(all_predictions):.10f}")
print(f"Targets variance: {np.var(all_targets):.10f}")

if np.var(all_predictions) < 1e-8:
    print("❌ ERROR: All predictions are identical!")
    print("Model is outputting constant value!")

# Check directional accuracy manually
actual_changes = np.sign(np.diff(all_targets))
pred_changes = np.sign(np.diff(all_predictions))

print(f"Actual changes (first 20): {actual_changes[:20]}")
print(f"Pred changes (first 20): {pred_changes[:20]}")
print(f"Agreement: {np.sum(actual_changes == pred_changes)}/{len(actual_changes)}")

# Calculate expected random accuracy
import random
random_agreement = 0
for _ in range(1000):
    rand_pred = np.random.choice(actual_changes, size=len(actual_changes))
    random_agreement += np.sum(rand_pred == actual_changes)
print(f"Random guessing agreement: {random_agreement}/1000 = {random_agreement/10}%")
```

---

### Action 3: Try Much Lower Learning Rate ⭐⭐⭐⭐⭐ CRITICAL

**Experiment:**
```python
# Test with very low LR
learning_rate = 0.0001  # 20× lower than Optuna recommendation

# Or even lower
learning_rate = 0.00001  # 200× lower
```

**Expected:** Model should start learning (loss curves decrease)

---

### Action 4: Simplify Model ⭐⭐⭐

**Try simpler architecture:**
```python
# Remove one LSTM layer
num_layers = 1  # Instead of 2

# Or reduce hidden size
hidden_size = 32  # Instead of 64
```

---

## 📈 Expected Results After Fixes

### If Learning Rate is the Problem:

| Metric | Current | Expected (LR=0.0001) | Expected (LR=0.00001) |
|--------|----------|---------------------|----------------------|
| **Train Loss** | 0.974 (flat) | 0.80-0.90 (decreasing) | 0.70-0.80 (decreasing) |
| **Val Loss** | 1.102 (flat) | 0.90-1.00 (decreasing) | 0.85-0.95 (decreasing) |
| **RMSE** | 0.000666 | 0.000550 | 0.000520 |
| **Dir Acc** | 0.01% (bugged) | 40-45% | 45-50% |

---

### If Dir Acc is the Bug:

**After fixing calculation:**
```
Current: 0.01% (impossible)
After Fix: 40-50% (possible, matches previous run)
```

---

## 🎯 Recommended Next Steps

### Step 1: Add Diagnostic Code (5 minutes)

Add debug code to `train.py` to output:
1. Prediction/target statistics
2. Prediction variance
3. Directional agreement calculation

### Step 2: Verify Dir Acc Calculation (5 minutes)

Test with random data to ensure function returns ~50%:
```python
np.random.seed(42)
y_test = np.random.randn(1000)
p_test = np.random.randn(1000)
acc = directional_accuracy(y_test, p_test)
print(f"Random Dir Acc: {acc}%")  # Should be ~50%
```

### Step 3: Roll Back Config (5 minutes)

Revert to previous configuration that achieved 41% Dir Acc.

### Step 4: Try Lower Learning Rate (20 minutes)

Retrain with LR=0.0001 to see if model starts learning.

---

## 📚 Summary

### Current Status: ❌ BROKEN

- **Train/val curves:** Flat horizontal lines ════
- **R²:** -0.000026 (model useless)
- **Dir Acc:** 0.01% (broken/impossible)
- **RMSE:** 0.000666 (worse than before)

### Root Causes:

1. **Learning rate too high** (90% confidence)
2. **Dir Acc calculation bug** (80% confidence)
3. **Optuna gave bad recommendations** (70% confidence)

### Actions:

1. ✅ **Roll back config** to previous working version
2. ✅ **Add debug code** to verify Dir Acc calculation
3. ✅ **Test with lower LR** (0.0001 or 0.00001)
4. ✅ **Verify model outputs vary** (not constant)

---

**Analysis Date:** 2026-06-19  
**Status:** CRITICAL ISSUES - Model Non-Functional  
**Confidence:** Root cause = Learning Rate + Dir Acc Bug  
**Next Action:** Debug and Rollback

---

*Last Updated: 2026-06-19*  
*Version: 1.0*  
*Author: Stock Volatility Prediction Team*