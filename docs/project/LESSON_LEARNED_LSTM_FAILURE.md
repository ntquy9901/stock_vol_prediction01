# Lesson Learned: LSTM Training Failure - Self-Reflection Document

**Date:** 2026-06-17
**Project:** stock_vol_prediction01
**Issue:** LSTM model not learning (QLIKE: 0.74, Dir Acc: 0.5%)
**Status:** Critical Failure

---

## 🔍 What Happened?

### Problem Statement

LSTM model for volatility prediction showed:
- **Learning curve flat** - Train/Val loss ~0.9, minimal improvement
- **QLIKE: 0.74** - Target was < 0.20 (3.7× worse)
- **Directional Accuracy: 0.5%** - Near random (50%)
- **Model completely useless** for prediction

### Symptoms Observed

1. ❌ Loss doesn't decrease over epochs
2. ❌ Train and validation loss nearly identical (no gap)
3. ❌ Model predictions close to mean (no actual learning)
4. ❌ Extremely poor evaluation metrics

---

## 🎯 Root Causes (To Be Verified)

### Suspected Root Causes:

1. **Target Scaling Issue** (HIGH PROBABILITY)
   - Line 58 in dataset.py: `all_targets = np.array(self.targets).reshape(-1, 1)`
   - Then: `self.target_scaler.fit(all_targets)`
   - **POTENTIAL BUG:** Targets may be scaled incorrectly
   - **IMPACT:** Model learns on wrong scale, predictions meaningless

2. **Model Capacity Too Low** (MEDIUM PROBABILITY)
   - Current: 32 hidden units, 4.5K parameters
   - May be insufficient for volatility patterns
   - **IMPACT:** Model underfits, cannot learn complex dynamics

3. **Learning Rate Too Low** (MEDIUM PROBABILITY)
   - Current: lr = 0.001
   - For scaled data (mean~0, std~1), may be too conservative
   - **IMPACT:** Slow convergence, gets stuck in local minima

4. **Feature Insufficiency** (LOW PROBABILITY)
   - Only using Parkinson volatility (1 feature)
   - Missing HAR features, volume, returns
   - **IMPACT:** Model lacks predictive information

---

## 🛠️ Debugging Actions Taken

### 1. Created Comprehensive Debug Script

**File:** `src/lstm_baseline/debug_training_failure.py`

**Checks performed:**
1. ✅ Data quality and statistics
2. ✅ Model architecture and parameter count
3. ✅ Training configuration
4. ✅ One epoch simulation
5. ✅ Evaluation metrics verification

### 2. Key Debug Points

```python
# Check target scaling
print(f"Targets scaled mean: {targets_scaled.mean():.6f}")
print(f"Targets original mean: {targets_original.mean():.6f}")

# Check model predictions
print(f"Predictions mean: {predictions.mean():.6f}")
print(f"Actuals mean: {actuals.mean():.6f}")

# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} grad norm: {param.grad.norm():.6f}")
```

---

## 📋 Prevention Strategy - How to Avoid This

### For Future LSTM Projects:

#### 1. **Pre-Training Validation Checklist**

```python
# BEFORE training starts, verify:

def pre_training_validation(dataset, model):
    """Validate before expensive training."""

    checks = []

    # 1. Data range check
    sample_targets = [dataset[i][1].item() for i in range(min(100, len(dataset)))]
    if np.std(sample_targets) < 1e-6:
        checks.append("WARNING: Targets have no variance!")

    # 2. Model capacity check
    total_params = sum(p.numel() for p in model.parameters())
    if total_params < 10000:
        checks.append("WARNING: Model capacity < 10K params")

    # 3. Learning rate sanity check
    # (Check if appropriate for data scale)

    # 4. Forward pass test
    test_output = model(torch.randn(1, 22, 1))
    if torch.isnan(test_output).any():
        checks.append("ERROR: Model outputs NaN!")

    return checks
```

#### 2. **Training Monitoring Requirements**

```python
# DURING training, monitor EVERY epoch:

def training_monitor(epoch, train_loss, val_loss, model, scaler):
    """Monitor training progress."""

    # 1. Loss improvement check
    if epoch > 5 and (train_losses[-5] - train_losses[-1]) < 0.01:
        print("WARNING: Loss not improving!")

    # 2. Overfitting check
    if val_loss > train_loss * 1.2:
        print("WARNING: Potential overfitting!")

    # 3. Gradient check
    if grad_norm < 1e-5:
        print("WARNING: Gradients vanishing!")

    # 4. Prediction sanity check
    sample_preds = model(sample_X)
    if sample_preds.mean() < 0:
        print("WARNING: Negative predictions!")
```

#### 3. **Post-Training Validation**

```python
# AFTER training, verify:

def post_training_validation(actuals, predictions, metrics):
    """Validate model is actually useful."""

    issues = []

    # 1. QLIKE check
    if metrics['QLIKE'] > 0.20:
        issues.append(f"CRITICAL: QLIKE {metrics['QLIKE']:.2f} > 0.20")

    # 2. Directional accuracy check
    if metrics['Directional_Acc'] < 0.55:
        issues.append(f"CRITICAL: Dir Acc {metrics['Directional_Acc']:.2%} < 55%")

    # 3. Mean prediction check
    if abs(predictions.mean() - actuals.mean()) < 0.01 * actuals.mean():
        issues.append("WARNING: Model predicting mean, not learning")

    # 4. Prediction range check
    if predictions.min() < 0:
        issues.append("WARNING: Negative predictions present")

    return issues
```

---

## 🔄 Process Improvement - ML/DS Common Rules Update

### Rule to Add: Training Validation Rule

**File to update:** `D:\bmad-projects\ml-ds-common-rules\COMMON_RULES.md`

**New Section:** Under "Research Best Practices"

```markdown
### Training Validation (MANDATORY)

**Pre-Flight Checks:**
1. ✅ Data variance check - targets must have variance
2. ✅ Scaling verification - inputs and targets scaled correctly
3. ✅ Forward pass test - model outputs valid values
4. ✅ Gradient flow test - gradients not vanishing/exploding

**During Training:**
1. ✅ Monitor loss decrease every epoch
2. ✅ Check train/val gap (overfitting detection)
3. ✅ Verify gradient norms
4. ✅ Sample prediction sanity check

**Post-Training:**
1. ✅ QLIKE < target threshold
2. ✅ Directional accuracy > random baseline
3. ✅ Predictions not equal to mean
4. ✅ Prediction range validation

**Early Stopping Criteria:**
- If loss doesn't decrease for 5 epochs → STOP and debug
- If gradients < 1e-5 for 3 epochs → STOP (vanishing gradient)
- If QLIKE > 0.30 after 10 epochs → STOP and reconfigure
```

---

## 🎓 Key Takeaways

### For This Project:

1. **Target scaling is critical** - Double-check inverse-transform
2. **Model capacity matters** - 32 units likely too small
3. **Learning rate tuning** - 0.001 may be too low for scaled data
4. **Comprehensive debugging** - Need detailed logging at each step

### For All ML Projects:

1. **NEVER train without pre-flight validation**
2. **ALWAYS monitor gradients during training**
3. **ALWAYS verify predictions are reasonable scale**
4. **ALWAYS check if model beats simple baseline**
5. **NEVER assume training will work - verify early**

### For Agent Self-Improvement:

1. **Add validation checks to common rules**
2. **Create debug templates for common failures**
3. **Document lessons learned after each failure**
4. **Update checklists based on new findings**
5. **Share learnings across projects via ml-ds-common-rules**

---

## 📞 Action Items

### Immediate (For This Project):

1. ✅ Run debug script: `python -m src.lstm_baseline.debug_training_failure`
2. ⏳ Identify exact root cause from debug output
3. ⏳ Apply fixes (scaling, capacity, learning rate)
4. ⏳ Re-train and verify metrics improve

### Short-term (For Common Rules):

1. ⏳ Add "Training Validation" section to COMMON_RULES.md
2. ⏳ Create validation checklist template
3. ⏳ Add gradient monitoring utilities

### Long-term (For All Projects):

1. ⏳ Integrate pre-flight validation into all training scripts
2. ⏳ Create automated failure detection
3. ⏳ Build lesson learned database

---

## 📚 References

- **Debug Script:** `src/lstm_baseline/debug_training_failure.py`
- **Training Script:** `src/lstm_baseline/train.py`
- **Dataset:** `src/lstm_baseline/dataset.py`
- **Model:** `src/lstm_baseline/model.py`

---

**Status:** 🟡 Debugging in progress
**Next Step:** Run debug script to confirm root cause
**Target Resolution:** QLIKE < 0.20, Dir Acc > 55%

---

**Last Updated:** 2026-06-17
**Version:** 1.0 - Initial documentation
