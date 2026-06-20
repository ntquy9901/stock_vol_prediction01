# Model Comparison Report - After Learning Rate Fix

**Date:** 2026-06-19 18:30
**Training Time:** 18:02-18:29
**Status:** ⚠️ LEARNING RATE FIX NOT SUFFICIENT
**Learning Rate:** 0.001 (10× faster than before)

---

## 📊 CONFIGURATION USED (All Models)

```python
'hidden_size': 128,        # Increased: 64 → 128
'learning_rate': 0.001,     # FIXED: 0.0001 → 0.001
'batch_size': 32,
'seq_length': 22,
'weight_decay': 1e-6,       # Reduced: 1e-5 → 1e-6
'dropout': 0.1,             # Reduced: 0.2 → 0.1
'num_layers': 3             # Increased: 2 → 3 (except Simple LSTM: 1)
```

**Early Stopping:**
- `patience = 10` (reduced from 20)
- `min_epochs = 15` (new feature to prevent premature stopping)

---

## 🎯 TEST METRICS COMPARISON

### **Primary Metric: Directional Accuracy**

| Model | Val Dir Acc | Test Dir Acc | Test RMSE | Test QLIKE | Test R² | Status |
|-------|-------------|--------------|-----------|------------|---------|--------|
| **HAR-R Linear** | - | **51.53%** ✅ | **0.000513** ✅ | 1.298 | 0.105 | **Baseline** |
| **Simple LSTM** | 49.05% | 48.11% ❌ | 0.000554 ❌ | 3.040 ❌ | 0.106 | Failed |
| **LSTM-HAR** | 49.08% | 48.01% ❌ | 0.000555 ❌ | 0.632 ❌ | 0.108 | Failed |
| **Enhanced LSTM-HAR** | 48.20% | 48.32% ❌ | 0.000557 ❌ | 0.639 ❌ | 0.101 | Failed |

### **Key Finding:**
❌ **ALL LSTM models still underperform HAR-R baseline** by 3-4% in directional accuracy

---

## ⚠️ CRITICAL FINDING

### **ALL LSTM Models STILL Underperform Baseline**

**Directional Accuracy (Primary Target):**
- ❌ Target: >55%
- ❌ HAR-R Baseline: **51.53%**
- ❌ All LSTM models: **48.01-48.32%** (3-4% worse than baseline!)

**RMSE (Secondary Target):**
- ❌ Target: <0.00050
- ❌ HAR-R Baseline: **0.000513**
- ❌ All LSTM models: **0.000554-0.000557** (8% worse than baseline!)

---

## 🔍 BEST EPOCH ANALYSIS

### **Learning Rate Fix IMPACT Assessment**

| Model | Before Fix (LR=0.0001) | After Fix (LR=0.001) | Improvement |
|-------|------------------------|----------------------|-------------|
| **Simple LSTM** | Epoch 6 ❌ | **Epoch 23** ✅ | +17 epochs (+283%) |
| **LSTM-HAR** | Epoch 4 ❌ | **Epoch 17** ✅ | +13 epochs (+325%) |
| **Enhanced LSTM-HAR** | Epoch 3 ❌ | **Epoch 14** ✅ | +11 epochs (+367%) |

### **Analysis:**

✅ **Learning Rate Fix WORKED!**
- Models now train longer (best epochs 14-23 vs 3-6 before)
- Successfully prevented premature stopping
- Models have time to explore solution space
- Minimum epochs constraint (15) prevented early stopping

❌ **BUT Performance Still Worse Than Linear Baseline**
- Despite longer training, LSTM models still underperform
- Suggests fundamental architectural or data issues
- Not just a hyperparameter tuning problem

---

## 📈 DETAILED METRICS COMPARISON

### **Test Metrics (All 6 Metrics):**

| Metric | HAR-R Linear | Simple LSTM | LSTM-HAR | Enhanced LSTM-HAR | Winner |
|--------|--------------|-------------|----------|-------------------|--------|
| **MSE** | 2.63e-07 | 3.07e-07 | 3.07e-07 | 3.10e-07 | **HAR-R** ✅ |
| **RMSE** | 0.000513 | 0.000554 | 0.000555 | 0.000557 | **HAR-R** ✅ |
| **MAE** | 0.000257 | 0.000258 | 0.000263 | 0.000258 | **HAR-R** ✅ |
| **R²** | 0.105 | 0.106 | 0.108 | 0.101 | **LSTM-HAR** |
| **QLIKE** | 1.298 | 3.040 | 0.632 | 0.639 | **LSTM-HAR** |
| **Dir Acc** | **51.53%** | 48.11% | 48.01% | 48.32% | **HAR-R** ✅ |

### **Validation Metrics:**

| Metric | Simple LSTM | LSTM-HAR | Enhanced LSTM-HAR |
|--------|-------------|----------|-------------------|
| **MSE** | 2.13e-07 | 2.13e-07 | 2.16e-07 |
| **RMSE** | 0.000462 | 0.000461 | 0.000464 |
| **MAE** | 0.000262 | 0.000267 | 0.000260 |
| **R²** | 0.112 | 0.115 | 0.104 |
| **QLIKE** | 2.497 | 0.560 | 0.571 |
| **Dir Acc** | 49.05% | 49.08% | 48.20% |

---

## 🔄 VALIDATION vs TEST COMPARISON

### **Overfitting Check:**

| Model | Val RMSE | Test RMSE | Diff | Val Dir Acc | Test Dir Acc | Diff |
|-------|----------|-----------|------|-------------|--------------|------|
| Simple LSTM | 0.000462 | 0.000554 | +0.000092 | 49.05% | 48.11% | **-0.94%** |
| LSTM-HAR | 0.000461 | 0.000555 | +0.000094 | 49.08% | 48.01% | **-1.06%** |
| Enhanced LSTM-HAR | 0.000464 | 0.000557 | +0.000092 | 48.20% | 48.32% | +0.12% |

### **Analysis:**
✅ **No Severe Overfitting**
- Test performance similar to validation
- Small degradation in RMSE (~0.00009)
- Dir Acc stable within ±1%

✅ **Good Generalization**
- Models not memorizing training data
- **Problem is UNDERFITTING, not overfitting**
- Models not learning effective patterns

---

## 🚨 ROOT CAUSE ANALYSIS

### **Why Do LSTM Models Underperform?**

**1. ❌ Model Capacity Issues (UNLIKELY)**
- Current: 128 hidden, 3 layers (204K params for LSTM-HAR)
- This should be sufficient for the task
- Increasing further may lead to overfitting

**2. ✅ Learning Rate Issues (FIXED)**
- ✅ Fixed: Learning rate too low (0.0001 → 0.001)
- ✅ Result: Best epochs improved (3-6 → 14-23)
- ❌ But: Still not beating baseline

**3. ⚠️ Feature Engineering Issues (VERY LIKELY)**
- HAR-R uses 3 HAR features: daily, weekly, monthly
- LSTM-HAR uses same 3 features
- Enhanced LSTM-HAR uses: raw, weekly, monthly (removed redundant daily)
- **Problem:** LSTM models not leveraging features effectively
- **Possible:** Linear combinations of HAR features already optimal

**4. ⚠️ Data Issues (POSSIBLE)**
- Parkinson volatility may be too noisy
- 5-day horizon may be too difficult for deep learning
- Temporal split may have data quality issues
- Possible non-stationarity in data

**5. ⚠️ Architecture Issues (POSSIBLE)**
- LSTM may not be suitable for this specific task
- May need different architecture (Transformer, TCN)
- May need attention mechanisms
- Simple LSTM may be too archaic (2026)

**6. ⚠️ Training Issues (POSSIBLE)**
- May need more epochs (>100)
- May need different optimizer (AdamW, SGD with momentum)
- May need learning rate scheduling (cosine, warmup)
- May need different loss function (QLIKE instead of MSE)

---

## 💡 POTENTIAL SOLUTIONS

### **Option 1: Feature Engineering (HIGH PRIORITY) ⭐**

**Add Technical Indicators:**
```python
# Momentum indicators
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Williams %R

# Trend indicators
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Parabolic SAR

# Volatility indicators
- ATR (Average True Range)
- Bollinger Bands
- Keltner Channels

# Volume indicators
- On-Balance Volume (OBV)
- Volume Rate of Change
- Money Flow Index (MFI)
```

**Expected Impact:**
- More informative features
- Better signal-to-noise ratio
- May allow LSTM to learn meaningful patterns
- **Potential improvement: +5-10% Dir Acc**

### **Option 2: Architecture Change (HIGH PRIORITY) ⭐**

**Transformer Architecture:**
```python
# Multi-head attention
- Better long-range dependencies
- Parallel processing (faster training)
- State-of-the-art for time series (2026)

# Expected improvement: +3-7% Dir Acc
```

**LSTM-GAT Hybrid (Already Designed):**
```python
# Captures cross-stock correlations
- Spatial relationships (stock-to-stock)
- Temporal patterns (time-to-time)
- Graph attention mechanisms

# Target metrics:
- RMSE: < 0.15 (vs current 0.55)
- Dir Acc: >75% (vs current 48%)

# See: docs/project/LSTM_GAT_ARCHITECTURE.md
```

### **Option 3: Hyperparameter Tuning (MEDIUM PRIORITY)**

**Systematic Search with Optuna:**
```python
# Wider ranges
- learning_rate: [1e-5, 1e-2] (log scale)
- hidden_size: [32, 64, 128, 256, 512]
- dropout: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
- weight_decay: [1e-7, 1e-6, 1e-5, 1e-4]
- batch_size: [16, 32, 64, 128]

# Expected improvement: +1-3% Dir Acc
```

### **Option 4: Training Strategy (LOW PRIORITY)**

**Longer Training:**
```python
# Current
num_epochs = 100

# Try
num_epochs = 200-300
patience = 5 (reduce)
min_epochs = 30 (increase)

# Expected improvement: +0-2% Dir Acc
```

**Learning Rate Scheduling:**
```python
# Cosine annealing
- Warmup for first 10 epochs
- Cosine decay to 1e-6
- Better convergence

# Expected improvement: +0-1% Dir Acc
```

### **Option 5: Ensemble Methods (EXPERIMENTAL)**

**Model Stacking:**
```python
# Combine predictions
- HAR-R (linear baseline)
- LSTM-HAR (non-linear)
- Simple LSTM (different features)

# Methods
- Simple average
- Weighted average (learned weights)
- Stacking (meta-learner)

# Expected improvement: +2-5% Dir Acc
```

---

## 📊 PERFORMANCE SUMMARY

### **Targets Achievement:**

| Target | Goal | HAR-R | Simple LSTM | LSTM-HAR | Enhanced LSTM-HAR |
|--------|------|-------|-------------|----------|-------------------|
| **Dir Acc** | >55% | 51.53% ❌ | 48.11% ❌ | 48.01% ❌ | 48.32% ❌ |
| **RMSE** | <0.00050 | 0.000513 ❌ | 0.000554 ❌ | 0.000555 ❌ | 0.000557 ❌ |
| **QLIKE** | <0.60 | 1.298 ❌ | 3.040 ❌ | 0.632 ❌ | 0.639 ❌ |
| **Beat Baseline** | YES | N/A | NO ❌ | NO ❌ | NO ❌ |

### **Status:**
❌ **NONE of the targets achieved by LSTM models**

### **Learning Rate Fix Impact:**

| Aspect | Before Fix | After Fix | Status |
|--------|------------|-----------|--------|
| **Best Epochs** | 3-6 | 14-23 | ✅ Improved |
| **Training Time** | Same | Same | ✅ OK |
| **Dir Acc** | 48-49% | 48-49% | ❌ No change |
| **RMSE** | 0.000553-0.000555 | 0.000554-0.000557 | ❌ Slightly worse |

**Conclusion:** Learning rate fix allowed models to train longer, but did NOT improve final performance.

---

## 🎯 RECOMMENDATIONS

### **Immediate Actions (Priority 1):**

**1. Add More Features** (HIGHEST PRIORITY) ⭐⭐⭐
```python
# Technical indicators (most promising)
- RSI, MACD, Bollinger Bands, ATR
- Volume features
- Macro variables (VNI index, interest rates)

# Expected timeline: 1-2 days
# Expected improvement: +5-10% Dir Acc
# Risk: Low (well-established techniques)
```

**2. Try Transformer Architecture** (HIGH PRIORITY) ⭐⭐
```python
# Replace LSTM with Transformer
- Multi-head attention
- Positional encoding
- Faster training (parallel)

# Expected timeline: 2-3 days
# Expected improvement: +3-7% Dir Acc
# Risk: Medium (new architecture)
```

### **Secondary Actions (Priority 2):**

**3. Implement LSTM-GAT Hybrid** (MEDIUM PRIORITY) ⭐
```python
# Already designed (see LSTM_GAT_ARCHITECTURE.md)
- Captures cross-stock correlations
- More complex implementation

# Expected timeline: 5-7 days
# Expected improvement: +10-15% Dir Acc (potentially)
# Risk: High (complex, untested)
```

**4. Hyperparameter Tuning with Optuna** (LOWER PRIORITY)
```python
# Systematic search
- Wider ranges
- More trials (100-200)

# Expected timeline: 3-5 days
# Expected improvement: +1-3% Dir Acc
# Risk: Low (safe optimization)
```

### **Experimental Actions (Priority 3):**

**5. Ensemble Methods**
```python
# Combine models
- HAR-R + LSTM-HAR
- Simple average first
- Then learn weights

# Expected timeline: 1 day
# Expected improvement: +2-5% Dir Acc
# Risk: Low
```

---

## 📈 LEARNING CURVES ANALYSIS

### **Best Epochs (After Fix):**

| Model | Best Epoch | Total Epochs | % of Max | Training Time |
|-------|------------|--------------|----------|---------------|
| Simple LSTM | 23 | 100 | 23% | ~15 min |
| LSTM-HAR | 17 | 100 | 17% | ~18 min |
| Enhanced LSTM-HAR | 14 | 100 | 14% | ~18 min |

### **Interpretation:**
✅ **Learning Rate Fix Successful**
- Models now train longer (not stopping at epoch 3-6)
- Allowed proper convergence
- Best epochs in reasonable range (14-23)
- Minimum epochs constraint working

❌ **But Performance Still Suboptimal**
- Despite longer training, can't beat linear baseline
- Suggests fundamental limitations
- Not just a hyperparameter issue

### **Training Behavior:**

**Simple LSTM (Best: Epoch 23/100):**
- Trained 23% of max epochs
- Validation loss plateaued after epoch 23
- Early stopping triggered at epoch 33 (patience 10)

**LSTM-HAR (Best: Epoch 17/100):**
- Trained 17% of max epochs
- Validation loss plateaued earlier
- Early stopping triggered at epoch 27

**Enhanced LSTM-HAR (Best: Epoch 14/100):**
- Trained 14% of max epochs
- Earliest plateau
- Early stopping triggered at epoch 24

**Pattern:** All models plateau around 14-23 epochs, suggesting:
- Models have learned all they can from current features
- More training won't help
- Need better features or different architecture

---

## 🚀 NEXT STEPS - ACTION PLAN

### **Phase 1: Feature Engineering (Recommended)**

**Timeline:** 1-2 days

**Tasks:**
```python
1. Add technical indicators
   - RSI (14 periods)
   - MACD (12, 26, 9)
   - Bollinger Bands (20, 2)
   - ATR (14 periods)

2. Add volume features
   - Volume volatility
   - Volume changes
   - OBV (On-Balance Volume)

3. Re-train all models with new features
   - Simple LSTM (raw + new features)
   - LSTM-HAR (HAR + new features)
   - Enhanced LSTM-HAR (current + new features)

4. Compare with baseline
   - Check if Dir Acc >55%
   - Check if RMSE <0.00050
   - Check if beat HAR-R
```

**Expected Outcome:**
- Dir Acc: 48% → 55-60% (+7-12%)
- RMSE: 0.00055 → 0.00048-0.00050 (+10%)
- **May finally beat HAR-R baseline**

### **Phase 2: Transformer Architecture (If Phase 1 insufficient)**

**Timeline:** 2-3 days

**Tasks:**
```python
1. Implement Transformer model
   - Multi-head attention (4 heads)
   - Positional encoding
   - 4-6 layers
   - 64-128 hidden size

2. Train with current features
   - Same training setup
   - Compare with LSTM

3. Train with new features (if Phase 1 successful)
   - Best features from Phase 1
   - Optimize for Transformer
```

**Expected Outcome:**
- Dir Acc: 48% → 53-56% (+5-8%)
- RMSE: 0.00055 → 0.00048-0.00052 (+5-12%)
- Faster training (parallel processing)

### **Phase 3: LSTM-GAT Hybrid (If Phase 1-2 insufficient)**

**Timeline:** 5-7 days

**Tasks:**
```python
1. Implement graph construction
   - Correlation-based edges
   - Volatility spillover edges

2. Implement GAT layers
   - Multi-head attention (4-8 heads)
   - Graph convolution

3. Combine with LSTM encoder
   - Per-stock LSTM
   - Graph attention
   - Fusion layer

4. Train and evaluate
   - Current features
   - Enhanced features (if Phase 1 successful)
```

**Expected Outcome:**
- Dir Acc: 48% → 60-75% (+12-27%)
- RMSE: 0.00055 → 0.00015-0.00040 (+27-73%)
- **Highest potential but most complex**

---

## 📝 CONCLUSION

**Status:** ⚠️ **Learning Rate Fix Not Sufficient**

**What Worked:**
- ✅ Learning rate fix (0.0001 → 0.001) successful
- ✅ Best epochs improved (3-6 → 14-23)
- ✅ Minimum epochs constraint working (no premature stopping)
- ✅ No severe overfitting (good generalization)

**What Didn't Work:**
- ❌ Still can't beat HAR-R baseline (51.53%)
- ❌ Dir Acc stuck at ~48% (target: >55%)
- ❌ RMSE stuck at ~0.00055 (target: <0.00050)
- ❌ All LSTM models underperform simple linear baseline

**Root Cause:**
- **NOT hyperparameter issue** (learning rate fixed, no improvement)
- **NOT overfitting issue** (generalization is good)
- **LIKELY feature quality issue** (need more informative features)
- **OR architecture issue** (LSTM may not be suitable for this task)

**Recommendation:**
1. **Add more features** (technical indicators, volume features) - **HIGHEST PRIORITY**
2. **Try Transformer architecture** (attention-based, state-of-the-art) - **HIGH PRIORITY**
3. **If still insufficient, implement LSTM-GAT hybrid** (most promising but complex)

**Expected Timeline:**
- Phase 1 (features): 1-2 days → **May solve the problem**
- Phase 2 (Transformer): 2-3 days → **If Phase 1 insufficient**
- Phase 3 (LSTM-GAT): 5-7 days → **If Phase 1-2 insufficient**

**Confidence Level:**
- Phase 1: 70% chance of success
- Phase 2: 50% chance of success
- Phase 3: 80% chance of success (but highest complexity)

---

**Report Generated:** 2026-06-19 18:30
**Next Training Run:** With technical indicators (Phase 1)
**Overall Status:** ⚠️ **IN PROGRESS - Need Feature Engineering**

---

*Author: Stock Volatility Prediction Team*
*Last Updated: 2026-06-19 18:30*
