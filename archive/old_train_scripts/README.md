# Archived Training Scripts (No Validation Split)

**Date Archived:** 2026-06-19
**Reason:** Replaced by train_with_validation.py files with proper 3-way temporal split

---

## 🗑️ Why These Files Were Archived

### **Critical Issue: Data Leakage**

These old training scripts used **80/20 random split** instead of **70/15/15 temporal split**:

```python
# ❌ OLD (WRONG) - Random split causes data leakage
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
```

**Problem:**
- Random split allows future data in training
- Causes **data leakage**
- Overestimates model performance
- Not suitable for time series forecasting

---

## 📁 Files Archived

### **1. train_enhanced.py**
- **Model:** Enhanced LSTM-HAR
- **Features:** Raw + HAR (weekly, monthly)
- **Split:** 80/20 random (WRONG)
- **Replaced by:** `src/lstm_har_enhanced/train_with_validation.py`

### **2. lstm_har_baseline_train.py**
- **Model:** LSTM-HAR Baseline
- **Features:** HAR (daily, weekly, monthly)
- **Split:** 80/20 random (WRONG)
- **Replaced by:** `src/lstm_har_baseline/train_with_validation.py`

### **3. lstm_baseline_train.py**
- **Model:** Simple LSTM
- **Features:** Raw Parkinson volatility
- **Split:** 80/20 random (WRONG)
- **Replaced by:** `src/lstm_baseline/train_with_validation.py`

---

## ✅ New Training Scripts (Correct)

All models now use **train_with_validation.py** with proper 3-way temporal split:

```python
# ✅ NEW (CORRECT) - Temporal split prevents data leakage
splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
train_loader, val_loader, test_loader = splitter.create_dataloaders(dataset)

# Proper temporal ordering:
# Train: 2006-2020 (70%)
# Val:   2020-2021 (15%) - for early stopping
# Test:  2021-2026 (15%) - final evaluation
```

**Benefits:**
- ✅ No data leakage
- ✅ Proper temporal ordering
- ✅ Validation set for early stopping
- ✅ Realistic performance estimation

---

## 📊 Performance Comparison

### **Before (Random Split - Overestimated):**
- Models appeared to perform better
- But metrics were biased due to data leakage
- Not representative of real performance

### **After (Temporal Split - Realistic):**
- Lower but more realistic metrics
- Proper evaluation of model's ability
- Suitable for production deployment

---

## 🔗 Related Documentation

- **Temporal Split Methodology:** `docs/project/TEMPORAL_SPLIT_EVALUATION.md`
- **Implementation:** `src/common/temporal_split.py`
- **New Training Scripts:** See `src/*/train_with_validation.py`

---

## ⚠️ DO NOT USE THESE ARCHIVED FILES

These files are kept for **reference only**. Do not use them for training as they produce biased results due to data leakage.

**Always use the new `train_with_validation.py` files instead.**

---

**Archived by:** Stock Volatility Prediction Team
**Date:** 2026-06-19
**Reason:** Data leakage in random split - replaced by proper temporal split
