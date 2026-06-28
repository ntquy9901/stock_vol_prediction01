# ⚠️ DATA LEAKAGE SCRIPTS - ARCHIVED

**Status:** 🔴 **CRITICAL DATA LEAKAGE - DO NOT USE**

**Archived Date:** 2026-06-21
**Reason:** All files in this archive contain **data leakage** bugs that violate time series principles

---

## 🚨 CRITICAL WARNING

**ALL FILES IN THIS ARCHIVE:**
- ❌ Use `torch.utils.data.random_split` for time series data
- ❌ Allow future data to leak into training set
- ❌ Produce OVERESTIMATED performance metrics
- ❌ Are NOT suitable for production use

**IMPACT OF DATA LEAKAGE:**
- Model performance is overestimated (inflated metrics)
- Real-world performance will be much worse
- Violates time series forecasting principles
- Future information contaminates past training

---

## 📁 Archived Files

### 1. train_enhanced_lstm_har_vn30_progress.py

**Data Leakage Location:** Lines 54-57
```python
# ❌ WRONG - Random split for time series
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
```

**Why It's Wrong:**
- Shuffles time series data chronology
- Future data appears in training set
- Past data appears in test set
- Model sees future during training → CHEATING

**Impact:**
- Validation metrics inflated
- Test metrics inflated
- Model fails in production

**Use Instead:**
```python
# ✅ CORRECT - Temporal split
from src.common.temporal_split import TemporalSplitter

splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
train_loader, val_loader, test_loader = splitter.create_dataloaders(
    dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=use_gpu
)
```

---

### 2. train_enhanced_lstm_har_vn30.py

**Data Leakage Location:** Lines 51-54
```python
# ❌ WRONG - Random split for time series
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
```

**Same Issues as Above.**

**Use Instead:**
```python
# ✅ CORRECT - Temporal split
from src.common.temporal_split import TemporalSplitter
```

---

### 3. train_all_baselines.py

**Data Leakage Location:** Calls deprecated script (Line 81)
```python
# ❌ WRONG - Calls deprecated script with data leakage
from src.lstm_har_enhanced.train_with_validation import train_enhanced_lstm_har_with_val
```

**Why It's Wrong:**
- Calls `src.lstm_har_enhanced.train_with_validation.py`
- That file has been archived (contains data leakage)
- Wrapper script inherits the data leakage bug

**Impact:**
- All models trained via this script have inflated metrics
- Not suitable for comparisons or production

**Use Instead:**
```python
# ✅ CORRECT - Call overfitting prevention script
from src.lstm_har_enhanced.train_with_overfitting_prevention import main

main(
    data_dir='data/processed/vn30_only',
    batch_size=32,
    num_epochs=70,
    patience=15
)
```

---

## 📊 Data Leakage Explained

### What Is Data Leakage?

**Data leakage** occurs when information from the future contaminates the training process, giving the model an unrealistic advantage.

### Example Timeline:

```
Time:    [Jan] [Feb] [Mar] [Apr] [May] [Jun] [Jul] [Aug] [Sep] [Oct]
Data:     D1    D2    D3    D4    D5    D6    D7    D8    D9    D10

✅ CORRECT (Temporal Split):
Train:    [D1----D7]        ← Train on past data
Val:                [D8--D9] ← Validate on recent data
Test:                    [D10]← Test on future data

❌ WRONG (Random Split):
Train:    [D1][D3][D5][D7][D9][D10] ← Train includes FUTURE (D10)
Val:      [D2][D8]               ← Val includes mixed time
Test:     [D4][D6]               ← Test includes PAST (D4, D6)
```

### Impact on Model Performance:

| Metric | With Leakage | Without Leakage | Reality |
|--------|-------------|------------------|---------|
| **Val RMSE** | 0.000569 | 0.000569 | Same |
| **Test RMSE** | **0.009943** | **0.000557** | **17.8x WORSE with leakage!** |
| **Val-Test Gap** | 0.00937 | 0.000094 | **99x LARGER with leakage** |
| **Dir Acc** | 48.13% | 48.56% | Slightly worse with leakage |

**Key Finding:** Models with data leakage look good during validation but FAIL in production.

---

## 🔧 How to Fix Data Leakage

### For Time Series Data:

**ALWAYS use temporal (chronological) split:**

```python
# ✅ CORRECT - Temporal split
from src.common.temporal_split import TemporalSplitter

splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
train_loader, val_loader, test_loader = splitter.create_dataloaders(
    dataset, batch_size=32, num_workers=0, pin_memory=True
)

# Data is split chronologically:
# Train: [0% to 70%]    → Past data
# Val:   [70% to 85%]   → Recent data
# Test:  [85% to 100%]  → Future data
```

**NEVER use random split:**

```python
# ❌ WRONG - Random split (DATA LEAKAGE!)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)
# This SHUFFLES time → future leaks into past
```

---

## 📚 Correct Training Scripts

**Use these scripts instead:**

### For Enhanced LSTM-HAR:
- ✅ `src/lstm_har_enhanced/train_with_overfitting_prevention.py`
  - Temporal split ✅
  - Overfitting prevention ✅
  - Comprehensive monitoring ✅

### For Simple LSTM:
- ✅ `src/lstm_baseline/train_with_validation.py`
  - Temporal split ✅
  - Early stopping ✅

### For LSTM-HAR:
- ✅ `src/lstm_har_baseline/train_with_validation.py`
  - Temporal split ✅
  - Early stopping ✅

### For Training Multiple Models:
- ✅ `train_all_with_validation.py` (in project root)
  - Calls correct scripts ✅
  - All models use temporal split ✅

- ✅ `train_all_models_vn30.py` (in project root)
  - Calls correct scripts ✅
  - VN30-focused ✅

---

## 🎯 Lessons Learned

### 1. Time Series Data Splitting
- ❌ **NEVER use random_split** for time series
- ✅ **ALWAYS use temporal split** (chronological order)
- ✅ Training data must come BEFORE test data chronologically

### 2. Validation is Critical
- Always check val-test metrics gap
- Gap > 0.05 indicates overfitting or data leakage
- Dir Acc < 55% indicates model issues

### 3. Documentation Importance
- Document WHY temporal split is required
- Add warnings in code comments
- Create comprehensive guides

### 4. Code Review Process
- Adversarial reviews found these bugs
- 40 bugs fixed across 3 reviews (TimesFM project)
- See: `docs/LESSONS_LEARNED_TIMESFM_ADVERSARIAL_REVIEWS.md`

---

## 📖 References

**Internal Documentation:**
- 📘 `CLAUDE.md` Section 3.A - Temporal Data Splitting (MANDATORY)
- 📘 `docs/project/TEMPORAL_SPLIT_EVALUATION.md` - Complete guide
- 📘 `docs/project/OVERFITTING_PREVENTION_APPLIED.md` - Overfitting prevention
- 📘 `VN30_PERFORMANCE_REPORT.md` - Performance analysis

**External References:**
- ml-ds-common-rules package: `D:\bmad-projects\ml-ds-common-rules`
- Time series cross-validation: sklearn TimeSeriesSplit

---

## ⚠️ IMPORTANT REMINDERS

**DO NOT:**
- ❌ Use these archived scripts for ANY purpose
- ❌ Copy code from these files
- ❌ Reference these scripts in new code
- ❌ Assume random_split is safe for time series

**DO:**
- ✅ Always use temporal split for time series
- ✅ Use scripts in `src/*/train_with_validation.py`
- ✅ Check val-test metrics gaps
- ✅ Review code for data leakage bugs
- ✅ Document temporal split requirements

---

## 🔍 Detection Checklist

**Before using any time series script:**

- [ ] Does it use `TemporalSplitter` or `create_temporal_dataloaders`?
- [ ] Does it AVOID `random_split`?
- [ ] Does it train on chronologically EARLIER data?
- [ ] Does it validate/test on chronologically LATER data?
- [ ] Are val-test gaps within acceptable thresholds (< 0.05)?

**If NO to any → DO NOT USE THE SCRIPT**

---

**Archive Purpose:** Keep for educational purposes only (what NOT to do)
**Maintenance:** No updates will be made to archived files
**Warning:** These files are kept as examples of bugs to avoid

---

**Created by:** Stock Volatility Prediction Team
**Date:** 2026-06-21
**Version:** 1.0 (Archive Documentation)
