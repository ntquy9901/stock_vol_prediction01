# Archive Directory

This directory contains deprecated and archived code from the stock volatility prediction project.

---

## 📁 Archive Structure

```
archive/
├── data_leakage_scripts/     ← Scripts with CRITICAL data leakage bugs
│   ├── README.md             ← Detailed documentation
│   ├── train_all_baselines.py
│   ├── train_enhanced_lstm_har_vn30.py
│   └── train_enhanced_lstm_har_vn30_progress.py
│
└── [future archives]
```

---

## 🚨 Data Leakage Scripts Archive

**Location:** `archive/data_leakage_scripts/`

**Status:** 🔴 **CRITICAL - DO NOT USE**

**What Happened:**
These scripts used `torch.utils.data.random_split` for time series data, causing:
- Future data to leak into training set
- Inflated performance metrics
- 17.8x worse test performance in reality

**Impact:**
- Test RMSE: 0.009943 (with leakage) vs 0.000557 (correct) → **17.8x worse**
- Val-Test Gap: 0.00937 (with leakage) vs 0.000094 (correct) → **99x larger**

**For Detailed Information:**
See: `archive/data_leakage_scripts/README.md`

---

## ✅ Safe to Use (Not Archived)

These files are CORRECT and use proper temporal split:

### In Project Root:
- ✅ `train_all_models_vn30.py` - Calls correct scripts
- ✅ `train_all_with_validation.py` - Calls correct scripts

### In Source Directories:
- ✅ `src/lstm_har_enhanced/train_with_overfitting_prevention.py` - Best choice
- ✅ `src/lstm_har_enhanced/archive/train_with_validation_DEPRECATED_2026-06-20.py` - Old version
- ✅ `src/lstm_baseline/train_with_validation.py` - Temporal split
- ✅ `src/lstm_har_baseline/train_with_validation.py` - Temporal split

---

## 📚 Key Rules

### Time Series Data Splitting:

**❌ NEVER:**
```python
# Random split shuffles time → data leakage
train, val, test = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
```

**✅ ALWAYS:**
```python
# Temporal split maintains chronology
from src.common.temporal_split import TemporalSplitter
splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
train, val, test = splitter.create_dataloaders(dataset)
```

---

## 🎯 Purpose of Archive

1. **Educational:** Examples of what NOT to do
2. **Historical:** Track project evolution
3. **Comparison:** Show before/after fixes
4. **Prevention:** Prevent future mistakes

---

**Last Updated:** 2026-06-21
**Maintained by:** Stock Volatility Prediction Team
