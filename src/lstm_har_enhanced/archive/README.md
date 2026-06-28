# Archived Training Scripts

This directory contains deprecated training scripts that have been replaced by improved versions.

---

## 📁 train_with_validation_DEPRECATED_2026-06-20.py

**Status:** ⚠️ **DEPRECATED - DO NOT USE**

**Reason for Archiving:**
- Severe overfitting detected in training results
- Test RMSE was 17.5x higher than validation RMSE (0.009943 vs 0.000569)
- Test R² was negative (-0.0472), indicating model worse than baseline
- Insufficient overfitting prevention techniques applied

**Issues Found:**
1. ❌ No gradient clipping (mandatory for RNN)
2. ❌ No FC dropout layer
3. ❌ No layer normalization
4. ❌ Weak regularization configuration
5. ❌ Limited monitoring during training

**Use Instead:**
- ✅ `../train_with_overfitting_prevention.py` - Comprehensive overfitting prevention

**Replacement File Benefits:**
- ✅ All mandatory anti-overfitting techniques applied
- ✅ Gradient clipping (max_norm=1.0)
- ✅ FC dropout (0.3) + Layer normalization
- ✅ Comprehensive monitoring with automatic overfitting detection
- ✅ Standardized hyperparameters (70 epochs, 15 patience)
- ✅ Val-test gap reduced by 99% (0.00937 → 0.000094)

**Results Comparison:**
| Metric | Old (Deprecated) | New (Fixed) | Improvement |
|--------|------------------|-------------|-------------|
| Test RMSE | 0.009943 | 0.000557 | **94.4% better** |
| Test R² | -0.0472 | 0.098 | **+308%** |
| Val-Test Gap | 0.00937 (17.5x) | 0.000094 (1.2x) | **99% reduction** |

**Archived Date:** 2026-06-21
**Deprecated Date:** 2026-06-20 (original file creation)

---

## 📚 Documentation

For detailed information about the overfitting issues and fixes, see:
- 📘 `D:\bmad-projects\stock_vol_prediction01\docs\OVERFITTING_PREVENTION_APPLIED.md`
- 📘 `D:\bmad-projects\stock_vol_prediction01\VN30_PERFORMANCE_REPORT.md` (Section: Overfitting Prevention)

---

## ⚠️ IMPORTANT NOTES

**DO NOT:**
- ❌ Use these archived scripts for training
- ❌ Reference these scripts in new code
- ❌ Copy code from these files

**DO:**
- ✅ Use `train_with_overfitting_prevention.py` for all training
- ✅ Apply mandatory overfitting prevention techniques
- ✅ Monitor val-test gaps during training
- ✅ Use standardized hyperparameters (70 epochs, 15 patience)

---

**Archive Purpose:** Keep for historical reference and comparison only.
**Maintenance:** No updates will be made to archived files.
