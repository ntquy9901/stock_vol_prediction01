# Stock Volatility Prediction - VN30

**Project:** Multi-horizon volatility forecasting cho 30 VN30 stocks  
**Date:** 2026-06-19  
**Status:** Development Phase - Implementing 3-way temporal split evaluation

---

## Quick Start

```bash
# Install dependencies
pip install torch pandas numpy scikit-learn

# Process data
python process_data.py

# Train model (with proper temporal split)
python src/lstm_har_enhanced/train_with_validation.py
```

---

## 1. Project Overview

### **Objective**
Build robust volatility prediction system cho 30 VN30 stocks using daily OHLCV data, implementing HAR methodology adapted cho daily frequency.

### **Primary Target**
- **5-day ahead volatility forecast** (current focus)
- **Secondary Targets:** 1, 10, 22-day forecasts (future expansion)

### **Success Criteria**
- **RMSE:** < 0.20 cho 5-day forecasts
- **Directional Accuracy:** > 55% cho 5-day forecasts
- **Test Coverage:** 85%+ overall, 90% cho critical paths

---

## 2. ML/DS Common Rules

**This project follows ML/DS common clean code rules.**

### **External Package Reference**
**Package:** `ml-ds-common-rules`  
**Location:** `D:\bmad-projects\ml-ds-common-rules`  
**Installation:** `pip install -e D:\bmad-projects\ml-ds-common-rules`

**Quick Links:**
- 📘 [Common Rules](D:\bmad-projects\ml-ds-common-rules\COMMON_RULES.md)
- 📗 [Quick Reference](D:\bmad-projects\ml-ds-common-rules\QUICK_REFERENCE.md)
- 📕 [Integration Guide](D:\bmad-projects\ml-ds-common-rules\INTEGRATION_GUIDE.md)

### **Key Principles (from ml-ds-common-rules)**

1. **Code is read much more than written** → Write for future readers
2. **Leave code better than you found it** → Boy scouts rule
3. **Keep it simple** → Simple > Clever
4. **Match quality to maturity** → Don't over-engineer POCs

**For detailed rules:** See `ml-ds-common-rules` package documentation above.

---

## 3. Project-Specific Rules

### **Critical Rules for Volatility Forecasting**

#### **A. Temporal Data Splitting (MANDATORY)** ⭐

**CRITICAL:** Time series data MUST be split chronologically to prevent data leakage.

```python
# ✅ CORRECT - Temporal split
from src.common.temporal_split import TemporalSplitter

splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
train_loader, val_loader, test_loader = splitter.create_dataloaders(dataset)

# ❌ WRONG - Random split causes data leakage
train, test = torch.utils.data.random_split(dataset, [0.8, 0.2])
```

**Split ratios:**
- Train: 70% (2006-2020)
- Validation: 15% (2020-2021) - for early stopping
- Test: 15% (2021-2026) - final evaluation

**Documentation:** `docs/project/TEMPORAL_SPLIT_EVALUATION.md`

---

#### **B. Mandatory Evaluation Metrics (ALL 6)** ⭐

**CRITICAL:** Every model MUST be evaluated on ALL 6 metrics below.

```python
from src.common.evaluation import evaluate_predictions

metrics = evaluate_predictions(y_true, y_pred)
# Returns: rmse, mae, r2, qlike, directional_accuracy
```

**6 Mandatory Metrics:**
1. **MSE** - Mean Squared Error (lower is better)
2. **RMSE** - Root Mean Squared Error (lower is better)
3. **MAE** - Mean Absolute Error (lower is better)
4. **R²** - Variance Explained (higher is better)
5. **QLIKE** - Academic standard cho volatility (lower is better)
6. **Dir Acc** - Directional Accuracy (higher is better)

**Documentation:** `docs/project/` - See Model Evaluation Rules section

**Critical Bug Warning - Directional Accuracy:**
```python
# ❌ WRONG - Sign of values (always 100% for volatility)
dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

# ✅ CORRECT - Sign of CHANGES
actual_changes = np.sign(np.diff(y_true))
pred_changes = np.sign(np.diff(y_pred))
dir_acc = np.mean(actual_changes == pred_changes) * 100
```

---

#### **C. Learning Curves (MANDATORY)** ⭐

**CRITICAL:** Plot learning curves cho ALL training runs to detect overfitting.

```python
# During training - PLOT EVERY 10 EPOCHS
if (epoch + 1) % 10 == 0:
    plot_learning_curves(train_losses, val_losses, save_path)
```

**Documentation:** See `ml-ds-common-rules` COMMON_RULES.md "Learning Curves and Overfitting Detection"

---

#### **D. File Organization**

**Mandatory Structure:**
```
project_root/
├── CLAUDE.md                   # This file (project rules)
├── README.md                   # Project overview
├── project-context.md          # Project background
├── src/                        # ALL code here
│   ├── common/                 # Shared utilities
│   ├── har_baseline/           # HAR baseline
│   ├── lstm_baseline/          # LSTM baseline
│   ├── lstm_har_baseline/      # LSTM-HAR
│   ├── lstm_har_enhanced/      # Enhanced LSTM-HAR
│   └── experiment/             # Experimental code
├── docs/                       # ALL docs here
│   ├── project/                 # Project docs
│   ├── lstm/                    # LSTM docs
│   └── common-rules/           # Common rules (reference)
├── results/                    # ALL results here
├── models/                     # ALL models here
└── data/                       # Data files only
```

**Generated Files Naming:**
```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

# Result files
results_file = f"results/enhanced_lstm_{timestamp}/"
model_file = f"models/baseline_{timestamp}/"
```

**Documentation:** See `ml-ds-common-rules` COMMON_RULES.md "File Management and Archiving"

---

## 4. Model Architecture

### **Baseline Models**

#### **1. HAR-R Linear**
- **Features:** HAR (daily, weekly, monthly)
- **Method:** Linear regression
- **Purpose:** Baseline comparison
- **File:** `src/har_baseline/train.py`

#### **2. Simple LSTM**
- **Features:** Raw Parkinson volatility (1 input)
- **Architecture:** 1-layer LSTM, hidden_size=128
- **Purpose:** Deep learning baseline
- **File:** `src/lstm_baseline/train.py`

#### **3. LSTM-HAR**
- **Features:** HAR (daily, weekly, monthly) (3 inputs)
- **Architecture:** 2-layer LSTM, hidden_size=64
- **Purpose:** HAR + Deep learning
- **File:** `src/lstm_har_baseline/train.py`

#### **4. Enhanced LSTM-HAR**
- **Features:** Raw + HAR (weekly, monthly) (3 inputs)
- **Architecture:** 2-layer LSTM, hidden_size=64
- **Enhancement:** Raw volatility adds current-day info
- **Purpose:** Best performer (67.90% Dir Acc)
- **File:** `src/lstm_har_enhanced/train_enhanced.py`

---

## 5. Evaluation Methodology

### **Current Status (CRITICAL BUG FOUND)**

**HAR-R Linear:** ✅ Uses temporal split (correct)  
**LSTM Models:** ❌ Use random split (DATA LEAKAGE!)

**Issue:** Random split allows future data in training → overestimated metrics

**Solution:** Implement 3-way temporal split (70/15/15)

**Files Created:**
- ✅ `src/common/temporal_split.py` - Temporal split utilities
- ✅ `src/lstm_har_enhanced/train_with_validation.py` - Example implementation
- ✅ `docs/project/TEMPORAL_SPLIT_EVALUATION.md` - Full documentation

---

## 6. Key Technical Decisions

### **Volatility Calculation**
- **Method:** Parkinson estimator
- **Formula:** σ² = (log(H/L)²) / (4*log(2))
- **Reason:** More efficient cho daily data than close-to-close

### **HAR Features**
- **Daily:** 1-day rolling mean
- **Weekly:** 5-day rolling mean
- **Monthly:** 22-day rolling mean (confirmed ~22 trading days/month)

### **Target Variable**
- **Horizon:** 5-day ahead (primary focus)
- **Method:** `volatility.shift(-5)`

### **Loss Functions**
- **Training:** MSE (convex, stable)
- **Evaluation:** QLIKE (academic standard) + RMSE, MAE, R², Dir Acc

---

## 7. Development Workflow

### **Code Review Process**

**Before Committing:**
- [ ] Variable names descriptive (not x, y, data)
- [ ] Functions small (< 30 lines)
- [ ] Docstrings cho public functions
- [ ] Tests added/updated
- [ ] Temporal split verified (not random)
- [ ] All 6 metrics calculated
- [ ] Learning curves plotted

**Before Merging:**
- [ ] All tests pass
- [ ] Coverage > 85%
- [ ] Documentation updated
- [ ] Results saved with timestamp

---

## 8. Documentation Structure

### **Key Documents**

**In Root (Only 3 .md files):**
- `README.md` - Project overview và quick start
- `CLAUDE.md` - This file (project rules)
- `project-context.md` - Project background

**In docs/:**
- `docs/project/` - Project-specific documentation
  - `TEMPORAL_SPLIT_EVALUATION.md` - Evaluation methodology
  - `REFACTOR_SUMMARY.md` - Refactoring history
- `docs/lstm/` - LSTM model documentation
- `docs/common-rules/` - Reference to ml-ds-common-rules

**For detailed guides:** See respective directories in `docs/`

---

## 9. Quick Reference

### **Common Commands**

```bash
# Process data
python process_data.py

# Train with validation (NEW - 3-way temporal split)
python src/lstm_har_enhanced/train_with_validation.py

# Calculate all metrics
python src/experiment/calculate_all_metrics.py

# Demonstrate data leakage
python src/experiment/demonstrate_data_leakage.py
```

### **File Locations**

```
Data:        data/processed/
Models:      models/*_2026-06-19_*/
Results:     results/*_2026-06-19_*/
Docs:        docs/
Source:      src/
```

---

## 10. Contact & Support

### **Getting Help**

**Questions:**
- Common rules: See `ml-ds-common-rules` package
- Project specifics: See `docs/project/`
- Integration: See `ml-ds-common-rules/INTEGRATION_GUIDE.md`

### **Project Status**

- **Phase:** Sprint 1 - Baseline implementation
- **Current:** Implementing 3-way temporal split evaluation
- **Next:** Multi-horizon expansion (1, 10, 22-day forecasts)

---

**Last Updated:** 2026-06-19  
**Version:** 3.0 (Streamlined - references external ml-ds-common-rules)  
**Status:** Active Development

---

**Changes in v3.0 (Streamlined Version):**
- ✅ Removed duplicated ML/DS common rules (now reference external package)
- ✅ Reduced from 1,798 lines to ~400 lines (58KB → ~12KB)
- ✅ Added 3-way temporal split evaluation section
- ✅ Added mandatory 6 metrics evaluation
- ✅ Streamlined to essential project-specific rules only
- ✅ Links to detailed docs in `docs/` instead of duplicating content
