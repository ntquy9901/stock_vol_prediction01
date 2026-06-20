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
# Returns: mse, rmse, mae, r2, qlike, directional_accuracy
```

**6 Mandatory Metrics:**
1. **MSE** - Mean Squared Error (lower is better) ⭐ NEW
2. **RMSE** - Root Mean Squared Error (lower is better)
3. **MAE** - Mean Absolute Error (lower is better)
4. **R²** - Variance Explained (higher is better)
5. **QLIKE** - Academic standard cho volatility (lower is better)
6. **Dir Acc** - Directional Accuracy (higher is better)

**Output Requirements (MANDATORY):**

**1. Console Output Format:**
```python
# Validation Results
print(f"Val MSE: {val_metrics['mse']:.6f}")
print(f"Val RMSE: {val_metrics['rmse']:.6f}")
print(f"Val MAE: {val_metrics['mae']:.6f}")
print(f"Val R²: {val_metrics['r2']:.6f}")
print(f"Val QLIKE: {val_metrics['qlike']:.6f}")
print(f"Val Dir Acc: {val_metrics['directional_accuracy']:.2f}%")

# Test Results (same format)
print(f"Test MSE: {test_metrics['mse']:.6f}")
print(f"Test RMSE: {test_metrics['rmse']:.6f}")
# ... etc
```

**2. JSON Output Format:**
```python
results = {
    'validation_metrics': {
        'mse': float(val_metrics['mse']),
        'rmse': float(val_metrics['rmse']),
        'mae': float(val_metrics['mae']),
        'r2': float(val_metrics['r2']),
        'qlike': float(val_metrics['qlike']),
        'directional_accuracy': float(val_metrics['directional_accuracy'])
    },
    'test_metrics': { ... },
    'val_test_diff': {
        'mse_diff': float(mse_diff),
        'rmse_diff': float(rmse_diff),
        'mae_diff': float(mae_diff),
        'r2_diff': float(r2_diff),
        'qlike_diff': float(qlike_diff),
        'dir_acc_diff': float(dir_acc_diff)
    }
}
```

**3. Comparison Table (for validation):**
```
Metric          Validation       Test            Difference
------------------------------------------------------------
MSE             0.xxxxxx        0.xxxxxx       +0.xxxxxx
RMSE            0.xxxxxx        0.xxxxxx       +0.xxxxxx
MAE             0.xxxxxx        0.xxxxxx       +0.xxxxxx
R²              0.xxxxxx        0.xxxxxx       +0.xxxxxx
QLIKE           0.xxxxxx        0.xxxxxx       +0.xxxxxx
Dir Acc         xx.xx%          xx.xx%         +x.xx%
```

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

### **Advanced Architecture: LSTM-GAT Hybrid** 🚀

#### **5. Temporal Graph Attention Network (TemporalGAT)** - NEXT GENERATION
- **Features:** 22 features (HAR + technical) for all 30 stocks simultaneously
- **Architecture:** LSTM (temporal) + Graph Attention Network (spatial)
- **Innovation:** Dynamic graph construction + multi-head attention
- **Target:** RMSE < 0.15, Dir Acc > 75% (vs current: 0.18, 67.90%)
- **Status:** Architecture design complete, ready for implementation
- **File:** `docs/project/LSTM_GAT_ARCHITECTURE.md`

**Key Components:**
1. **Per-Stock LSTM Encoder:** Temporal feature learning for each stock
2. **Dynamic Graph Builder:** Correlation + volatility spillover based edges
3. **Graph Attention Layers:** Multi-head attention for spatial relationships
4. **Temporal-Spatial Fusion:** Combines both branches for final prediction

**Performance Targets:**
- RMSE: 0.18 → **< 0.15** (17% improvement)
- Dir Acc: 67.90% → **> 75%** (7% improvement)
- QLIKE: ~0.12 → **< 0.10** (17% improvement)
- R²: ~0.65 → **> 0.75** (15% improvement)

**Advantages over LSTM-only:**
- ✅ Captures cross-stock correlations and spillover effects
- ✅ Dynamic graph adapts to changing market conditions
- ✅ Attention mechanism learns influential stocks
- ✅ Multi-scale: temporal (LSTM) + spatial (GAT)

**Implementation Roadmap:**
- Week 1: Data preparation (technical indicators, graph utilities)
- Week 2: Model development (LSTM encoder, GAT layers, fusion)
- Week 3: Training & evaluation (hyperparameter tuning, comparison)
- Week 4: Analysis & deployment (attention visualization, ablation)

#### **6. TimesFM 2.5 LoRA Fine-Tuning** - FOUNDATION MODEL APPROACH
- **Features:** Parkinson volatility (univariate time series)
- **Architecture:** TimesFM 2.5 (232M params) + LoRA adapters (~1.4M trainable params, 0.6%)
- **Method:** Decoder-only transformer with LoRA fine-tuning
- **Purpose:** State-of-the-art foundation model for time series
- **File:** `src/timesfm_baseline/timesfm_lora_finetuning.py`
- **Status:** Implementation complete, tested, reviewed
- **Documentation:** See `docs/timesfm/` for architecture and implementation details

**Key Innovations:**
- ✅ Foundation model approach (pre-trained on massive time series data)
- ✅ Parameter-efficient fine-tuning (LoRA adapters)
- ✅ Random window sampling (data-efficient training)
- ✅ No external normalization (TimesFM handles RevIN internally)
- ✅ Comprehensive testing (34 tests, 100% pass rate)

**Performance Targets:**
- RMSE: < 0.18 (baseline) → **< 0.15** (target)
- Dir Acc: > 55% (baseline) → **> 60%** (target)
- Training time: ~2 hours on GPU (vs ~30 min for LSTM)
- Trainable params: 1.4M (vs 65K for LSTM-HAR)

**Lessons Learned:**
- ⚠️ **3 adversarial reviews conducted** - Found 40 bugs total
- 📚 **Comprehensive lessons learned documented** - See `docs/LESSONS_LEARNED_TIMESFM_ADVERSARIAL_REVIEWS.md`
- ✅ **Quick reference checklist created** - See `docs/QUICK_REFERENCE_CHECKLIST.md`
- 🔍 **All bugs fixed and tested** - 34/34 tests passing

---

## 5. Adversarial Review Process & Lessons Learned

This project uses adversarial code reviews to ensure production-ready code quality.

### **Adversarial Review Process**
1. **Cynical review** - Assume code has problems, look for hidden bugs
2. **Find 10+ issues** - Minimum threshold for review depth
3. **Fix all issues** - No exceptions, all HIGH/MEDIUM must be fixed
4. **Add unit tests** - Every fix must be tested
5. **Document lessons** - Add to knowledge base

### **TimesFM LoRA Review Results**
- **Review 1:** 15 bugs found (3 HIGH, 9 MEDIUM, 3 LOW)
- **Review 2:** 12 bugs found (3 HIGH, 6 MEDIUM, 3 LOW)
- **Review 3:** 13 bugs found (4 HIGH, 6 MEDIUM, 3 LOW)
- **Total:** 40 bugs fixed across 3 reviews

### **Key Lessons Learned Documents**
- 📘 **[Full Lessons Learned](docs/LESSONS_LEARNED_TIMESFM_ADVERSARIAL_REVIEWS.md)** - Comprehensive guide with anti-patterns and mandatory practices
- 📋 **[Quick Reference Checklist](docs/QUICK_REFERENCE_CHECKLIST.md)** - Fast checklist for code reviews (87 items)
- 📊 **[Bug Statistics](docs/LESSONS_LEARNED_TIMESFM_ADVERSARIAL_REVIEWS.md#-summary-quick-reference)** - Breakdown by category and severity

### **Top 5 Bug Categories**
1. **Memory Management** (7 bugs) - Memory leaks, unbounded growth, improper cleanup
2. **Input Validation** (6 bugs) - Missing checks, late validation, poor error messages
3. **Data Pipeline** (8 bugs) - Silent data loss, inefficient tensor creation
4. **MLflow Integration** (4 bugs) - Metrics loss on crashes, poor error handling
5. **Resource Cleanup** (4 bugs) - File handles, model references, GPU memory

### **Mandatory Practices for Future Development**
- ✅ Validate all parameters at function entry (not deep in code)
- ✅ Create tensors on-the-fly (never pre-create in `__init__`)
- ✅ Use `pin_memory=True` + `non_blocking=True` for GPU training
- ✅ Save checkpoints BEFORE batch work (not after)
- ✅ Wrap MLflow calls in try/except per epoch
- ✅ Delete large objects ASAP (del + empty_cache)
- ✅ Test edge cases (empty, single, invalid inputs)
- ✅ Provide helpful error messages (what + why + how)

### **Quick Reference for Code Review**
Before approving any ML/DS code, verify:
- [ ] Data pipeline uses on-the-fly tensor creation
- [ ] GPU training uses `pin_memory=True` and `non_blocking=True`
- [ ] No silent data loss (warn if using `drop_last=True`)
- [ ] Checkpoints saved before work (not after)
- [ ] All parameters validated at entry point
- [ ] Error messages include what/why/how
- [ ] MLflow calls wrapped in try/except per epoch
- [ ] Large objects deleted explicitly (del + empty_cache)
- [ ] Tests for edge cases (empty, single, invalid)
- [ ] No memory leaks (profile long runs)

**See `docs/QUICK_REFERENCE_CHECKLIST.md` for full 87-item checklist.**

---

## 6. Evaluation Methodology

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

### **Standard Hyperparameters (ALL Models)** ⭐

**CRITICAL:** All models MUST use these standardized hyperparameters for fair comparison.

**Training Configuration:**
```python
# ALL models (LSTM variants)
num_epochs = 70          # Maximum training epochs
patience = 15            # Early stopping patience
```

**Applied to All Files:**
- ✅ `src/lstm_har_enhanced/train_with_validation.py`
- ✅ `src/lstm_har_enhanced/train_enhanced.py`
- ✅ `src/lstm_har_baseline/train_with_validation.py`
- ✅ `src/lstm_har_baseline/train.py`
- ✅ `src/lstm_baseline/train_with_validation.py`
- ✅ `src/lstm_baseline/train.py`

**Why 70 epochs?**
- Sufficient for convergence without overfitting
- Allows early stopping to prevent overtraining
- Balances training time with model performance

**Why patience=15?**
- Gives model enough room to plateau before stopping
- Prevents premature stopping during temporary loss increases
- Standard practice for time series forecasting

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
**Version:** 3.1 (Standardized Hyperparameters & Metrics)
**Status:** Active Development

---

**Changes in v3.1 (Current Version):**
- ✅ **Standardized hyperparameters:** 70 epochs, 15 patience for ALL models
- ✅ **Added MSE to 6 mandatory metrics** (was 5, now 6)
- ✅ **Mandatory output format:** Console + JSON must include all 6 metrics
- ✅ **Updated all training files** to use standardized hyperparameters
- ✅ **Enhanced metrics reporting:** Added MSE to console, JSON, and comparison tables

**Changes in v3.0 (Previous Version):**
- ✅ Removed duplicated ML/DS common rules (now reference external package)
- ✅ Reduced from 1,798 lines to ~400 lines (58KB → ~12KB)
- ✅ Added 3-way temporal split evaluation section
- ✅ Added mandatory 6 metrics evaluation
- ✅ Streamlined to essential project-specific rules only
- ✅ Links to detailed docs in `docs/` instead of duplicating content
