# ML/DS Common Rules Integration Summary

**Date:** 2026-06-17
**Project:** stock_vol_prediction01
**Integration:** ml-ds-common-rules package

---

## What Was Done

### 1. Created ml-ds-common-rules Package

**Location:** `D:\bmad-projects\ml-ds-common-rules`

**Package Structure:**
```
ml_ds_common_rules/
├── __init__.py                    # Main package exports
├── README.md                      # Package API documentation
├── data_quality/                  # Data validation
│   ├── validators.py              # DataQualityValidator class
│   └── __init__.py
├── testing/                       # Testing framework
│   ├── preflight.py               # PreFlightValidator class
│   ├── coverage_rules.py          # Test coverage validation
│   └── __init__.py
├── validation/                    # Model validation
│   ├── metrics.py                 # CommonMetrics, ModelValidator
│   └── __init__.py
└── code_quality/                  # Code quality checks
    ├── patterns.py                # Anti-pattern detection
    └── __init__.py
```

**Core Components:**
- ✅ `DataQualityValidator` - Data file validation, missing data checks, leakage detection
- ✅ `PreFlightValidator` - Environment checks, dependency validation, GPU detection
- ✅ `CommonMetrics` - Universal metrics for classification and regression
- ✅ `ModelValidator` - Overfitting detection, statistical tests, baseline comparison
- ✅ Code quality checks - Monkey patching detection, import validation

### 2. Updated stock_vol_prediction01/CLAUDE.md

**Section 2 Updated:**

**Before:**
```markdown
## 2. Common ML/DS Research Rules ⭐

**This project follows the ML/DS common clean code rules.**

**Reference:** `docs/common-rules/COMMON_RULES.md`
**Quick Reference:** `docs/common-rules/QUICK_REFERENCE.md`
```

**After:**
```markdown
## 2. Common ML/DS Research Rules ⭐

**This project follows the ML/DS common clean code rules.**

**References:**

**Documentation:**
- Main Project: `D:\bmad-projects\ml-ds-common-rules`
- Common Rules: `D:\bmad-projects\ml-ds-common-rules\COMMON_RULES.md`
- Quick Reference: `D:\bmad-projects\ml-ds-common-rules\QUICK_REFERENCE.md`
- Integration Guide: `D:\bmad-projects\ml-ds-common-rules\INTEGRATION_GUIDE.md`

**Python Package:**
- Package Name: `ml_ds_common_rules`
- Installation: `pip install -e D:\bmad-projects\ml-ds-common-rules`
- API Docs: `D:\bmad-projects\ml-ds-common-rules\ml_ds_common_rules\README.md`

**[Core Principles section...]**

**[Critical Rules section...]**

**[Research Best Practices section...]**

#### **Using ml_ds_common_rules Package**

This project integrates `ml_ds_common_rules` package for universal validation:

**Installation:**
```bash
pip install -e D:\bmad-projects\ml-ds-common-rules
```

**Usage in code:**
```python
# Data validation
from ml_ds_common_rules.data_quality import DataQualityValidator

validator = DataQualityValidator()
validator.validate_no_missing_data(df, critical_columns=['parkinson_volatility'])
validator.check_no_leakage(train_data, test_data, key_column='ticker')

# Model evaluation
from ml_ds_common_rules.validation import CommonMetrics, ModelValidator

metrics = CommonMetrics.calculate_all(y_true, y_pred, task_type='regression')

validator = ModelValidator()
overfitting_result = validator.detect_overfitting(train_loss, test_loss)

# Pre-flight checks
from ml_ds_common_rules.testing import PreFlightValidator

preflight = PreFlightValidator()
preflight.validate_dependencies(['torch', 'pandas', 'numpy'])
preflight.validate_gpu_availability()
```

**Integration Points:**
- Data Processing: Use `DataQualityValidator` in `src/common/parkinson_utils.py`
- Model Training: Use `PreFlightValidator` before expensive training runs
- Evaluation: Use `CommonMetrics` in `src/common/evaluation.py`
- Code Review: Use code quality checks during development
```

---

## How to Use ml_ds_common_rules in stock_vol_prediction01

### 1. Installation

```bash
# Install package (if not already installed)
cd D:\bmad-projects
pip install -e ml-ds-common-rules

# Verify installation
python -c "from ml_ds_common_rules.validation import CommonMetrics; print('OK')"
```

### 2. Integration in Code

#### A. Data Processing Integration

**File:** `src/common/parkinson_utils.py`

```python
from ml_ds_common_rules.data_quality import DataQualityValidator

def process_single_stock(raw_path: str, output_dir: str) -> tuple:
    """Process single stock from raw OHLCV to Parkinson volatility."""
    # Load data
    df = pd.read_csv(raw_path)

    # Add validation
    validator = DataQualityValidator()

    # Check for missing data
    critical_columns = ['High', 'Low', 'Close', 'Volume']
    if not validator.validate_no_missing_data(df, critical_columns=critical_columns):
        print(f"Skipping {ticker}: Missing critical data")
        return None, 0

    # Calculate Parkinson volatility
    # ... (existing code)

    return ticker, len(processed_df)
```

#### B. Model Training Integration

**File:** `src/lstm_baseline/train.py`

```python
from ml_ds_common_rules.testing import PreFlightValidator
from ml_ds_common_rules.validation import ModelValidator

def train_simple_lstm(data_dir: str, output_dir: str = None):
    """Train Simple LSTM with pre-flight validation."""

    # Pre-flight checks
    print("Running pre-flight validation...")
    preflight = PreFlightValidator()
    preflight.validate_dependencies(['torch', 'pandas', 'numpy'])
    gpu_info = preflight.validate_gpu_availability()

    if not gpu_info['available']:
        print("WARNING: GPU not available, training will be slow")

    # ... (existing training code)

    # Overfitting detection
    validator = ModelValidator()
    overfitting_result = validator.detect_overfitting(
        train_loss=train_losses[-1],
        test_loss=val_losses[-1]
    )

    if overfitting_result['overfitting']:
        print(f"WARNING: {overfitting_result['interpretation']}")

    return model, metrics
```

#### C. Evaluation Integration

**File:** `src/common/evaluation.py`

```python
from ml_ds_common_rules.validation import CommonMetrics

def evaluate_predictions(actuals: np.ndarray, predictions: np.ndarray) -> dict:
    """Evaluate predictions using universal metrics."""
    # Use CommonMetrics for universal metrics
    metrics = CommonMetrics.calculate_all(
        actuals, predictions,
        task_type='regression'
    )

    # Add custom metrics for this project
    metrics['QLIKE'] = qlike_loss(actuals, predictions)
    metrics['Directional_Acc'] = directional_accuracy(actuals, predictions)

    return metrics
```

### 3. Code Review Integration

**During Development:**

```python
# Check for code quality issues
from ml_ds_common_rules.code_quality import check_no_monkey_patching, check_generic_names

# Before committing code
result = check_no_monkey_patching('src/lstm_baseline/train.py')
if result['status'] == 'FAIL':
    print(f"ERROR: {result['message']}")

# Check function sizes
from ml_ds_common_rules.code_quality import check_function_size

result = check_function_size('src/lstm_baseline/train.py', max_lines=30)
if result['status'] == 'WARNING':
    print(f"WARNING: {result['message']}")
```

---

## Benefits for stock_vol_prediction01

### ✅ Universal Validation

- Data quality checks before training
- Missing data detection
- Data leakage prevention (temporal integrity)
- Statistical validation

### ✅ Consistent Metrics

- Universal metrics (RMSE, MAE, R²)
- Custom project metrics (QLIKE, Directional Accuracy)
- Baseline comparison
- Overfitting detection

### ✅ Early Issue Detection

- Pre-flight environment checks
- Dependency validation
- GPU availability detection
- Code quality checks

### ✅ Reusable Across Projects

- Same validation framework for HAR baseline
- Same metrics for LSTM baseline
- Same checks for future baselines
- Standardized best practices

---

## Next Steps

### 1. Integrate into Existing Code

**Priority Files to Update:**
- [ ] `src/common/parkinson_utils.py` - Add DataQualityValidator
- [ ] `src/common/evaluation.py` - Use CommonMetrics
- [ ] `src/lstm_baseline/train.py` - Add PreFlightValidator
- [ ] `src/har_baseline/train.py` - Add PreFlightValidator (when implemented)

### 2. Add to Testing Suite

```python
# tests/test_integration_ml_ds_rules.py
def test_ml_ds_rules_integration():
    """Test that ml_ds_common_rules package is integrated."""
    from ml_ds_common_rules.validation import CommonMetrics
    from ml_ds_common_rules.data_quality import DataQualityValidator

    # Test basic functionality
    import numpy as np
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])

    metrics = CommonMetrics.calculate_rmse(y_true, y_pred)
    assert metrics > 0
```

### 3. Update Documentation

- [ ] Add to `README.md` - Installation instructions
- [ ] Add to `docs/project/` - Integration guide
- [ ] Update training scripts - Use PreFlightValidator

### 4. Share with Other Projects

The `ml-ds-common-rules` package can now be used in:
- New ML/DS projects
- Existing projects (via git submodule or pip install)
- Team projects (standardized validation)

---

## Summary

✅ **ml-ds-common-rules package created and tested**
✅ **stock_vol_prediction01/CLAUDE.md updated**
✅ **Integration guide provided**
✅ **Ready to use in code**

**Key Achievement:** Created a **universal ML/DS validation framework** that can be reused across all projects, ensuring consistent best practices and quality standards.

---

**Status:** Ready for integration into code
**Next Priority:** Integrate into `src/common/parkinson_utils.py` and `src/common/evaluation.py`
