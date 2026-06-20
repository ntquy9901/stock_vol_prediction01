# Stock Volatility Prediction - VN30

**Project:** Multi-horizon volatility forecasting cho VN30 stocks  
**Date:** 2026-06-15  
**Methodology:** HAR-R with Parkinson volatility, 5-day focus initially  
**Status:** Development Phase - Sprint 1 Preparation

---

## 1. Project Overview

### **Objective**
Build robust volatility prediction system cho 30 VN30 stocks using daily OHLCV data, implementing HAR (Heterogeneous Autoregressive) methodology adapted cho daily frequency.

### **Key Requirements**
- **Primary Target:** 5-day ahead volatility forecast (current focus) ✅
- **Secondary Targets:** 1, 10, 22-day forecasts (future expansion)
- **Input Data:** Daily OHLCV (30 stocks, 2006-2026)
- **Approach:** HAR-R baseline with Parkinson volatility, enhanced with ML models

### **Success Criteria**
- **RMSE:** < 0.20 cho 5-day forecasts
- **Directional Accuracy:** > 55% cho 5-day forecasts
- **QLIKE Loss:** Academic standard cho volatility forecasting
- **Test Coverage:** 85%+ overall, 90% cho critical paths
- **Code Quality:** ML/DS common rules compliance

### **Project Timeline**
- **Week 1-2:** 5-day HAR-R baseline implementation
- **Week 3-4:** Enhanced models (extended features, LSTM)
- **Week 5:** Multi-horizon expansion decision
- **Week 6+:** Production deployment

---

## 2. Common ML/DS Research Rules ⭐

**This project follows the ML/DS common clean code rules.**

### **References**

**Documentation:**
- Main Project: `D:\bmad-projects\ml-ds-common-rules`
- Common Rules: `D:\bmad-projects\ml-ds-common-rules\COMMON_RULES.md`
- Quick Reference: `D:\bmad-projects\ml-ds-common-rules\QUICK_REFERENCE.md`
- Integration Guide: `D:\bmad-projects\ml-ds-common-rules\INTEGRATION_GUIDE.md`

**Python Package:**
- Package Name: `ml_ds_common_rules`
- Installation: `pip install -e D:\bmad-projects\ml-ds-common-rules`
- API Docs: `D:\bmad-projects\ml-ds-common-rules\ml_ds_common_rules\README.md`

### **Core Principles**
1. **Code is read much more than written** - Write for future readers
2. **Leave code better than you found it** - Boy scouts rule
3. **Keep it simple** - Simple > Clever
4. **Match quality to maturity** - Don't over-engineer POCs

### **Code Quality Levels**

**Level 1: Total Mess (Exploration Phase)**
- Quick POC, testing ideas, learning algorithms
- Poor variable names, thrown together in one script
- ✅ Acceptable for exploration/learning
- ⚠️ **Never commit to production without refactoring**

**Level 2: Readable (Sharing Phase)**
- Others can understand what's going on
- Meaningful variable names, well-divided into functions
- ✅ Minimum quality for sharing code
- ✅ Good starting point for refactoring/code review

**Level 3: Production (Deployment Phase)**
- Clean, tested, documented, follows all guidelines
- Comprehensive tests, production-ready, easy to maintain
- ✅ **Required for production deployment**
- ✅ Required for critical systems and long-term projects

### **Pythonic Code Patterns** (From "Clean Code in Python" by Mariano Anaya)

#### **DRY Principle (Don't Repeat Yourself)**
```python
# ✅ GOOD - Extract common pattern into decorator
def log_execution_time(func):
    """Decorator to log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed:.4f}s")
        return result
    return wrapper

@log_execution_time
def load_training_data(path):
    return pd.read_csv(path)

# ❌ BAD - Same logging code repeated everywhere
def load_training_data(path):
    start = time.time()
    data = pd.read_csv(path)
    elapsed = time.time() - start
    logger.info(f"load_training_data executed in {elapsed:.4f}s")
    return data
```

#### **EAFP vs LBYL (Exception Handling)**
```python
# ✅ GOOD - EAFP: Try first, handle exceptions (Pythonic)
def process_user_id(user_id, users):
    try:
        user = users[user_id]
        return user.process()
    except KeyError:
        logger.warning(f"User {user_id} not found")
        return None

# ❌ BAD - LBYL: Check first, then act (Less Pythonic)
def process_user_id(user_id, users):
    if user_id in users:  # Check first
        return users[user_id].process()
    else:
        logger.warning(f"User {user_id} not found")
        return None
```

#### **Context Managers for Resource Management**
```python
# ✅ GOOD - Context manager for automatic cleanup
@contextlib.contextmanager
def timer_context(task_name):
    """Context manager for timing operations"""
    start = time.time()
    logger.info(f"Starting {task_name}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"Completed {task_name} in {elapsed:.4f}s")

# Usage
with timer_context("data preprocessing"):
    preprocess_data(training_data)
```

#### **Properties vs Get/Set Methods**
```python
# ✅ GOOD - Using properties
class ModelTrainer:
    def __init__(self, model):
        self._model = model
        self._training_state = 'idle'
    
    @property
    def is_training(self):
        """Check if training is in progress"""
        return self._training_state == 'training'

# Usage: trainer.is_training (not trainer.is_training())

# ❌ BAD - Java-like get/set methods
class ModelTrainer:
    def is_training(self):  # Extra parentheses
        return self._training_state == 'training'
```

#### **PEP Standards Compliance**
```python
# ✅ GOOD - PEP 8 compliant
# Imports: grouped and ordered
import os                  # Standard library
import random             # Standard library
import numpy as np         # Third-party
import pandas as pd        # Third-party
from myproject import utils  # Local imports

# Type hints (PEP 3107)
def calculate_metrics(predictions: np.ndarray, 
                      labels: np.ndarray) -> Dict[str, float]:
    """Calculate classification metrics"""
    return {'accuracy': 0.95}

# ❌ BAD - Not PEP 8 compliant
import numpy as np
import os
from myproject import utils
import random  # Not grouped

def calculate_metrics(predictions, labels):  # No type hints
    return {'accuracy': 0.95}
```

### **Critical Rules cho This Project**

#### **Naming Conventions**
```python
# ✅ CORRECT - Descriptive and Clear
volatility_forecast = model.predict(data)
train_accuracy = 0.95
parkinson_volatility = calculate_parkinson_volatility(data)
har_weekly_features = create_har_features(volatility, window=5)

# ❌ AVOID - Cryptic and Generic
vol = model.predict(data)                 # Too generic
acc = 0.95                              # Abbreviated
pred = forecast(data)                    # Unclear context
har = create_features(vol, 5)         # Abbreviated
```

#### **Function Design**
- **Small functions:** Maximum 30 lines, ideally < 20
- **Single responsibility:** One function does one thing well
- **Few parameters:** Prefer < 3 parameters, use configs cho more
- **No side effects:** Functions should return, not print/modify global state

**✅ GOOD - Small, Focused Functions:**
```python
def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate realized volatility using rolling standard deviation"""
    return returns.rolling(window).std()

def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize data to zero mean and unit variance"""
    return (data - data.mean()) / data.std()
```

**❌ BAD - Large, Multi-Purpose Functions:**
```python
def process_and_train(data_path, model_type, epochs, lr):
    """
    Load data, preprocess it, create model, train it, and save results
    (This function does TOO MANY things!)
    """
    # 50+ lines of code doing multiple things
    pass
```

**✅ GOOD - Early Returns to Avoid Deep Nesting:**
```python
def should_continue_training(epoch, max_epochs, early_stop):
    """Check if training should continue"""
    if epoch >= max_epochs:
        return False
    if early_stop:
        return False
    return True  # Clear logic
```

**❌ BAD - Deep Nesting:**
```python
def should_continue_training(epoch, max_epochs, early_stop):
    """Complex nested logic (hard to read)"""
    if epoch < max_epochs:
        if not early_stop:
            if some_other_condition:
                if yet_another_condition:
                    return True
    return False
```

#### **Code Organization**
- **One concern per file:** Separate data processing, model training, evaluation
- **High-level at top:** Main orchestration functions first, helpers below
- **Related code close:** Keep related functions vertically adjacent

#### **Documentation**
- **Explain WHY not HOW:** Comments should explain reasoning, not obvious mechanics
- **Self-documenting code:** Use good names instead of obvious comments
- **Docstrings:** All public functions必须有 comprehensive docstrings

#### **Testing**
- **Coverage targets:** 85%+ overall, 90% cho critical paths
- **Test pyramid:** 60% unit, 30% integration, 10% E2E
- **Pre-flight validation:** Environment checks before expensive operations

**Testing Best Practices:**
```python
# ✅ GOOD - Test structure (Arrange, Act, Assert)
class TestComponent:
    def test_feature_x(self):
        # Arrange
        input_data = create_test_data()
        
        # Act
        result = function_to_test(input_data)
        
        # Assert
        assert result is not None
        assert result == expected_output
```

### **Anti-Patterns to Avoid**

#### **❌ Naming Anti-Patterns**
```python
# Single letters (except loop counters)
x = compute()               # ❌ What does x mean?
y = transform()             # ❌ Why not y_transformed?
data = load()               # ❌ Too generic

# Generic names
manager = Manager()         # ❌ Manager of what?
processor = Processor()     # ❌ Processor for what data?

# Abbreviations
lr = 0.001                  # ❌ Use learning_rate
clf = classifier            # ❌ Use classifier
pred = predictions          # ❌ Use predictions

# Same word, different meanings
def get_data(): pass        # ❌ Fetches vs computes vs creates
def get_result(): pass      # ❌ Inconsistent meaning
def get_model(): pass       # ❌ Creates vs retrieves
```

#### **❌ Function Anti-Patterns**
```python
# Large functions (>50 lines)
def do_everything(data):
    # 100+ lines doing many things
    # Load, Preprocess, Train, Evaluate, Save
    pass

# Multiple responsibilities
def load_and_train_and_save(data_path):
    # Does multiple things - should be split
    pass

# Many parameters (>5)
def train(model, loader, val_loader, opt, sch, ep, lr, mom, dev):
    # Too many parameters! Use config dict instead
    pass

# Side effects
def calculate_and_print(data):
    result = calculate(data)
    print(result)  # ❌ Side effect!
    return result

# Commented code
# def old_function():
#     pass  # ❌ Delete it, git has your back
```

#### **❌ Comment Anti-Patterns**
```python
# Commenting the obvious
i = i + 1  # ❌ increment i by one
data = pd.read_csv('data.csv')  # ❌ Load data from file

# Outdated comments
# This function trains the model (but code was changed to evaluate!)
def evaluate_model(model, data):  # ❌ Misleading!
    pass

# Noise comments
# This is a function  # ❌ Not useful
def function_name():
    pass
```

### **Refactoring Rules**

#### **When to Refactor**
- ✅ **Before moving on** - Refactor while you still remember the context
- ✅ **Before committing** - Leave code better than you found it
- ✅ **During code review** - Fix issues when you see them
- ✅ **Before adding features** - Clean foundation makes adding easier

#### **How to Refactor**
1. **Write tests first** - Ensure behavior doesn't change
2. **Small improvements** - Better code happens gradually
3. **One change at a time** - Don't refactor everything at once
4. **Test after each change** - Verify nothing broke
5. **Commit frequently** - Save progress incrementally

#### **Boy Scouts Rule**
**"Always leave the code better than you found it"**

When you see bad code:
- ✅ Rename confusing variable
- ✅ Extract large function
- ✅ Add missing docstring
- ✅ Remove commented code
- ✅ Fix formatting

### **Learning Curves and Overfitting Detection** ⭐ MANDATORY

**✅ GOOD - Learning Curve Practices:**
```python
def plot_learning_curves(train_losses, val_losses, save_path):
    """
    Plot learning curves to detect overfitting.
    
    MANDATORY for all training runs to visualize:
    - Training loss vs validation loss
    - Convergence behavior
    - Overfitting/underfitting signs
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Check for overfitting
    if len(val_losses) > 10:
        recent_val_trend = val_losses[-5:]
        if all(recent_val_trend[i] > recent_val_trend[i-1] for i in range(1, len(recent_val_trend))):
            print("⚠️ WARNING: Validation loss increasing - possible overfitting!")

# During training - PLOT EVERY 10 EPOCHS
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = validate_epoch(model, val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Plot learning curves every 10 epochs
    if (epoch + 1) % 10 == 0:
        plot_path = f'results/learning_curves_epoch_{epoch+1}_{timestamp}.png'
        plot_learning_curves(train_losses, val_losses, plot_path)
```

**❌ BAD - No Learning Curves:**
```python
# No learning curves at all
for epoch in range(num_epochs):
    train(model, data)
# How do you know if it's overfitting?

# Only log numerical values (no visualization)
print(f"Epoch {epoch}: Train Loss={train_loss}, Val Loss={val_loss}")
# Hard to see trends without visualization
```

**Learning Curve Rules:**
- ⭐ **Mandatory visualization** - Plot learning curves for ALL training runs
- ⭐ **Save during training** - Plot every N epochs (e.g., every 10 epochs)
- ⭐ **Detect overfitting** - Validation loss increasing while training loss decreasing
- ⭐ **Save final plot** - Always save final learning curve with experiment results
- ⭐ **Include in report** - Learning curves must be included in experiment reports

#### **Research Best Practices**
- **Version control:** Track all experiments with git + MLflow
- **Reproducibility:** Fixed random seeds (42)
- **Hyperparameter tracking:** Document all experiments
- **Checkpoint saving:** Save intermediate results
- **Learning curves:** Plot training/validation curves (mandatory)

**File Management and Archiving:**
```python
# ✅ GOOD - When fixing or creating new versions
# Old file: src/data_processing.py
# New location: archived/data_processing_old_2025-06-13_bugfix_v1.py

# ✅ GOOD - Archive deprecated files
project/
├── src/                    # Current, active code only
│   ├── data_processing.py
│   └── model_training.py
└── archived/               # Deprecated but reference code
    ├── data_processing_old_2025-06-13_bugfix_v1.py
    └── train_old_2025-06-13_refactored.py

# ❌ BAD - Leave old files cluttering directory
# src/data_processing.py
# src/data_processing_old.py  # Clutters directory!
# src/data_processing_v2.py    # Which one is current?
```

**Generated Files Naming Convention** ⭐ CRITICAL:
```python
from datetime import datetime

# ✅ GOOD - ALL generated files MUST include timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

# Result files
results_file = f"experiment_results_{timestamp}.csv"
plot_file = f"learning_curves_{timestamp}.png"
metrics_file = f"test_metrics_{timestamp}.json"

# Model files
model_file = f"best_model_{timestamp}.pth"
checkpoint_file = f"checkpoint_epoch_{epoch}_{timestamp}.pt"

# Log files
log_file = f"training_log_{timestamp}.txt"
error_log = f"error_log_{timestamp}.log"

# Report files
report_file = f"experiment_report_{timestamp}.md"

# ❌ BAD - Static filenames without timestamp
results_file = "experiment_results.csv"  # Gets overwritten!
model_file = "best_model.pth"            # Which version?
log_file = "training_log.txt"            # Lost previous logs
```

**File Types Requiring Timestamp:**
1. **Result files:** CSV, JSON, TXT with metrics/predictions
2. **Model files:** .pth, .pt, .pkl, .h5, .joblib
3. **Plot files:** PNG, JPG, PDF with graphs/curves
4. **Log files:** TXT, LOG with training/error logs
5. **Report files:** MD, TXT, PDF with analysis/summaries
6. **Checkpoint files:** Intermediate model states

**Exceptions (NO timestamp required):**
- ✅ Source code files (.py in src/)
- ✅ Configuration files (config.yaml, settings.json, .env)
- ✅ Documentation files (README.md, CONTRIBUTING.md)
- ✅ Reference files (CLAUDE.md, project-context.md)

**Directory Organization with Timestamps:**
```
project/
├── results/
│   ├── experiment_1_2026-06-17_225000/
│   │   ├── metrics_2026-06-17_225000.csv
│   │   ├── predictions_2026-06-17_225000.csv
│   │   └── plot_2026-06-17_225000.png
│   └── experiment_2_2026-06-18_143000/
│       ├── metrics_2026-06-18_143000.csv
│       └── plot_2026-06-18_143000.png
├── models/
│   ├── baseline_2026-06-17_225000.pth
│   └── enhanced_2026-06-18_143000.pth
└── logs/
    ├── training_2026-06-17_225000.txt
    └── evaluation_2026-06-18_143000.txt
```

#### **Using ml_ds_common_rules Package**

This project integrates `ml_ds_common_rules` package for universal validation:

**Installation (if not already installed):**
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

# Calculate universal metrics
metrics = CommonMetrics.calculate_all(
    y_true, y_pred,
    task_type='regression'  # or 'classification'
)

# Validate model performance
validator = ModelValidator()
overfitting_result = validator.detect_overfitting(train_loss, test_loss)
baseline_comparison = validator.compare_to_baseline(model_metrics, baseline_metrics)

# Pre-flight checks before training
from ml_ds_common_rules.testing import PreFlightValidator

preflight = PreFlightValidator()
preflight.validate_dependencies(['torch', 'pandas', 'numpy'])
preflight.validate_gpu_availability()

# Code quality checks
from ml_ds_common_rules.code_quality import check_no_monkey_patching

result = check_no_monkey_patching('src/lstm_baseline/train.py')
```

**Integration Points:**
- **Data Processing:** Use `DataQualityValidator` in `src/common/parkinson_utils.py`
- **Model Training:** Use `PreFlightValidator` before expensive training runs
- **Evaluation:** Use `CommonMetrics` in `src/common/evaluation.py`
- **Code Review:** Use code quality checks during development

**Benefits:**
- ✅ Universal validation across all ML/DS projects
- ✅ Consistent metrics and evaluation
- ✅ Early detection of data/quality issues
- ✅ Standardized best practices

---

## 3. Project Organization Rules ⭐ CRITICAL

**QUICK REFERENCE - All Organization Rules in One Place**

### **8 Golden Rules for Project Structure**

#### **Rule 1: Code Organization Rule**
- ✅ **ALL .py files MUST be in `src/` subdirectories**
- ❌ **NO .py files in root directory**
- Structure:
  ```
  src/
  ├── common/              # Shared utilities
  ├── har_baseline/       # HAR-R baseline
  ├── lstm_baseline/      # LSTM baseline
  └── experiment/         # Experimental code
  ```

#### **Rule 2: Results Organization Rule**
- ✅ **ALL results MUST go in `results/` folder**
- ✅ **Subfolder MUST include timestamp + model name**
- Format: `{model_name}_{YYYY-MM-DD_HHMMSS}/`
- Examples:
  - ✅ `results/simple_lstm_2026-06-17_225000/`
  - ❌ `results_simple_lstm/` (missing timestamp)

#### **Rule 3: Models Organization Rule**
- ✅ **ALL trained models MUST go in `models/` subdirectories**
- ✅ **Subfolder MUST include timestamp + baseline name**
- Format: `{baseline}_{YYYY-MM-DD_HHMMSS}/`
- Examples:
  - ✅ `models/har_baseline_2026-06-15_231200/`
  - ❌ `models/har_baseline/` (no timestamp)

#### **Rule 4: Documentation Organization Rule**
- ✅ **Only 3 .md files in root directory:**
  - `README.md` - Project overview
  - `CLAUDE.md` - Project rules
  - `project-context.md` - Project context
- ✅ **ALL other .md files MUST go in `docs/` subdirectories**
- Structure:
  ```
  docs/
  ├── lstm/              # LSTM documentation
  ├── project/           # Project docs
  ├── common-rules/      # ML/DS rules
  └── baseline/          # Other baselines
  ```

#### **Rule 5: Common Code Rule**
- ✅ **Shared utilities in `src/common/`**
- Examples:
  - ✅ `src/common/evaluation.py` - Used by all baselines
  - ❌ `src/lstm_baseline/evaluation.py` - Duplicate code

#### **Rule 6: Baseline Organization Rule**
- ✅ **Each baseline has own subdirectory in `src/`**
- Format: `src/{baseline_name}/`
- Examples:
  - ✅ `src/har_baseline/`
  - ✅ `src/lstm_baseline/`

#### **Rule 7: Experimental Code Rule**
- ✅ **Temporary/experimental code in `src/experiment/`**
- ✅ **Debug scripts MUST be in `src/experiment/`, not main source folders**
- ✅ **Debug files are TEMPORARY - delete after issue resolved**
- Examples:
  - ✅ `src/experiment/debug_scaling.py`
  - ✅ `src/experiment/debug_training_failure.py`
  - ✅ `src/experiment/test_issue_xyz.py`
  - ❌ `src/lstm_baseline/debug_xxx.py` (debug in main folder)
  - ❌ Root level experimental scripts

**Debug File Guidelines:**
- **Purpose:** Quick investigation, NOT production code
- **Lifetime:** Temporary - delete after bug fixed
- **Naming:** Start with `debug_` or `test_` prefix
- **Documentation:** Minimal - comment what issue being debugged
- **Cleanup:** Remove from git after resolving, add to .gitignore if needed

**Examples of Debug Files:**
```python
# ✅ CORRECT - In src/experiment/
src/experiment/debug_training_failure.py    # Investigate training issue
src/experiment/test_scaling_bug.py          # Test specific scaler bug
src/experiment/verify_gradient_flow.py     # Debug gradients

# ❌ WRONG - In main source folders
src/lstm_baseline/debug_training.py         # Clutters main code!
src/common/debug_data_processing.py        # Should be in experiment/
```

#### **Rule 8: Timestamp Naming Rule ⭐ CRITICAL**
- ✅ **ALL generated files MUST include timestamp in filename**
- ✅ **Applies to: Markdown reports, result files, model files, logs, etc.**
- ✅ **Format: `{name}_{YYYY-MM-DD_HHMMSS}.{ext}`**
- ❌ **NO static filenames without timestamp (except source code)**

**Purpose:** Track versions, identify latest results, prevent overwriting

**Examples by file type:**

**Markdown Reports:**
```
docs/lstm/
  ✅ DATA_ORGANIZATION_REPORT_2026-06-17_234500.md
  ✅ TRAINING_SUMMARY_2026-06-17_225000.md
  ✅ EXPERIMENT_RESULTS_2026-06-18_143000.md
  ❌ DATA_ORGANIZATION_REPORT.md (no timestamp)
```

**Result Files:**
```
results/simple_lstm_2026-06-17_225000/
  ✅ test_metrics_2026-06-17_225000.csv
  ✅ training_curves_2026-06-17_225000.png
  ✅ predictions_2026-06-17_225000.csv
  ❌ test_metrics.csv (no timestamp - overridden by folder timestamp OK)
```

**Model Files:**
```
models/lstm_baseline_2026-06-17_225000/
  ✅ best_simple_lstm_2026-06-17_225000.pth
  ✅ training_history_2026-06-17_225000.json
  ✅ VCB_model_2026-06-17_225000.pt
  ❌ best_simple_lstm.pth (no timestamp)
```

**Log Files:**
```
logs/
  ✅ training_log_2026-06-17_225000.txt
  ✅ evaluation_log_2026-06-18_143000.txt
  ❌ training_log.txt (no timestamp)
```

**Exceptions (NO timestamp required):**
- ✅ Source code files (.py in src/)
- ✅ Configuration files (config.yaml, .env)
- ✅ README files (README.md)
- ✅ Documentation reference (CLAUDE.md, project-context.md)

**Implementation:**
```python
from datetime import datetime

# Generate timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

# Use in filenames
report_file = f"DATA_ORGANIZATION_REPORT_{timestamp}.md"
model_file = f"best_simple_lstm_{timestamp}.pth"
log_file = f"training_log_{timestamp}.txt"
```

### **Folder/File Naming Conventions**

**Results:**
```
results/{model_name}_{YYYY-MM-DD_HHMMSS}/
  ✅ simple_lstm_2026-06-17_225000/
  ✅ har_baseline_2026-06-18_143000/
```

**Models:**
```
models/{baseline}_{YYYY-MM-DD_HHMMSS}/
  ✅ har_baseline_2026-06-15_231200/
  ✅ lstm_baseline_2026-06-17_225000/
```

**Documentation:**
```
docs/{topic}/{filename}.md
  ✅ docs/lstm/ENHANCED_LSTM_GUIDE.md
  ✅ docs/project/REFACTOR_SUMMARY.md
```

### **Quick Verification Checklist**

```bash
# Root directory should only have:
ls *.md  # Should show: README.md, CLAUDE.md, project-context.md

# No code files in root:
ls *.py  # Should show nothing

# Results use timestamps:
ls results/  # Should show: simple_lstm_2026-06-17_225000/

# Models use timestamps:
ls models/  # Should show: har_baseline_2026-06-15_231200/
```

### **Mandatory Directory Structure**

```
project_root/
├── README.md                   # ✅ 1 of 3 .md in root
├── CLAUDE.md                   # ✅ 2 of 3 .md in root
├── project-context.md          # ✅ 3 of 3 .md in root
├── src/                        # ✅ ALL code here
│   ├── common/
│   ├── har_baseline/
│   ├── lstm_baseline/
│   └── experiment/
├── docs/                       # ✅ ALL docs here
│   ├── lstm/
│   ├── project/
│   └── common-rules/
├── results/                    # ✅ ALL results here
│   ├── simple_lstm_2026-06-17_225000/
│   └── archive/
├── models/                     # ✅ ALL models here
│   ├── har_baseline_2026-06-15_231200/
│   └── lstm_baseline_2026-06-16_000100/
├── tests/                      # ✅ ALL tests here
├── data/                       # ✅ Data files only
└── archive/                   # ✅ Old logs/artifacts
```

**For detailed rules, see sections below:**
- Code Organization: Section 4.1 (Code Organization)
- Documentation: Section 4.4 (Documentation Standards)
- Results: Section 4 (Results Organization Rule)
- Models: Section 4 (Models Organization Rule)

---

## 4. Project-Specific Rules

### **Financial ML Context**

#### **Time Series Integrity**
```python
# ✅ CORRECT - Temporal data splitting
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Validate temporal order
assert train_data.index[-1] < test_data.index[0], "Train/test split not chronological!"

# ❌ AVOID - Random splitting cho time series
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2)  # WRONG!
```

#### **Volatility Calculations**
```python
# ✅ CORRECT - Log returns preferred
returns = np.log(prices / prices.shift(1))
volatility = returns.rolling(window).std()

# ✅ CORRECT - Parkinson estimator cho daily data
parkinson_vol = (np.log(high/low)**2) / (4*np.log(2))

# ❌ AVOID - Raw returns unless specifically needed
returns = (prices - prices.shift(1)) / prices.shift(1)  # Less ideal
```

#### **Data Quality Validation**
```python
# OHLCV consistency checks
def validate_ohlc_data(df):
    """Validate OHLCV data integrity."""
    assert all(df['High'] >= df['Close']), "High must be ≥ Close"
    assert all(df['Close'] >= df['Low']), "Close must be ≥ Low"
    assert all(df['High'] >= df['Low']), "High must be ≥ Low"
    assert all(df['Volume'] > 0), "Volume must be positive"
    return True
```

### **Feature Engineering Rules**

#### **HAR Feature Creation**
```python
# ✅ CORRECT - 22-day confirmed monthly window
def create_har_features(volatility_series):
    """Create HAR features with confirmed 22-day monthly window."""
    features = pd.DataFrame({
        'har_daily_vol': volatility_series.rolling(1).mean(),    # Daily
        'har_weekly_vol': volatility_series.rolling(5).mean(),   # Weekly
        'har_monthly_vol': volatility_series.rolling(22).mean() # Monthly ✅
    })
    return features
```

#### **Target Variable Creation**
```python
# ✅ CORRECT - 5-day focus (Phase 1)
def create_5day_target(volatility_series):
    """Create 5-day ahead target - single horizon focus."""
    target = volatility_series.shift(-5)
    return target

# ❌ AVOID - Multi-target until methodology validated
# def create_multi_targets(volatility_series):
#     return {
#         'target_1d': volatility_series.shift(-1),
#         'target_5d': volatility_series.shift(-5),
#         'target_10d': volatility_series.shift(-10),
#         'target_22d': volatility_series.shift(-22)
#     }
```

### **Model Training Rules**

#### **Reproducibility**
```python
# ✅ CORRECT - Fixed seeds cho reproducibility
def set_random_seeds(seed=42):
    """Set all random seeds cho reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Call before every training run
set_random_seeds(42)
```

#### **Data Leakage Prevention**
```python
# ✅ CORRECT - Scale after splitting
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Fit scaler on training data only
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)  # Use same scaler

# ❌ AVOID - Global scaling causes leakage
scaler = StandardScaler()
all_scaled = scaler.fit_transform(data)  # LEAKAGE!
train_scaled, test_scaled = train_test_split(all_scaled)  # COMPOUNDED LEAKAGE
```

#### **Overfitting Detection**
```python
# ✅ CORRECT - Monitor train/test gap
def detect_overfitting(model, X_train, X_test, y_train, y_test):
    """Check for overfitting by comparing train vs test performance."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_error = np.mean((y_train - train_pred)**2)
    test_error = np.mean((y_test - test_pred)**2)
    
    overfitting_ratio = test_error / train_error
    if overfitting_ratio > 1.2:  # More than 20% degradation
        print(f"⚠️ WARNING: Potential overfitting (test/train ratio: {overfitting_ratio:.2f})")
        return True
    return False
```

### **Model Evaluation Rules** ⭐ CRITICAL

#### **Mandatory Metrics - All 6 Metrics Required** ⭐

**CRITICAL RULE:** Every model trained in this project MUST be evaluated on ALL 6 metrics below. No exceptions.

```python
# ✅ CORRECT - Calculate ALL 6 metrics for every model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model_comprehensive(y_true, y_pred):
    """
    Calculate ALL 6 mandatory metrics for model comparison.
    
    CRITICAL: This function MUST be called for every model trained.
    Missing any metric makes model comparison incomplete.
    """
    metrics = {
        # Error metrics (lower is better)
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        
        # Variance explained (higher is better)
        'r2': r2_score(y_true, y_pred),
        
        # Academic standard (lower is better)
        'qlike': calculate_qlike(y_true, y_pred),
        
        # Trading metric (higher is better)
        'directional_accuracy': calculate_directional_accuracy(y_true, y_pred)
    }
    
    return metrics

# ❌ WRONG - Only calculating some metrics
def evaluate_model_incomplete(y_true, y_pred):
    """WRONG: Missing metrics means incomplete comparison."""
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }
    # WRONG: Missing MSE, R², QLIKE, Dir Acc!
    return metrics
```

#### **The 6 Mandatory Metrics**

**1. MSE (Mean Squared Error)**
- Formula: MSE = mean((y_true - y_pred)²)
- Purpose: Penalize large errors (squared penalty)
- Use case: Sensitive to outliers, squared error focus
- Lower is better

**2. RMSE (Root Mean Squared Error)**
- Formula: RMSE = √MSE
- Purpose: Same units as target (easy to interpret)
- Use case: General prediction error assessment
- Lower is better

**3. MAE (Mean Absolute Error)**
- Formula: MAE = mean(|y_true - y_pred|)
- Purpose: Robust to outliers (linear penalty)
- Use case: Stable error metric, less sensitive to extreme values
- Lower is better

**4. R² (Variance Explained)**
- Formula: R² = 1 - (SS_res / SS_tot)
- Purpose: Percentage of variance explained by model
- Use case: How much volatility pattern model captures
- Range: 0 to 1 (higher is better, 1 = perfect)
- Typical values in volatility: 0.10-0.20 (10-20% variance)

**5. QLIKE (Academic Standard for Volatility)**
- Formula: QLIKE = mean(y_true/y_pred - log(y_true/y_pred) - 1)
- Purpose: "Stylized favorite of volatility forecasting literature"
- Use case: Academic publication, robust to noisy volatility proxies
- Lower is better
- Reference: Patton (2011) "Volatility Forecast Comparison"

**6. Dir Acc (Directional Accuracy)**
- Formula: Dir Acc = mean(sign(diff(y_true)) == sign(diff(y_pred))) × 100%
- Purpose: Percentage of correct direction predictions
- Use case: Trading applications, trend prediction
- Range: 0-100% (higher is better, 50% = random guess)
- Critical: **NOT sign(y_true) vs sign(y_pred)**, but sign of CHANGES!

#### **Why All 6 Metrics Are Necessary**

**Different metrics serve different purposes:**

```python
# Example: 4 models with different strengths
models = {
    'HAR-R Linear': {
        'mse': 2.63e-7,        # Best error metrics
        'rmse': 0.000513,      # Best prediction error
        'mae': 0.000257,       # Most robust
        'r2': 0.105,          # Worst variance explanation
        'qlike': 1.298,       # Worst academic metric
        'dir_acc': 51.53%      # Worst trading metric
    },
    'Simple LSTM': {
        'mse': 3.39e-7,
        'rmse': 0.000582,
        'mae': 0.000296,
        'r2': 0.155,          # Competitive
        'qlike': 0.578,       # Best academic metric ⭐
        'dir_acc': 67.55%      # Competitive trading
    },
    'LSTM-HAR': {
        'mse': 3.17e-7,
        'rmse': 0.000563,
        'mae': 0.000301,
        'r2': 0.159,          # Best variance explanation ⭐
        'qlike': 0.880,
        'dir_acc': 67.76%
    },
    'Enhanced LSTM-HAR': {
        'mse': 3.64e-7,       # Worst error metrics
        'rmse': 0.000603,     # Highest prediction error
        'mae': 0.000303,
        'r2': 0.136,
        'qlike': 0.587,       # Second best academic
        'dir_acc': 67.90%      # Best trading metric ⭐
    }
}
```

**Key insight: No model wins all metrics!**
- HAR-R Linear wins error metrics (MSE, RMSE, MAE) but loses accuracy
- Simple LSTM wins QLIKE (academic) but competitive in others
- LSTM-HAR wins R² (variance explanation) with balanced performance
- Enhanced LSTM-HAR wins Dir Acc (trading) despite higher error

**If only measuring RMSE:**
- You'd pick HAR-R Linear (best RMSE: 0.000513)
- But miss best trading model (Enhanced: 67.90% Dir Acc)
- And miss best academic model (Simple: 0.578 QLIKE)

**Conclusion: ALL 6 metrics required for complete picture.**

#### **Correct Directional Accuracy Calculation** ⭐ CRITICAL BUG

**❌ WRONG - Common mistake:**
```python
# WRONG: Comparing sign of values (always positive for volatility)
direction_true = np.sign(y_true)   # Always +1 for volatility
direction_pred = np.sign(y_pred)   # Always +1 for volatility
dir_acc = np.mean(direction_true == direction_pred) * 100

# Result: 100% Dir Acc (WRONG!) because volatility is always positive
```

**✅ CORRECT - Compare direction of CHANGES:**
```python
# CORRECT: Compare sign of changes (diff)
actual_changes = np.sign(np.diff(y_true))   # Can be -1, 0, +1
pred_changes = np.sign(np.diff(y_pred))     # Can be -1, 0, +1
dir_acc = np.mean(actual_changes == pred_changes) * 100

# Result: Correct Dir Acc (typically 50-70% for volatility)
```

**Implementation:**
```python
def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy (CORRECT METHOD).
    
    CRITICAL: Must compare direction of CHANGES, not sign of values.
    Volatility is always positive, so sign(y_true) is always +1 = 100%.
    
    Formula: mean(sign(diff(y_true)) == sign(diff(y_pred)))
    """
    # Calculate changes (can be negative, zero, or positive)
    actual_changes = np.sign(np.diff(y_true))
    pred_changes = np.sign(np.diff(y_pred))
    
    # Count agreements
    agreements = np.sum(actual_changes == pred_changes)
    total = len(actual_changes)
    
    # Return percentage
    dir_acc = (agreements / total) * 100
    return dir_acc
```

#### **Evaluation Workflow - Mandatory Steps**

**Step 1: Generate predictions**
```python
# After training model
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        predictions = model(X_batch)
        
        # Inverse transform (if using scaled data)
        predictions_np = scaler.inverse_transform(predictions.cpu().numpy())
        targets_np = scaler.inverse_transform(y_batch.numpy().reshape(-1, 1))
        
        all_predictions.extend(predictions_np.flatten())
        all_targets.extend(targets_np.flatten())

y_pred = np.array(all_predictions)
y_true = np.array(all_targets)
```

**Step 2: Calculate ALL 6 metrics**
```python
# Calculate ALL 6 metrics (MANDATORY)
metrics = evaluate_model_comprehensive(y_true, y_pred)

# Expected output structure:
# {
#     'mse': float,
#     'rmse': float,
#     'mae': float,
#     'r2': float,
#     'qlike': float,
#     'directional_accuracy': float
# }
```

**Step 3: Save results to JSON**
```python
import json
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

# Create results dict
results = {
    'model': 'Model Name',
    'timestamp': timestamp,
    'features': 'Feature description',
    'configuration': {...},
    'test_metrics': {
        'mse': float(metrics['mse']),
        'rmse': float(metrics['rmse']),
        'mae': float(metrics['mae']),
        'r2': float(metrics['r2']),
        'qlike': float(metrics['qlike']),
        'directional_accuracy': float(metrics['directional_accuracy'])
    }
}

# Save to results directory
results_file = f'results/model_results_{timestamp}.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
```

**Step 4: Compare with existing models**
```python
# Load all model results
import glob
import pandas as pd

all_results = []
for file in glob.glob('results/*_results_*.json'):
    with open(file, 'r') as f:
        data = json.load(f)
        all_results.append(data)

# Create comparison table
comparison_df = pd.DataFrame([
    {
        'Model': r['model'],
        'RMSE': r['test_metrics']['rmse'],
        'MAE': r['test_metrics']['mae'],
        'R²': r['test_metrics']['r2'],
        'QLIKE': r['test_metrics']['qlike'],
        'Dir Acc': r['test_metrics']['directional_accuracy']
    }
    for r in all_results
])

# Sort by different metrics for different use cases
print("Ranking by Trading (Dir Acc):")
print(comparison_df.sort_values('Dir Acc', ascending=False))

print("\nRanking by Academic (QLIKE):")
print(comparison_df.sort_values('QLIKE', ascending=True))

print("\nRanking by Research (R²):")
print(comparison_df.sort_values('R²', ascending=False))

print("\nRanking by Error (RMSE):")
print(comparison_df.sort_values('RMSE', ascending=True))
```

#### **Model Comparison Standards**

**Every model training run MUST:**
1. ✅ Calculate ALL 6 metrics (no exceptions)
2. ✅ Save results to JSON with timestamp
3. ✅ Compare with previous models
4. ✅ Document winner for each use case:
   - Trading (Dir Acc focus)
   - Academic (QLIKE focus)
   - Research (R² focus)
   - Baseline (RMSE focus)

**Use Case Rankings:**
- **Trading:** Sort by `directional_accuracy` (descending) → Highest wins
- **Academic:** Sort by `qlike` (ascending) → Lowest wins
- **Research:** Sort by `r2` (descending) → Highest wins
- **Baseline:** Sort by `rmse` (ascending) → Lowest wins

#### **Common Anti-Patterns to Avoid**

**❌ Missing Metrics:**
```python
# WRONG: Only calculating RMSE and MAE
metrics = {
    'rmse': 0.000513,
    'mae': 0.000257
}
# Missing: MSE, R², QLIKE, Dir Acc → Incomplete comparison
```

**❌ Wrong Dir Acc Calculation:**
```python
# WRONG: Sign of values instead of sign of changes
dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
# Result: Always ~100% for volatility (values always positive)
```

**❌ Not Saving Results:**
```python
# WRONG: Only printing metrics
print(f"RMSE: {rmse:.6f}")
print(f"Dir Acc: {dir_acc:.2f}%")
# Results lost after session ends, no comparison possible
```

**❌ No Model Comparison:**
```python
# WRONG: Training model in isolation
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"Test RMSE: {rmse:.6f}")
# No comparison with baselines, no context if performance is good or bad
```

### **Documentation Standards**

#### **Function Documentation**
```python
def calculate_parkinson_volatility(df):
    """
    Calculate Parkinson volatility estimator from daily OHLCV data.

    Formula: σ² = (log(H/L)²) / (4*log(2))

    This estimator is preferred cho daily data because:
    - Uses intraday range information (High-Low)
    - More efficient than close-to-close estimator
    - Robust cho daily frequency data
    - Industry standard cho range-based volatility

    Args:
        df: DataFrame containing 'High' and 'Low' columns
             Must be validated OHLCV data (High ≥ Close ≥ Low)

    Returns:
        pd.Series: Parkinson volatility values

    Raises:
        ValueError: If df contains NaN values
        ValueError: If High < Low (OHLCV inconsistency)

    Example:
        >>> data = pd.DataFrame({'High': [102, 105], 'Low': [98, 100]})
        >>> parkinson_vol = calculate_parkinson_volatility(data)
        >>> print(f"Mean Parkinson vol: {parkinson_vol.mean():.6f}")

    Note:
        Parkinson estimator assumes continuous trading. For stocks with
        trading halts, consider alternative estimators.
    """
    high = df['High']
    low = df['Low']
    return (np.log(high / low) ** 2) / (4 * np.log(2))
```

### **Code Organization**

#### **Repository Structure Rules** ⭐ CRITICAL

**Mandatory Structure:**
```
project_root/
├── src/                        # All production code MUST be here
│   ├── common/                 # Code shared across all baselines
│   │   ├── data_processing.py  # Parkinson volatility, data validation
│   │   ├── evaluation.py       # QLIKE, RMSE, directional accuracy
│   │   └── utils.py            # Common utilities
│   ├── har_baseline/           # HAR-R model specific code
│   │   ├── model.py            # HAR-R model implementation
│   │   ├── train.py            # HAR-R training logic
│   │   └── features.py         # HAR feature engineering
│   ├── lstm_baseline/          # LSTM model specific code
│   │   ├── model.py            # LSTM model implementation
│   │   ├── train.py            # LSTM training logic
│   │   └── dataset.py          # LSTM dataset class
│   └── [other_baseline]/       # Future baselines follow same pattern
├── tests/                      # All test code MUST be here
│   ├── test_common/            # Tests for common code
│   ├── test_har_baseline/      # Tests for HAR baseline
│   └── test_lstm_baseline/      # Tests for LSTM baseline
├── experiment/                 # Temporary/experimental code
│   └── [temp_scripts]/         # One-off experiments, exploratory code
├── process_data.py             # Main data processing script (root level)
├── train_simple_lstm.py        # Main training script (root level)
└── train_har_baseline.py       # Main training script (root level)
```

**Strict Rules:**
1. **All production code MUST be in `src/`** - **NO .py files in root directory**
2. **Subfolders for each baseline** - `src/har_baseline/`, `src/lstm_baseline/`, etc.
3. **Common code in `src/common/`** - Functions used by multiple baselines
4. **Tests in `tests/`** - All test code, organized by baseline
5. **Experimental code in `src/experiment/`** - Exploratory scripts (NOT in root)
6. **Minimal code in `src/`** - **UPDATE existing files, DON'T create new ones unless necessary**
7. **Main scripts in `src/`** - Even orchestration scripts must be in appropriate subfolders

**Updated Structure:**
```
project_root/
├── src/                        # ALL code must be here (NO .py files in root!)
│   ├── common/                 # Shared code across baselines
│   ├── har_baseline/           # HAR-R baseline
│   ├── lstm_baseline/          # LSTM baseline
│   ├── experiment/             # Experimental/exploratory code
│   └── [other_baseline]/       # Other baselines
├── tests/                      # Test code
├── results/                    # ALL results go here
│   ├── simple_lstm_2026-06-17_143000/  # Timestamp + model name
│   ├── har_baseline_2026-06-17_150000/
│   └── ...
└── data/                       # Data files only
```

**Update vs Create Rule:**
- ✅ **UPDATE existing files** when adding/modifying functionality
- ❌ **DON'T create new files** unless absolutely necessary
- Example: If adding a new metric to evaluation, UPDATE `src/common/evaluation.py`
- Example: If fixing LSTM training bug, UPDATE `src/lstm_baseline/train.py`
- Only create new files for entirely new baselines or major features

**Results Organization Rule:**
- ✅ **ALL results MUST go in `results/` folder**
- ✅ **Subfolder/filename MUST include timestamp + model name**
- ✅ **Format: `{model_name}_{YYYY-MM-DD_HHMMSS}/`**
- Examples:
  - `results/simple_lstm_2026-06-17_143000/`
  - `results/har_baseline_2026-06-17_150000/`
  - `results/enhanced_lstm_2026-06-17_163045/`
- Files inside result folder:
  - `test_metrics.csv`
  - `training_curves.png`
  - `best_model.pth`

**Models Organization Rule:**
- ✅ **ALL trained models MUST go in `models/` subdirectories**
- ✅ **Subfolder/filename MUST include timestamp + baseline name**
- ✅ **Format: `{baseline}_{YYYY-MM-DD_HHMMSS}/` or `{TICKER}_{baseline}_{YYYY-MM-DD}.{ext}`**
- Structure:
  ```
  models/
  ├── har_baseline_2026-06-15_231200/  # HAR-R models with timestamp
  │   ├── ACB_har_r_model.pkl
  │   ├── BCM_har_r_model.pkl
  │   └── ... (30 stocks)
  ├── lstm_baseline_2026-06-17_225000/  # LSTM models with timestamp
  │   ├── simple_lstm.pth               # Best model
  │   ├── VCB_lstm_model.pth
  │   └── training_history.json
  └── archive/                           # Old models
      └── lstm_baseline_2026-06-16/
  ```
- Examples:
  - ✅ `models/har_baseline_2026-06-15/ACB_har_r_model.pkl`
  - ✅ `models/lstm_baseline_2026-06-17/simple_lstm.pth`
  - ❌ `models/har_baseline/ACB_model.pkl` (NO timestamp!)
  - ❌ `model.pkl` in root (WRONG!)

**Documentation Organization Rule:**
- ✅ **Only 3 .md files in root directory:**
  - `README.md` - Project overview and quick start
  - `CLAUDE.md` - Project rules and AI context
  - `project-context.md` - Project context and background
- ✅ **ALL other .md files MUST go in `docs/` subdirectories**
- ✅ **Organize by topic/model in subdirectories**
- Structure:
  ```
  docs/
  ├── lstm/                    # LSTM model documentation
  │   ├── ENHANCED_LSTM_GUIDE.md
  │   ├── LSTM_ARCHITECTURE_ANALYSIS.md
  │   └── LSTM_COMPARISON_*.md
  ├── har/                     # HAR-R baseline documentation
  │   └── HAR_FEATURES_GUIDE.md
  ├── project/                 # Project management docs
  │   └── REFACTOR_SUMMARY.md
  ├── baseline/                # Other baselines documentation
  └── common-rules/            # ML/DS common rules
  ```
- Examples:
  - ✅ `project-context.md` in root (ALLOWED - project context)
  - ✅ `docs/lstm/ENHANCED_LSTM_GUIDE.md`
  - ✅ `docs/project/REFACTOR_SUMMARY.md`
  - ❌ `ENHANCED_LSTM_GUIDE.md` in root (WRONG!)
  - ❌ `LSTM_ARCHITECTURE_ANALYSIS.md` in root (WRONG!)

**Examples:**

✅ CORRECT:
```python
# Common evaluation metrics used by both HAR and LSTM
# Location: src/common/evaluation.py
def qlike_loss(y_true, y_pred):
    # Implementation shared across baselines
    pass

# HAR-specific training logic
# Location: src/har_baseline/train.py
def train_har_model(X, y):
    from src.common.evaluation import qlike_loss  # Import from common
    # HAR-specific implementation
    pass
```

❌ WRONG:
```python
# DON'T: Put common functions in baseline-specific folders
# Location: src/har_baseline/evaluation.py  # WRONG!
def qlike_loss(y_true, y_pred):
    # This should be in src/common/evaluation.py
    pass

# DON'T: Create duplicate functions
# Location: src/lstm_baseline/evaluation.py  # WRONG!
def qlike_loss(y_true, y_pred):
    # Duplicate code - use src/common/evaluation.py instead
    pass
```

#### **File Layout (Newspaper Principle)**
```python
# 1. Imports (grouped: stdlib, third-party, local)
import random
from typing import Dict, List
import numpy as np
import torch

# 2. Constants (ALL_CAPS)
RANDOM_SEED = 42
HORIZON_DAYS = 5
HAR_WINDOWS = {'daily': 1, 'weekly': 5, 'monthly': 22}

# 3. High-level functions (main orchestration)
def main():
    """Main training pipeline orchestration."""
    data = load_data()
    model = train_model(data)
    evaluate_model(model)

# 4. Mid-level functions (specific logic)
def load_data():
    """Load and preprocess training dataset."""
    pass

def train_model(data):
    """Train model with processed data."""
    pass

# 5. Low-level functions (helpers, utilities)
def _validate_inputs(data):
    """Internal helper cho input validation."""
    pass
```

---

## 4. Technical Architecture

### **System Architecture**
```
Data Input (OHLCV) → Volatility Calculation → HAR Features → Model Training → 5-Day Predictions
        ↓                   ↓                    ↓                ↓              ↓
   Parkinson Vol        HAR-R Features      Linear/LSTM      QLIKE Loss     Evaluation
        ↓                   ↓                    ↓                ↓              ↓
  Quality Checks      Validation         Training         Metrics       Deployment
```

### **Technology Stack**
- **Language:** Python 3.11+
- **Core Libraries:** pandas, numpy, scikit-learn
- **Deep Learning:** PyTorch (cho LSTM/GNN models)
- **Time Series:** statsmodels, pmdarima
- **Experiment Tracking:** MLflow
- **Testing:** pytest, pytest-cov

### **Key Design Decisions**

#### **Single-Horizon Focus**
```python
# Phase 1: 5-day forecasts only (validate methodology)
FORECAST_CONFIG = {
    'horizon': 5,
    'target_column': 'target_5d',
    'features': ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol'],
    'loss_function': 'QLIKE'
}
```

#### **22-Day Consistency**
```python
# Confirmed 22-day consistency across feature engineering & forecasting
HAR_CONFIG = {
    'har_monthly_feature': 22,  # Lookback: average of past 22 days
    'monthly_forecast': 22,     # Forward: predict 22 days ahead (future)
    'rationale': '~22 trading days per month, academic standard'
}
```

#### **QLIKE Loss Function**
```python
# Primary loss function - academic standard
def qlike_loss(y_true, y_pred, epsilon=1e-8):
    """
    QLIKE Loss - "stylized favorite of volatility forecasting literature"
    
    Formula: L = (1/n) * Σ(y_true/y_pred - log(y_true/y_pred) - 1)
    
    Benefits:
    - Specifically designed cho volatility forecasting
    - Robust to noisy volatility proxies  
    - Asymmetric penalty (underprediction > overprediction)
    - Superior model selection properties
    """
    y_pred = np.maximum(y_pred, epsilon)
    y_true = np.maximum(y_true, epsilon)
    
    ratio = y_true / y_pred
    qlike = ratio - np.log(ratio) - 1
    return np.mean(qlike)
```

---

## 5. Development Workflow

### **Code Review Process**
1. **Self-review:** Check against QUICK_REFERENCE.md checklist
2. **Automated checks:** Run tests, coverage check, pre-flight validation
3. **Peer review:** Team member validates against common rules checklist
4. **Approval:** Merge only when all quality gates pass

### **Quality Gates**
- **Pre-commit:** Tests pass, coverage ≥ 85%, no critical linting issues
- **Pre-merge:** Code review approval, no critical issues, documentation complete
- **Pre-deploy:** Performance benchmarks met (< 100ms inference), monitoring configured

### **Experiment Workflow**
1. **Setup:** Fixed seeds, create MLflow run, document configuration
2. **Training:** Monitor learning curves, save checkpoints every 10 epochs
3. **Evaluation:** Calculate QLIKE, RMSE, MAE, R², directional accuracy
4. **Documentation:** Update experiment logs, save artifacts, document decisions

---

## 6. Project-Specific Success Metrics

### **Model Performance Metrics**
```python
SUCCESS_METRICS = {
    'primary_metrics': {
        'QLIKE': 'Lower is better (academic standard)',
        'RMSE': '< 0.20 cho 5-day forecasts',
        'Directional_Accuracy': '> 55% cho all horizons'
    },
    
    'secondary_metrics': {
        'MAE': '< 0.15 cho 5-day forecasts',
        'R²': '> 0.5 (explain variance)',
        'Theil_U': '< 1.0 (better than random walk)'
    },
    
    'quality_metrics': {
        'Test_Coverage': '≥ 85%',
        'Code_Quality': 'ML/DS rules compliance',
        'Reproducibility': 'Fixed seed consistency',
        'Documentation': 'All public functions docstringed'
    }
}
```

### **Data Quality Metrics**
```python
DATA_QUALITY = {
    'completeness': 'Less than 10% missing data per stock',
    'temporal_integrity': 'Chronological order maintained',
    'ohlc_consistency': 'OHLCV relationships validated',
    'volatility_range': 'Parkinson vol between 0.0001 and 0.05',
    'feature_correlation': 'No perfect multicollinearity (r < 0.99)'
}
```

---

## 7. Tools & Automation

### **Pre-commit Configuration**
```bash
# .pre-commit-config.yaml (to be set up)
repos:
  - repo: local
    hooks:
      - id: check-imports
        name: Check imports using ML/DS common rules
      - id: test-coverage  
        name: Check test coverage (85%+ target)
      - id: check-docstrings
        name: Verify all public functions have docstrings
      - id: pre-flight-env
        name: Run environment checks before expensive operations
```

### **Automated Quality Checks**
```bash
# Integration with ML/DS common rules validation
python -m ml_ds_common_rules.code_quality.validate_imports src/
python -m ml_ds_common_rules.testing.pre_flight_env
pytest --cov=src --cov-report=html --cov-fail-under=85
```

---

## 8. Monitoring & Continuous Improvement

### **Quality Tracking**
- **Weekly metrics:** Code coverage trend, bug rate, code review cycle time
- **Monthly review:** Common rules updates, process improvements
- **Quarterly audit:** Remove deprecated practices, add new learnings

### **Team Workflow**
- **Code reviews:** Always reference common rules checklist
- **Retrospectives:** Discuss what's working, what's not
- **Training:** Regular sessions on clean code practices
- **Documentation updates:** Keep docs/common-rules/ synced with latest improvements

---

## 9. Quick Reference cho Development

### **Daily Development Checklist**
```markdown
## Before Committing Code
- [ ] Variable names descriptive (not x, y, data)?
- [ ] Function names specific (not get, handle, process)?
- [ ] Functions small (< 30 lines)?
- [ ] No side effects in functions?
- [ ] Docstrings cho public functions?
- [ ] Tests added/updated?
- [ ] No data leakage (temporal split validated)?
- [ ] Fixed random seed (42) used?

## Before Merging to Main
- [ ] All tests pass?
- [ ] Code coverage > 85%?
- [ ] Documentation updated?
- [ ] Common rules checklist passed?
- [ ] Performance benchmarks met?
```

### **Common Patterns cho This Project**
```python
# Volatility calculation
volatility = calculate_parkinson_volatility(ohlc_data)

# HAR feature creation (22-day confirmed)
har_features = create_har_features(volatility, windows=[1,5,22])

# 5-day target creation
target = volatility.shift(-5)

# Model training with reproducibility
set_random_seeds(42)
model = train_har_model(X_train, y_train)

# Evaluation with QLIKE (academic standard)
loss = qlike_loss(y_true, y_predictions)
metrics = evaluate_model(y_true, predictions)
```

---

## 10. Contact & Support

### **Project Information**
- **Repository:** stock_vol_prediction01
- **Documentation:** `docs/common-rules/` for ML/DS rules
- **Common Rules:** `docs/common-rules/COMMON_RULES.md` 
- **Quick Reference:** `docs/common-rules/QUICK_REFERENCE.md`

### **Getting Help**
- **Common Rules Questions:** See `docs/common-rules/README.md`
- **Integration Issues:** See `docs/common-rules/INTEGRATION_GUIDE.md`
- **Project Questions:** See project-specific sections above

---

**Last Updated:** 2026-06-18  
**Version:** 2.0  
**Status:** Development Phase - Ready for Sprint 1

**This document follows ML/DS common clean code rules and serves as the main reference cho the stock volatility prediction project.**

**Changes in v2.0:**
- ✅ Added Pythonic Code Patterns section (EAFP vs LBYL, Context Managers, Properties)
- ✅ Added Code Quality Levels (Total Mess, Readable, Production)
- ✅ Added comprehensive Anti-Patterns section (naming, functions, comments)
- ✅ Added detailed Refactoring Rules (when and how to refactor)
- ✅ Added Learning Curves and Overfitting Detection (mandatory visualization)
- ✅ Added File Management and Archiving rules
- ✅ Added Generated Files Naming Convention (timestamp rules)
- ✅ Enhanced Function Design section with more examples
- ✅ Added PEP Standards compliance section