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

**Reference:** `docs/common-rules/COMMON_RULES.md`  
**Quick Reference:** `docs/common-rules/QUICK_REFERENCE.md`

### **Core Principles**
1. **Code is read much more than written** - Write for future readers
2. **Leave code better than you found it** - Boy scouts rule
3. **Keep it simple** - Simple > Clever
4. **Match quality to maturity** - Don't over-engineer POCs

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

#### **Research Best Practices**
- **Version control:** Track all experiments with git + MLflow
- **Reproducibility:** Fixed random seeds (42)
- **Hyperparameter tracking:** Document all experiments
- **Checkpoint saving:** Save intermediate results
- **Learning curves:** Plot training/validation curves (mandatory)

---

## 3. Project-Specific Rules

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

#### **Module Structure**
```
src/
├── data_processing.py          # Volatility calculation, HAR features
├── model_training.py           # HAR-R baseline, LSTM models
├── evaluation.py              # QLIKE, RMSE, directional accuracy
└── utils.py                   # Helper functions
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

**Last Updated:** 2026-06-15  
**Version:** 1.0  
**Status:** Development Phase - Ready for Sprint 1

**This document follows ML/DS common clean code rules and serves as the main reference cho the stock volatility prediction project.**