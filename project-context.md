# Project Context - Stock Volatility Prediction VN30

**Project:** Multi-horizon volatility forecasting for VN30 stocks  
**Focus:** 5-day ahead forecasts (Phase 1)  
**Methodology:** HAR-R with Parkinson volatility, enhanced with LSTM, GNN, TimesFM  
**Last Updated:** 2026-06-15

---

## 🎯 PROJECT OVERVIEW

### Objective
Build robust volatility prediction system for 30 VN30 stocks using daily OHLCV data (2006-2026), implementing HAR methodology adapted for daily frequency.

### Primary Target (Phase 1)
- **5-day ahead volatility forecast** ✅ CURRENT FOCUS
- **Secondary Targets:** 1, 10, 22-day forecasts (future expansion)
- **Input:** Daily OHLCV data for 30 stocks
- **Approach:** HAR-R baseline → Enhanced models (LSTM, GNN, TimesFM)

### Success Criteria
- **RMSE < 0.20** for 5-day forecasts
- **Directional accuracy > 55%**
- **QLIKE loss** (academic standard)
- **Test coverage > 85%**
- **ML/DS common rules compliance**

---

## 📋 KEY CONFIGURATIONS

### ✅ 22-Day Consistency (CONFIRMED)
```python
# HAR monthly feature = monthly forecast horizon
HAR_MONTHLY_FEATURE = 22  # trading days
MONTHLY_FORECAST_HORIZON = 22  # days ahead
```

### Loss Function Priority
```python
LOSS_FUNCTION_PRIORITY = {
    'primary': 'QLIKE',  # "Stylized favorite of volatility literature"
    'secondary': 'MSE',  # Comparison standard
    'tertiary': 'MAE'     # Robustness check
}
```

### Current Focus: 5-Day Horizon
```python
SINGLE_HORIZON_CONFIG = {
    'horizon': '5-day ahead',
    'target_column': 'target_5d',
    'features': ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol'],
    'loss': 'QLIKE'
}
```

---

## 🏗️ TECHNICAL ARCHITECTURE

### Data Processing Pipeline
```
OHLCV (5 columns) → Parkinson Volatility → HAR Features (3) → 51+ Engineered Features
```

### Model Comparison Strategy
```python
MODELS_TO_COMPARE = [
    'HAR-R (Baseline)',           # Linear regression with HAR features
    'LSTM',                        # Deep learning temporal
    'LSTM + GNN',                  # Graph neural network enhancement
    'HAR-X + GNN',                 # Hybrid approach
    'TimesFM'                      # Foundation model
]
```

### Single Model Architecture
- **One model for all 30 stocks** (not individual stock models)
- **Stock identifier as feature** or **panel data approach**
- **Unified training** across all VN30 stocks

---

## 📁 CRITICAL FILE LOCATIONS

### Documentation
- **Main documentation:** `CLAUDE.md` - Project overview, common rules, technical architecture
- **Requirements:** `docs/requirements.md` - Functional and non-functional requirements
- **Technical config:** `docs/technical_config.md` - HAR configuration, feature engineering
- **Data schema:** `docs/data_schema.md` - 51+ features specification
- **Loss functions:** `docs/loss_functions.md` - QLIKE and evaluation metrics
- **Strategy:** `docs/single_horizon_strategy.md` - 5-day focus approach
- **Integration:** `docs/common_rules_integration.md` - ML/DS rules integration

### ML/DS Common Rules
- **Submodule location:** `docs/common-rules/`
- **Main rules:** `docs/common-rules/COMMON_RULES.md`
- **Quick reference:** `docs/common-rules/QUICK_REFERENCE.md`
- **Template:** `docs/common-rules/CLAUDE_TEMPLATE.md`

### Data
- **Raw data:** `data/raw/prices/` - 30 stocks OHLCV files
- **Collection summary:** `data/raw/prices/collection_summary.csv`
- **Processed data:** `data/processed/` (to be created)

---

## 🔧 CRITICAL IMPLEMENTATION RULES

### Financial ML Context
```python
# Time series integrity - chronological split only
train_split = int(len(data) * 0.8)
assert train_index < test_index  # Validate temporal order

# Volatility calculation
parkinson_vol = (np.log(high / low) ** 2) / (4 * np.log(2))

# OHLCV consistency validation
assert all(ohlc['high'] >= ohlc['close'])
assert all(ohlc['close'] >= ohlc['low'])
```

### Naming Conventions (ML/DS Common Rules)
```python
# ✅ CORRECT - Descriptive and clear
volatility_forecast = model.predict(data)
parkinson_volatility = calculate_parkinson_volatility(data)
train_accuracy = 0.95

# ❌ AVOID - Too generic or abbreviated
vol = model.predict(data)
pred = forecast(data)
acc = 0.95
```

### Code Quality Standards
- **Function size:** < 30 lines per function
- **Parameters:** < 3 preferred
- **Single responsibility:** One concern per function
- **Documentation:** Docstrings for public functions
- **Comments:** Explain WHY not HOW

### Testing Requirements
```python
COVERAGE_TARGETS = {
    'overall': 0.85,
    'data_processing': 0.90,
    'model_training': 0.80,
    'integration': 0.30,
    'e2e': 0.10
}
```

### Research Best Practices
- **Fixed seeds:** random.seed(42), np.random.seed(42), torch.manual_seed(42)
- **Experiment tracking:** MLflow for all experiments
- **Reproducibility:** Same seed → same results
- **Learning curves:** Plot training progress (mandatory)
- **Checkpoint saving:** Save intermediate results

---

## 📊 FEATURE ENGINEERING

### HAR Features (Confirmed 22-day)
```python
def create_har_features(volatility_series):
    """Create HAR features with confirmed 22-day monthly window."""
    return pd.DataFrame({
        'har_daily_vol': volatility_series.rolling(1).mean(),
        'har_weekly_vol': volatility_series.rolling(5).mean(),
        'har_monthly_vol': volatility_series.rolling(22).mean()  # ✅ CONFIRMED
    })
```

### Target Variables
```python
def create_forecast_targets(volatility_series):
    """Create multi-horizon targets with 22-day monthly horizon."""
    targets = pd.DataFrame()
    targets['target_1d'] = volatility_series.shift(-1)
    targets['target_5d'] = volatility_series.shift(-5)   # Phase 1 focus
    targets['target_10d'] = volatility_series.shift(-10)
    targets['target_22d'] = volatility_series.shift(-22)  # ✅ CONFIRMED
    return targets
```

### Feature Categories
- **HAR features (3):** Daily, weekly (5d), monthly (22d) volatility
- **Lagged returns (4):** Return lags at 1, 5, 10, 20 days
- **Volume indicators (2):** Volume MA, volume ratio
- **Technical indicators (6):** RSI, MACD, Bollinger Bands
- **Temporal features (5):** Day of week, month, quarter, month-end flag
- **Total:** 51+ engineered features from 5 raw OHLCV columns

---

## 🎯 IMPLEMENTATION STRATEGY

### Phase 1: 5-Day Baseline (Week 1-4)
```python
PHASE_1_GOALS = {
    'Week 1-2': 'HAR-R baseline for 5-day forecasts',
    'Week 3-4': 'Enhanced models (LSTM, extended features)',
    'Success': 'RMSE < 0.20, Directional Accuracy > 55%'
}
```

### Phase 2: Multi-Horizon Expansion (Week 5+, Conditional)
```python
PHASE_2_CONDITION = {
    'if': '5-day model meets success criteria',
    'then': 'Expand to 1, 5, 10, 22-day forecasts',
    'else': 'Continue 5-day refinement'
}
```

### Quality Gates
- **Pre-commit:** Tests pass, coverage sufficient
- **Pre-merge:** Code review approval, no critical issues
- **Pre-deploy:** Performance benchmarks met, documentation complete

---

## 🚀 NEXT ACTIONS

### Immediate (Sprint 1)
1. **Data processing pipeline**
   - Implement Parkinson volatility calculation
   - Create HAR features (22-day confirmed)
   - Generate 5-day targets only
   - Validate OHLCV consistency

2. **HAR-R baseline**
   - Train linear regression with HAR features
   - Evaluate with QLIKE, RMSE, directional accuracy
   - Compare vs random walk baseline

3. **Testing infrastructure**
   - Set up pytest with 85%+ coverage target
   - Create data quality tests
   - Implement reproducibility tests (fixed seed)

### Documentation
- **All functions:** Public functions must have docstrings
- **Experiment tracking:** MLflow for all runs
- **Code reviews:** Use ML/DS common rules checklist
- **Onboarding:** New members read common rules (Day 1)

---

## 📖 KEY REFERENCES

### Academic Sources
- **HAR methodology:** Corsi (2009) - HAR-R for daily data
- **QLIKE loss:** "Stylized favorite of volatility forecasting literature"
- **22-day standard:** ~22 trading days per month (industry convention)

### Project Documentation
- **Project requirements:** `docs/requirements.md`
- **Technical configuration:** `docs/technical_config.md`
- **ML/DS common rules:** `docs/common-rules/COMMON_RULES.md`

### External Resources
- **ML/DS common rules repo:** `D:\bmad-projects\ml-ds-common-rules`
- **HAR-X paper:** `docs/paper/1-s2.0-S1544612323003641-main.pdf`

---

**Project Status:** ✅ Documentation complete, ready for Sprint 1 implementation  
**Current Focus:** 5-day HAR-R baseline with QLIKE loss function  
**Quality Standard:** 85%+ test coverage, ML/DS common rules compliance
