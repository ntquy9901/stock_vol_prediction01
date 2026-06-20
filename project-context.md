# Project Context - Stock Volatility Prediction VN30

**Project:** Multi-horizon volatility forecasting for VN30 stocks
**Focus:** 5-day ahead forecasts (Phase 1)
**Methodology:** HAR-R with Parkinson volatility, enhanced with LSTM, GNN, TimesFM
**Last Updated:** 2026-06-19 (Updated: Standardized hyperparameters & metrics)

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

### ✅ Standard Hyperparameters (ALL Models) - UPDATED 2026-06-19
```python
# Applied to ALL LSTM model training (6 files)
STANDARD_HYPERPARAMETERS = {
    'num_epochs': 70,      # Maximum training epochs (all models)
    'patience': 15,        # Early stopping patience (all models)
    'loss_function': 'MSE',  # Training loss (convex, stable)
    'optimizer': 'Adam',    # Default optimizer
}

# Applied to these files:
# - src/lstm_har_enhanced/train_with_validation.py
# - src/lstm_har_enhanced/train_enhanced.py
# - src/lstm_har_baseline/train_with_validation.py
# - src/lstm_har_baseline/train.py
# - src/lstm_baseline/train_with_validation.py
# - src/lstm_baseline/train.py
```

### Loss Function Priority
```python
LOSS_FUNCTION_PRIORITY = {
    'primary': 'QLIKE',  # "Stylized favorite of volatility literature"
    'secondary': 'MSE',  # Comparison standard (training + evaluation)
    'tertiary': 'MAE'     # Robustness check
}
```

### ✅ Mandatory Metrics (6 Total) - UPDATED 2026-06-19
```python
# ALL models must report these 6 metrics in BOTH console and JSON
MANDATORY_METRICS = {
    1: 'MSE',           # Mean Squared Error (lower is better) ⭐ ADDED
    2: 'RMSE',          # Root Mean Squared Error (lower is better)
    3: 'MAE',           # Mean Absolute Error (lower is better)
    4: 'R²',            # Variance Explained (higher is better)
    5: 'QLIKE',         # Academic standard cho volatility (lower is better)
    6: 'Dir Acc'        # Directional Accuracy (higher is better)
}

# Output requirements:
# 1. Console: Print all 6 metrics for validation + test
# 2. JSON: Save all 6 metrics in validation_metrics, test_metrics, val_test_diff
# 3. Comparison table: Show all 6 metrics with differences
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

### LSTM-GAT Hybrid Architecture (Advanced) 🚀
```python
LSTM_GAT_HYBRID = {
    'temporal_branch': 'LSTM encoder (per-stock temporal learning)',
    'spatial_branch': 'Graph Attention Network (cross-stock relationships)',
    'graph_construction': 'Dynamic correlation + volatility spillover',
    'attention_mechanism': 'Multi-head attention (4-8 heads)',
    'fusion_strategy': 'Concatenate + MLP for final prediction',
    'expected_improvement': 'RMSE 17% ↓, Dir Acc 7% ↑',
    'architecture_doc': 'docs/project/LSTM_GAT_ARCHITECTURE.md',
    'based_on': [
        'TemporalGAT (arXiv 2410.16858v1, 2024)',
        'FSTGAT (MDPI Symmetry, 2024)',
        'STGAT (MDPI Applied Sciences, 2025)'
    ]
}

# Input: (batch, seq_len, 30_stocks, 22_features)
# Output: (batch, 30_stocks, 1_prediction)
```

**Key Innovation:**
- Processes all 30 VN30 stocks simultaneously
- Dynamic graph captures market-wide dependencies
- Attention weights reveal influential stocks

### Single Model Architecture
- **One model for all 30 stocks** (not individual stock models)
- **Stock identifier as feature** or **panel data approach**
- **Unified training** across all VN30 stocks

---

## 📁 CRITICAL FILE LOCATIONS

### Documentation
- **Main documentation:** `CLAUDE.md` - Project overview, common rules, technical architecture
- **LSTM-GAT Architecture:** `docs/project/LSTM_GAT_ARCHITECTURE.md` - Advanced hybrid model design 🚀 NEW
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

### Phase 3: Advanced LSTM-GAT Hybrid (Week 9-12) 🚀
```python
PHASE_3_GOALS = {
    'Week 9': 'Data preparation (technical indicators, graph utilities)',
    'Week 10': 'Model development (LSTM encoder, GAT layers, fusion)',
    'Week 11': 'Training & evaluation (hyperparameter tuning, comparison)',
    'Week 12': 'Analysis & deployment (attention visualization, ablation)',
    'Target': 'RMSE < 0.15, Dir Acc > 75% (vs 0.18, 67.90% current)',
    'Architecture': 'LSTM (temporal) + Graph Attention Network (spatial)',
    'Documentation': 'docs/project/LSTM_GAT_ARCHITECTURE.md'
}

# Key improvements expected
IMPROVEMENT_TARGETS = {
    'RMSE': '0.18 → < 0.15 (17% ↓)',
    'Dir_Acc': '67.90% → > 75% (7% ↑)',
    'QLIKE': '~0.12 → < 0.10 (17% ↓)',
    'R²': '~0.65 → > 0.75 (15% ↑)'
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

---

## 📝 UPDATE HISTORY

### 2026-06-19 - Standardization Update
**Changes:**
- ✅ **Standardized hyperparameters:** 70 epochs, 15 patience for ALL 6 LSTM models
- ✅ **Added MSE as 6th mandatory metric** (was 5 metrics: RMSE, MAE, R², QLIKE, Dir Acc)
- ✅ **Mandatory output format:** All 6 metrics must appear in console + JSON
- ✅ **Updated all training files:** lstm_har_enhanced, lstm_har_baseline, lstm_baseline
- ✅ **Enhanced comparison tables:** Added MSE to all val/test comparisons

**Impact:**
- All models now use consistent hyperparameters for fair comparison
- Complete metrics reporting (all 6: MSE, RMSE, MAE, R², QLIKE, Dir Acc)
- Better reproducibility and model comparison

**Files Updated:**
- `CLAUDE.md` - Added standard hyperparameters section, updated metrics section
- `src/common/evaluation.py` - Added MSE to evaluate_predictions()
- All 6 training files - Updated epochs, patience, and MSE output

### 2026-06-15 - Initial Documentation
**Created:**
- Project context document
- Technical architecture
- Implementation strategy
- Quality standards
