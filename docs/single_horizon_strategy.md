# Single-Horizon Strategy - 5-Day Forecast Focus

**Project:** Stock Volatility Prediction for VN30 Stocks  
**Date:** 2026-06-15  
**Strategy:** Focus on single 5-day horizon first, then expand

---

## 🎯 **STRATEGIC DECISION: 1-HORIZON APPROACH**

### **Why Single Horizon First?**

**✅ ADVANTAGES:**
- **Methodology Validation:** Validate HAR-R approach với single target
- **Simpler Debugging:** Easier to identify issues
- **Faster Iteration:** Quick feedback loops
- **Clear Success Criteria:** Defined metrics for 5-day forecasts
- **Risk Mitigation:** Less complexity, lower failure risk

**❌ MULTI-HORIZON CHALLENGES:**
- **Weight Tuning Complexity:** How to weight 1d vs 5d vs 10d vs 22d?
- **Conflicting Objectives:** Optimizing short-term vs long-term accuracy
- **Evaluation Ambiguity:** Hard to diagnose which horizon fails
- **Implementation Complexity:** Need 4 separate outputs/models
- **Hyperparameter Explosion:** More parameters to tune

---

## 📊 **REVISED IMPLEMENTATION STRATEGY**

### **PHASE 1: 5-DAY HORIZON FOCUS** ⭐

**Target Variable:**
```python
SINGLE_TARGET_CONFIG = {
    'horizon': '5-day ahead',
    'target_column': 'target_5d',
    'description': 'Predict volatility 5 trading days into the future',
    'business_context': 'One-week ahead volatility forecast'
}
```

**Model Configuration:**
```python
SINGLE_HORIZON_MODEL = {
    'input_features': [
        'har_daily_vol',   # HAR feature 1
        'har_weekly_vol',  # HAR feature 2  
        'har_monthly_vol'  # HAR feature 3 (22-day) ✅
    ],
    'target': 'target_5d',  # Single target
    'loss_function': 'QLIKE',  # Academic standard
    'evaluation_metrics': ['QLIKE', 'RMSE', 'MAE', 'R²', 'Directional_Accuracy']
}
```

**Success Criteria:**
```python
SUCCESS_CRITERIA_5D = {
    'QLIKE': 'Lower is better (compare vs baseline)',
    'RMSE': '< 0.20 (5-day forecasts)',
    'MAE': '< 0.15 (5-day forecasts)',
    'R²': '> 0.5 (explain variance)',
    'Directional_Accuracy': '> 55% (correct direction prediction)'
}
```

---

## 🔧 **IMPLEMENTATION PHASES**

### **PHASE 1A: Baseline HAR-R for 5-Day (Week 1-2)**

**Objective:** Validate HAR-R methodology cho 5-day forecasts

**Code Structure:**
```python
def create_5day_target_only(df):
    """Create ONLY 5-day ahead target - simplified approach."""
    
    # Calculate Parkinson volatility
    df['parkinson_vol'] = calculate_parkinson_volatility(df)
    
    # Create HAR features (confirmed 22-day monthly)
    df['har_daily_vol'] = df['parkinson_vol'].rolling(1).mean()
    df['har_weekly_vol'] = df['parkinson_vol'].rolling(5).mean()
    df['har_monthly_vol'] = df['parkinson_vol'].rolling(22).mean()  # ✅
    
    # Create ONLY 5-day target
    df['target_5d'] = df['parkinson_vol'].shift(-5)
    
    # Keep only essential columns
    essential_cols = [
        'Date', 'ticker',
        'parkinson_vol',  # For reference
        'har_daily_vol', 'har_weekly_vol', 'har_monthly_vol',  # Inputs
        'target_5d'  # Single target
    ]
    
    return df[essential_cols]
```

**Model Training:**
```python
def train_har_r_5day(X_train, y_train):
    """Train HAR-R specifically cho 5-day forecasts."""
    
    # Simple linear regression cho baseline
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

# Evaluation
def evaluate_5day_model(model, X_test, y_test):
    """Evaluate 5-day forecast performance."""
    
    predictions = model.predict(X_test)
    
    metrics = {
        'qlike': qlike_loss(y_test, predictions),
        'rmse': np.sqrt(np.mean((y_test - predictions)**2)),
        'mae': np.mean(np.abs(y_test - predictions)),
        'r2': r2_score(y_test, predictions),
        'directional_accuracy': directional_accuracy(y_test, predictions)
    }
    
    return metrics
```

---

### **PHASE 1B: Enhanced Models for 5-Day (Week 3-4)**

**After baseline validation, add complexity:**

```python
ENHANCED_5DAY_MODELS = {
    'HAR-R + Extended Features': {
        'inputs': [
            'har_daily_vol', 'har_weekly_vol', 'har_monthly_vol',
            'return_lag_5', 'volume_ratio', 'rsi_14', 'day_of_week'
        ],
        'target': 'target_5d'
    },
    
    'HAR-R + LSTM': {
        'inputs': ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol'],
        'architecture': 'LSTM with lookback=20',
        'target': 'target_5d'
    },
    
    'Pure LSTM': {
        'inputs': ['parkinson_vol', 'log_return', 'volume_ratio'],
        'architecture': 'LSTM multi-layer',
        'target': 'target_5d'
    }
}
```

---

### **PHASE 2: Multi-Horizon Expansion (Week 5+)**

**Only AFTER validating 5-day approach:**

```python
MULTI_HORIZON_EXPANSION = {
    'condition': '5-day model meets success criteria',
    'additional_horizons': ['1d', '10d', '22d'],
    'strategy': 'Transfer learning from validated 5-day model',
    'approach': 'Use 5-day weights as starting point'
}
```

---

## 🎯 **UPDATED PROJECT STRUCTURE**

### **Sprint 1-2: Single-Horizon Focus**

**File Structure:**
```
src/
├── data_processing.py          # 5-day target creation
├── feature_engineering.py        # HAR features (22-day confirmed)
├── models/
│   ├── baseline_har_r.py        # Linear Regression for 5-day
│   └── evaluation.py             # 5-day specific metrics
└── experiments/
    ├── experiment_5d_baseline/   # Baseline results
    └── experiment_5d_enhanced/   # Enhanced model results
```

**Configuration Files:**
```python
# config/model_config.py
MODEL_CONFIG = {
    'current_focus': '5-day horizon',
    'target_horizon': 5,
    'target_column': 'target_5d',
    'loss_function': 'QLIKE',
    'features': {
        'primary': ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol'],
        'secondary': ['return_lag_5', 'volume_ratio', 'rsi_14']
    }
}
```

---

## 📈 **EXPECTED BENEFITS OF SINGLE-HORIZON APPROACH**

### **1. Faster Time to Value**
- **Week 1-2:** Working 5-day HAR-R baseline
- **Week 3-4:** Enhanced models for 5-day
- **Week 5:** Evaluate and decide on multi-horizon expansion

**vs Multi-Horizon:**
- **Week 1-2:** Still debugging multi-target issues
- **Week 3-4:** Still tuning horizon weights
- **Week 5:** May not have working model for any horizon

### **2. Clear Methodology Validation**
```python
# Single-horizon evaluation is straightforward:
if metrics_5d['rmse'] < 0.20 and metrics_5d['directional_accuracy'] > 0.55:
    print("✅ Methodology validated! Expand to other horizons.")
else:
    print("❌ Fix methodology before expanding.")
```

### **3. Simpler Hyperparameter Tuning**
```python
# Single horizon = simpler search space
HYPERPARAMETERS = {
    'model_type': ['Linear', 'Ridge', 'Lasso'],
    'features': ['minimal', 'enhanced', 'extended'],
    'regularization': [0.1, 1.0, 10.0]
}
```

---

## 🔧 **IMPLEMENTATION CODE**

### **Complete 5-Day Pipeline:**

```python
class FiveDayVolatilityPredictor:
    """Simplified pipeline cho 5-day volatility prediction."""
    
    def __init__(self):
        self.target_horizon = 5  # 5 trading days ahead
        self.model = None
        self.features = ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol']
        
    def prepare_data(self, df):
        """Prepare data cho 5-day prediction only."""
        
        # Calculate volatility
        df['parkinson_vol'] = self.calculate_parkinson(df)
        
        # Create HAR features
        df['har_daily_vol'] = df['parkinson_vol'].rolling(1).mean()
        df['har_weekly_vol'] = df['parkinson_vol'].rolling(5).mean()
        df['har_monthly_vol'] = df['parkinson_vol'].rolling(22).mean()  # ✅
        
        # Create single target
        df['target_5d'] = df['parkinson_vol'].shift(-self.target_horizon)
        
        # Remove rows with missing targets
        df = df.dropna(subset=['target_5d'])
        
        return df
    
    def train(self, df):
        """Train model cho 5-day forecasts."""
        
        X = df[self.features].values
        y = df['target_5d'].values
        
        # Split data chronologically
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_5day(X_test, y_test)
        
        return metrics
    
    def evaluate_5day(self, X_test, y_test):
        """Evaluate 5-day forecast performance."""
        
        predictions = self.model.predict(X_test)
        
        return {
            'qlike': qlike_loss(y_test, predictions),
            'rmse': np.sqrt(np.mean((y_test - predictions)**2)),
            'mae': np.mean(np.abs(y_test - predictions)),
            'r2': r2_score(y_test, predictions),
            'directional_accuracy': directional_accuracy(y_test, predictions)
        }
    
    def predict(self, df):
        """Make 5-day ahead predictions."""
        
        X = df[self.features].values
        predictions_5d = self.model.predict(X)
        
        return predictions_5d
```

---

## 📋 **UPDATED SUCCESS CRITERIA**

### **Phase 1 Success Gates (5-Day Focus)**

**Gate 1: Baseline Performance (Week 2)**
```python
BASELINE_SUCCESS = {
    'rmse_5d': '< 0.20',
    'directional_accuracy_5d': '> 55%',
    'qlike_improvement': 'Better than random walk baseline',
    'code_quality': '85%+ test coverage, ML/DS rules compliance'
}
```

**Gate 2: Enhanced Model (Week 4)**
```python
ENHANCED_SUCCESS = {
    'rmse_improvement': '10%+ better than baseline',
    'directional_accuracy_5d': '> 60%',
    'no_overfitting': 'Train/test gap < 20%',
    'reproducibility': 'Fixed seed consistency'
}
```

**Gate 3: Multi-Horizon Decision (Week 5)**
```python
EXPANSION_DECISION = {
    'if_5d_success': 'Proceed to multi-horizon implementation',
    'else': 'Debug and fix 5-day approach first'
}
```

---

## 🚀 **REVISED PROJECT TIMELINE**

### **Single-Horizon Focus Approach:**

**Week 1-2: 5-Day Baseline**
- HAR-R model cho 5-day forecasts
- QLIKE loss function
- Baseline metrics establishment

**Week 3-4: 5-Day Enhanced Models**  
- Extended features
- LSTM implementation
- Model comparison

**Week 5: Multi-Horizon Expansion** (Conditional)
- If 5-day successful → expand to 1, 5, 10, 22 days
- If not → continue 5-day refinement

**Week 6+: Production Deployment**
- Optimize best performing approach
- API development
- Monitoring setup

---

## ✅ **FINAL RECOMMENDATION**

### **🎯 ADOPT SINGLE-HORIZON STRATEGY:**

**Phase 1: Focus Exclusively on 5-Day Forecasts**
- **Target:** `target_5d` only
- **Features:** HAR features (3 columns)  
- **Model:** Start with Linear Regression
- **Loss:** QLIKE (academic standard)
- **Success:** RMSE < 0.20, Directional Accuracy > 55%

**Phase 2: Multi-Horizon Expansion** (Conditional)
- Only after validating 5-day approach
- Use learnings to inform multi-horizon weights
- Transfer successful methodology

### **🎯 BENEFITS:**
- ✅ **Faster iteration** - Quick validation
- ✅ **Clearer debugging** - Easier issue diagnosis
- ✅ **Lower risk** - Validate before expanding
- ✅ **Better focus** - Single objective optimization

### **🎯 NEXT STEPS:**
1. Implement 5-day data processing
2. Train HAR-R baseline for 5-day
3. Evaluate and iterate
4. Decide on multi-horizon expansion

**Single-horizon approach validated!** 🚀

---

**Configuration Status:** ✅ **UPDATED TO 5-DAY FOCUS  
**Next Action:** Begin Sprint 1 with single-horizon implementation