# Technical Configuration: HAR-R Model for VN30 Volatility Prediction

**Project:** Stock Volatility Prediction for VN30 Stocks  
**Configuration Version:** 1.0  
**Last Updated:** 2026-06-15  
**Status:** CONFIRMED ✅

---

## 🎯 HAR Model Configuration

### Core HAR Feature Windows

```python
HAR_CONFIG = {
    'model_type': 'HAR-R',  # Range-based Heterogeneous Autoregressive
    'volatility_estimator': 'parkinson',
    'features': {
        'daily': {
            'window': 1,
            'description': 'Daily volatility - lag 1 trading day',
            'calculation': 'rolling(1).mean()',
            'purpose': 'Short-term volatility component'
        },
        'weekly': {
            'window': 5,
            'description': 'Weekly volatility - 5 trading days average',
            'calculation': 'rolling(5).mean()',
            'purpose': 'Medium-term volatility component'
        },
        'monthly': {
            'window': 22,  # ✅ CONFIRMED CONFIGURATION
            'description': 'Monthly volatility - 22 trading days average',
            'calculation': 'rolling(22).mean()',
            'purpose': 'Long-term volatility component'
        }
    }
}
```

### Volatility Calculation Method

**Parkinson Estimator (Primary):**
```python
def calculate_parkinson_volatility(df):
    """
    Calculate Parkinson volatility from daily OHLC data.
    
    Formula: σ² = (log(H/L)²) / (4*log(2))
    
    Advantages:
    - Uses intraday range information (High-Low)
    - More efficient than close-to-close estimator
    - Robust for daily data frequency
    - Industry standard for range-based volatility
    
    Args:
        df: DataFrame with High, Low columns
        
    Returns:
        Series: Parkinson volatility values
    """
    high = df['High']
    low = df['Low']
    return (np.log(high / low) ** 2) / (4 * np.log(2))
```

**Alternative Estimators (Fallback):**
```python
# Garman-Klass Estimator
def calculate_gk_volatility(df):
    high, low, close, open_price = df['High'], df['Low'], df['Close'], df['Open']
    return 0.5 * (np.log(high/low)**2) - (2*np.log(2)-1) * (np.log(close/open_price)**2)

# Squared Returns (Simple RV Proxy)
def calculate_squared_returns(df):
    returns = np.log(df['Close'] / df['Close'].shift(1))
    return returns ** 2

# Absolute Returns (Robust Alternative)
def calculate_absolute_returns(df):
    returns = np.log(df['Close'] / df['Close'].shift(1))
    return np.abs(returns)
```

---

## 📊 Multi-Horizon Forecasting Configuration

### Prediction Horizons

```python
FORECAST_HORIZONS = {
    '1d': {'days': 1, 'description': '1-day ahead forecast'},
    '1w': {'days': 5, 'description': '1-week ahead forecast'},
    '2w': {'days': 10, 'description': '2-week ahead forecast'},
    '1m': {'days': 22, 'description': '1-month ahead forecast'}  # ✅ 22 DAYS CONFIRMED
}
```

**✅ 22-DAY CONSISTENCY CONFIGURATION:**
- **HAR Monthly Feature:** 22-day lookback (average of past 22 trading days)
- **Monthly Forecast:** 22-day forward prediction (predict 22 days ahead)
- **Academic Standard:** ~22 trading days per month
- **Benefits:** Consistency, literature alignment, uniform implementation

### Target Variable Construction

```python
def create_forecast_targets(volatility_series, horizons=[1, 5, 10, 22]):
    """
    Create multi-horizon forecast targets.
    
    For each horizon h, target[t] = volatility[t + h]
    
    Args:
        volatility_series: Calculated Parkinson volatility
        horizons: List of forecast horizons in days [1, 5, 10, 22] ✅
        
    Returns:
        DataFrame: Columns for each horizon target
    """
    targets = pd.DataFrame()
    for h in horizons:
        targets[f'target_{h}d'] = volatility_series.shift(-h)
    return targets
```

---

## 🔧 Feature Engineering Pipeline

### Complete Feature Set

```python
FEATURE_ENGINEERING_CONFIG = {
    'har_features': {
        'daily_vol': 'rolling(1).mean()',
        'weekly_vol': 'rolling(5).mean()', 
        'monthly_vol': 'rolling(22).mean()'  # ✅ 22 days confirmed
    },
    
    'lagged_features': {
        'returns_lag_1': 'log(Close/Close.shift(1)).shift(1)',
        'returns_lag_5': 'log(Close/Close.shift(1)).shift(5)',
        'returns_lag_10': 'log(Close/Close.shift(1)).shift(10)',
        'returns_lag_20': 'log(Close/Close.shift(1)).shift(20)'
    },
    
    'volume_features': {
        'volume_ma_5': 'rolling(5).mean()',
        'volume_ratio': 'Volume / rolling(20).mean()'
    },
    
    'technical_indicators': {
        'rsi_14': 'RSI(Close, 14)',
        'macd': 'MACD(Close)',
        'bollinger_upper': 'Bollinger_Bands(Close)[0]',
        'bollinger_lower': 'Bollinger_Bands(Close)[2]'
    },
    
    'temporal_features': {
        'day_of_week': 'dt.dayofweek',
        'month': 'dt.month',
        'quarter': 'dt.quarter',
        'is_month_end': 'dt.is_month_end'
    }
}
```

### Feature Processing Steps

```python
def process_features(df, config=FEATURE_ENGINEERING_CONFIG):
    """
    Complete feature engineering pipeline.
    
    Steps:
    1. Calculate Parkinson volatility
    2. Create HAR features (1, 5, 22 day windows)
    3. Add lagged returns
    4. Add volume indicators
    5. Add technical indicators
    6. Add temporal features
    7. Handle missing values
    8. Validate feature consistency
    
    Args:
        df: Raw OHLCV DataFrame
        config: Feature engineering configuration
        
    Returns:
        DataFrame: Processed features ready for modeling
    """
    # Step 1: Calculate base volatility
    parkinson_vol = calculate_parkinson_volatility(df)
    
    # Step 2: HAR features (confirmed 22-day monthly)
    har_features = pd.DataFrame({
        'daily_vol': parkinson_vol.rolling(1).mean(),
        'weekly_vol': parkinson_vol.rolling(5).mean(),
        'monthly_vol': parkinson_vol.rolling(22).mean()  # ✅ CONFIRMED
    })
    
    # Additional feature processing...
    return processed_features
```

---

## 🎯 Model Training Configuration

### Data Splitting Strategy

```python
TRAINING_CONFIG = {
    'split_method': 'temporal',  # Time-series appropriate splitting
    'train_ratio': 0.7,         # 70% training
    'val_ratio': 0.15,          # 15% validation  
    'test_ratio': 0.15,         # 15% testing
    
    # Alternative: Chronological split
    'train_end': '2020-12-31',   # Training data until 2020
    'val_end': '2022-12-31',     # Validation data 2021-2022
    'test_start': '2023-01-01',  # Test data from 2023 onwards
    
    'minimum_train_size': 252,   # Minimum 1 year of trading data
    'rolling_window': 252,       # Rolling window for cross-validation
}
```

### Model Hyperparameters

```python
BASELINE_MODEL_CONFIG = {
    'har_r': {
        'model_type': 'LinearRegression',
        'features': ['daily_vol', 'weekly_vol', 'monthly_vol'],
        'normalization': True,
        'fit_intercept': True
    },
    
    'har_r_ridge': {
        'model_type': 'Ridge',
        'alpha': 1.0,
        'features': ['daily_vol', 'weekly_vol', 'monthly_vol'],
        'normalization': True
    },
    
    'har_r_lasso': {
        'model_type': 'Lasso', 
        'alpha': 0.1,
        'features': ['daily_vol', 'weekly_vol', 'monthly_vol'],
        'normalization': True
    }
}
```

---

## 📈 Evaluation Metrics Configuration

```python
EVALUATION_CONFIG = {
    'primary_metrics': {
        'RMSE': 'Root Mean Squared Error',
        'MAE': 'Mean Absolute Error', 
        'R2': 'R-squared',
        'Directional_Accuracy': 'Percentage of correct direction predictions'
    },
    
    'secondary_metrics': {
        'MAPE': 'Mean Absolute Percentage Error',
        'Theil_U': 'Theil\'s U statistic (vs random walk)',
        'Diebold_Mariano': 'Statistical significance test'
    },
    
    'baseline_comparison': {
        'random_walk': 'Previous day volatility',
        'historical_mean': 'Long-term average volatility',
        'garch': 'GARCH(1,1) benchmark'
    },
    
    'thresholds': {
        'rmse_1d_target': 0.15,
        'rmse_20d_target': 0.25,
        'directional_accuracy_min': 0.55,
        'r2_min': 0.6
    }
}
```

---

## 🔍 Data Quality Configuration

### Validation Rules

```python
DATA_QUALITY_CONFIG = {
    'ohlc_consistency': {
        'high_ge_close': 'High >= Close',
        'close_ge_low': 'Close >= Low', 
        'high_ge_low': 'High >= Low',
        'volume_positive': 'Volume > 0'
    },
    
    'missing_data_handling': {
        'max_gap_size': 5,  # Maximum consecutive missing days
        'interpolation_method': 'linear',
        'forward_fill_limit': 5,
        'backward_fill_limit': 2
    },
    
    'outlier_detection': {
        'method': 'iqr',
        'threshold': 3,  # 3 * IQR
        'volatility_bounds': [0.0001, 0.05]  # Reasonable volatility range
    },
    
    'minimum_data_requirements': {
        'min_history_days': 252,  # 1 year of trading data
        'min_valid_observations': 200,
        'max_missing_percentage': 0.10  # Maximum 10% missing data
    }
}
```

---

## 🚀 Production Configuration

### Model Serving Parameters

```python
PRODUCTION_CONFIG = {
    'inference': {
        'batch_size': 30,  # All VN30 stocks
        'max_latency_ms': 100,
        'parallel_processing': True,
        'cache_predictions': True
    },
    
    'model_management': {
        'retraining_frequency': 'monthly',
        'performance_threshold': 0.10,  # 10% degradation triggers retrain
        'rollback_enabled': True,
        'model_versioning': 'semantic'
    },
    
    'monitoring': {
        'prediction_drift_threshold': 0.05,
        'error_rate_alert': 0.01,  # 1% error rate triggers alert
        'latency_p95_ms': 150,
        'memory_limit_mb': 8192
    }
}
```

---

## 📝 Implementation Notes

### Key Configuration Decisions

**✅ 22-Day Monthly Window - CONFIRMED**
- **Rationale:** Standard ~22 trading days per month
- **Benefits:** 
  - Consistent with HAR literature (Corsi, 2009)
  - Balances recent information with statistical stability
  - Industry-accepted approximation
- **Alternatives Considered:** 20 days (simpler), 25 days (conservative)
- **Final Decision:** 22 days ✅

**✅ Parkinson Volatility Estimator - PRIMARY**
- **Rationale:** Efficient use of daily OHLC range data
- **Benefits:** Better than close-to-close, robust for daily data
- **Fallbacks:** Garman-Klass, squared returns

**✅ Multi-Horizon Approach - CONFIRMED**
- **Horizons:** 1, 5, 10, 20 days ahead
- **Method:** Separate models or multi-output approach
- **Evaluation:** Horizon-specific metrics

### Testing Configuration

```python
TESTING_CONFIG = {
    'unit_test_coverage_target': 0.85,
    'integration_test_scenarios': [
        'end_to_end_pipeline',
        'model_training',
        'batch_inference',
        'single_stock_prediction'
    ],
    'performance_tests': {
        'inference_latency_ms': 100,
        'memory_usage_mb': 8192,
        'throughput_predictions_per_second': 300
    }
}
```

---

## 🔗 Configuration Dependencies

This configuration depends on:
- **Data:** Daily OHLCV format for 30 VN30 stocks
- **Libraries:** pandas, numpy, scikit-learn, PyTorch
- **Standards:** ML/DS Common Rules (`D:\bmad-projects\ml-ds-common-rules`)

Configuration file location:
- **Main Config:** `config/model_config.yaml` (to be created)
- **Reference:** `D:\bmad-projects\stock_vol_prediction01\docs\requirements.md`

---

**Configuration Status:** ✅ CONFIRMED AND READY FOR IMPLEMENTATION  
**Next Step:** Begin Sprint 1 - Data Foundation with confirmed configuration