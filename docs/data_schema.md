# Data Processing Schema - Complete Feature Set

**Project:** Stock Volatility Prediction for VN30 Stocks  
**Date:** 2026-06-15  
**Context:** Daily OHLCV data → Complete feature set cho HAR-R baseline & ML models

---

## 📊 **RAW DATA INPUT (5 COLUMNS)**

```python
RAW_COLUMNS = {
    # Basic OHLCV Data from source
    'Date': 'Trading date (datetime)',
    'Open': 'Opening price (float)',
    'High': 'Highest price (float)', 
    'Low': 'Lowest price (float)',
    'Close': 'Closing price (float)',
    'Volume': 'Trading volume (integer)'
}
```

---

## 🔧 **ENGINEERED FEATURES (50+ COLUMNS)**

### **Category 1: Volatility Measures (6 columns)**

**🎯 Primary Volatility Features:**

```python
VOLATILITY_FEATURES = {
    # Primary: Parkinson Volatility (MAIN FEATURE)
    'parkinson_vol': 'Parkinson volatility estimator - PRIMARY TARGET',
    
    # Alternative Volatility Measures
    'garman_klass_vol': 'Garman-Klass volatility estimator',
    'squared_returns': 'Squared log returns (RV proxy)',
    'absolute_returns': 'Absolute log returns (robust alternative)',
    'close_to_close': 'Traditional close-to-close volatility',
    'range_vol': 'Simple range-based volatility (High-Low)'
}
```

**Implementation:**
```python
def calculate_volatility_measures(df):
    """Calculate all volatility measures from OHLCV data."""
    
    # 1. Parkinson Volatility (PRIMARY)
    df['parkinson_vol'] = (np.log(df['High']/df['Low'])**2) / (4*np.log(2))
    
    # 2. Garman-Klass Volatility
    df['garman_klass_vol'] = 0.5 * (np.log(df['High']/df['Low'])**2) - \
                             (2*np.log(2)-1) * (np.log(df['Close']/df['Open'])**2)
    
    # 3. Squared Returns (RV Proxy)
    log_returns = np.log(df['Close']/df['Close'].shift(1))
    df['squared_returns'] = log_returns ** 2
    
    # 4. Absolute Returns (Robust)
    df['absolute_returns'] = np.abs(log_returns)
    
    # 5. Close-to-Close Volatility
    df['close_to_close'] = log_returns.rolling(22).std() * np.sqrt(252)
    
    # 6. Simple Range Volatility
    df['range_vol'] = (df['High'] - df['Low']) / df['Close']
    
    return df
```

---

### **Category 2: HAR Features (3 columns)** ⭐

**🎯 Core HAR Features for Model Input:**

```python
HAR_FEATURES = {
    'har_daily_vol': 'Daily volatility - 1 day lookback (PRIMARY INPUT)',
    'har_weekly_vol': 'Weekly volatility - 5 day lookback',
    'har_monthly_vol': 'Monthly volatility - 22 day lookback ✅ CONFIRMED'
}
```

**Implementation:**
```python
def create_har_features(df, volatility_col='parkinson_vol'):
    """Create HAR features with confirmed 22-day configuration."""
    
    # HAR Features with CONFIRMED 22-day monthly window
    df['har_daily_vol'] = df[volatility_col].rolling(1).mean()   # Daily
    df['har_weekly_vol'] = df[volatility_col].rolling(5).mean()  # Weekly  
    df['har_monthly_vol'] = df[volatility_col].rolling(22).mean() # Monthly ✅ CONFIRMED
    
    return df
```

---

### **Category 3: Return Features (8 columns)**

**🎯 Price Change Features:**

```python
RETURN_FEATURES = {
    # Basic Returns
    'log_return': 'Log returns (Close/Close lag1)',
    'percentage_return': 'Percentage returns ((Close-Close lag1)/Close lag1)',
    
    # Lagged Returns
    'return_lag_1': '1-day lagged returns',
    'return_lag_5': '5-day lagged returns',
    'return_lag_10': '10-day lagged returns',
    'return_lag_22': '22-day lagged returns',
    
    # Return Statistics
    'return_mean_5': '5-day mean of returns',
    'return_std_5': '5-day std of returns'
}
```

**Implementation:**
```python
def calculate_return_features(df):
    """Calculate return-based features."""
    
    # Basic Returns
    df['log_return'] = np.log(df['Close']/df['Close'].shift(1))
    df['percentage_return'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Lagged Returns (matching forecast horizons)
    df['return_lag_1'] = df['log_return'].shift(1)
    df['return_lag_5'] = df['log_return'].shift(5)
    df['return_lag_10'] = df['log_return'].shift(10)
    df['return_lag_22'] = df['log_return'].shift(22)  # ✅ CONFIRMED
    
    # Return Statistics
    df['return_mean_5'] = df['log_return'].rolling(5).mean()
    df['return_std_5'] = df['log_return'].rolling(5).std()
    
    return df
```

---

### **Category 4: Price Level Features (6 columns)**

**🎯 Normalized Price Features:**

```python
PRICE_FEATURES = {
    # Log Prices (better statistical properties)
    'log_close': 'Log of closing price',
    'log_high': 'Log of highest price',
    'log_low': 'Log of lowest price',
    'log_open': 'Log of opening price',
    
    # Price Ratios
    'close_to_open': 'Close/Open ratio (daily price change)',
    'high_low_ratio': 'High/Low ratio (daily range)'
}
```

**Implementation:**
```python
def calculate_price_features(df):
    """Calculate log-transformed price features."""
    
    # Log Prices (better for statistical modeling)
    df['log_close'] = np.log(df['Close'])
    df['log_high'] = np.log(df['High'])
    df['log_low'] = np.log(df['Low'])
    df['log_open'] = np.log(df['Open'])
    
    # Price Ratios
    df['close_to_open'] = df['Close'] / df['Open']
    df['high_low_ratio'] = df['High'] / df['Low']
    
    return df
```

---

### **Category 5: Technical Indicators (12 columns)**

**🎯 Standard Technical Analysis Features:**

```python
TECHNICAL_INDICATORS = {
    # Momentum Indicators
    'rsi_14': 'RSI (Relative Strength Index) - 14 period',
    'rsi_30': 'RSI - 30 period (alternative)',
    
    # Moving Averages
    'ma_5': '5-day Moving Average',
    'ma_10': '10-day Moving Average',
    'ma_22': '22-day Moving Average',  # ✅ CONFIRMED
    
    # Bollinger Bands
    'bollinger_upper': 'Upper Bollinger Band',
    'bollinger_middle': 'Middle Bollinger Band (MA)', 
    'bollinger_lower': 'Lower Bollinger Band',
    'bollinger_width': 'BB Width (Upper-Lower)',
    
    # Momentum
    'momentum_5': '5-day Momentum (Price change)',
    'roc_5': 'Rate of Change - 5 day'
}
```

**Implementation:**
```python
def calculate_technical_indicators(df):
    """Calculate standard technical analysis indicators."""
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['ma_22'] = df['Close'].rolling(22).mean()  # ✅ CONFIRMED
    
    # Bollinger Bands
    df['bollinger_middle'] = df['Close'].rolling(20).mean()
    df['bollinger_std'] = df['Close'].rolling(20).std()
    df['bollinger_upper'] = df['bollinger_middle'] + 2 * df['bollinger_std']
    df['bollinger_lower'] = df['bollinger_middle'] - 2 * df['bollinger_std']
    df['bollinger_width'] = df['bollinger_upper'] - df['bollinger_lower']
    
    # Momentum Indicators
    df['momentum_5'] = df['Close'] - df['Close'].shift(5)
    df['roc_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
    
    return df
```

---

### **Category 6: Volume Features (5 columns)**

**🎯 Trading Volume Analysis:**

```python
VOLUME_FEATURES = {
    'volume_ma_5': '5-day average volume',
    'volume_ma_22': '22-day average volume',  # ✅ CONFIRMED
    'volume_ratio': 'Current volume / 22-day average volume',
    'volume_change': 'Day-over-day volume change',
    'volume_std_5': '5-day volume volatility'
}
```

**Implementation:**
```python
def calculate_volume_features(df):
    """Calculate volume-related features."""
    
    # Moving Averages
    df['volume_ma_5'] = df['Volume'].rolling(5).mean()
    df['volume_ma_22'] = df['Volume'].rolling(22).mean()  # ✅ CONFIRMED
    
    # Volume Ratios
    df['volume_ratio'] = df['Volume'] / df['volume_ma_22']
    df['volume_change'] = df['Volume'].diff()
    df['volume_std_5'] = df['Volume'].rolling(5).std()
    
    return df
```

---

### **Category 7: Temporal Features (8 columns)**

**🎯 Time-Based Features:**

```python
TEMPORAL_FEATURES = {
    # Day of Week
    'day_of_week': 'Day of week (0=Monday, 4=Friday)',
    'is_monday': 'Monday indicator',
    'is_friday': 'Friday indicator',
    
    # Month Period
    'day_of_month': 'Day of month (1-31)',
    'is_month_start': 'First 3 days of month',
    'is_month_end': 'Last 3 days of month',
    
    # Quarter
    'quarter': 'Quarter (1-4)',
    'is_quarter_end': 'Quarter end indicator'
}
```

**Implementation:**
```python
def calculate_temporal_features(df):
    """Calculate temporal features from Date."""
    
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    df['day_of_month'] = df['Date'].dt.day
    df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
    
    df['quarter'] = df['Date'].dt.quarter
    # Quarter end: Mar(1), Jun(2), Sep(3), Dec(4)
    df['is_quarter_end'] = ((df['Date'].dt.month % 3 == 0) & 
                           (df['Date'].dt.day >= 28)).astype(int)
    
    return df
```

---

### **Category 8: Target Variables (4 columns)** ⭐

**🎯 Multi-Horizon Forecast Targets:**

```python
TARGET_VARIABLES = {
    'target_1d': '1-day ahead volatility target',
    'target_5d': '5-day ahead volatility target',
    'target_10d': '10-day ahead volatility target',
    'target_22d': '22-day ahead volatility target'  # ✅ CONFIRMED
}
```

**Implementation:**
```python
def create_target_variables(df, volatility_col='parkinson_vol'):
    """Create multi-horizon target variables."""
    
    # Target variables with confirmed 22-day monthly horizon
    df['target_1d'] = df[volatility_col].shift(-1)
    df['target_5d'] = df[volatility_col].shift(-5)
    df['target_10d'] = df[volatility_col].shift(-10)
    df['target_22d'] = df[volatility_col].shift(-22)  # ✅ CONFIRMED
    
    return df
```

---

### **Category 9: Stock Identification (2 columns)**

**🎯 Multi-Stock Model Support:**

```python
STOCK_FEATURES = {
    'ticker': 'Stock symbol (e.g., VC, VNM)',
    'stock_id': 'Numerical stock identifier (0-29)'
}
```

---

### **Category 10: Advanced Features (Optional, 8 columns)**

**🎯 Advanced Volatility Features:**

```python
ADVANCED_FEATURES = {
    # Realized Measures (if applicable)
    'realized_semi_variance': 'Downside realized semi-variance',
    'realized_bipower_variation': 'Realized bipower variation',
    
    # Volatility of Volatility
    'vol_of_vol_5': '5-day volatility of volatility',
    'vol_of_vol_22': '22-day volatility of volatility',
    
    # Jump Detection
    'jump_indicator': 'Large price jump indicator',
    'gap_indicator': 'Overnight gap indicator'
}
```

**Implementation:**
```python
def calculate_advanced_features(df):
    """Calculate advanced volatility features (optional)."""
    
    # Realized Semi-Variance (downside risk)
    returns = df['log_return']
    df['realized_semi_variance'] = np.where(returns < 0, returns**2, 0).rolling(22).sum()
    
    # Volatility of Volatility
    df['vol_of_vol_5'] = df['parkinson_vol'].rolling(5).std()
    df['vol_of_vol_22'] = df['parkinson_vol'].rolling(22).std()
    
    # Jump Detection (large price moves)
    df['jump_indicator'] = (np.abs(returns) > returns.std() * 2).astype(int)
    df['gap_indicator'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) > 0.02).astype(int)
    
    return df
```

---

## 📊 **COMPLETE DATA SCHEMA SUMMARY**

### **Final Dataset Structure:**

```python
FINAL_DATA_SCHEMA = {
    # Identification (2 columns)
    'identification': ['Date', 'ticker', 'stock_id'],
    
    # Raw OHLCV (5 columns) 
    'raw_data': ['Open', 'High', 'Low', 'Close', 'Volume'],
    
    # Volatility Measures (6 columns)
    'volatility': ['parkinson_vol', 'garman_klass_vol', 'squared_returns', 
                  'absolute_returns', 'close_to_close', 'range_vol'],
    
    # HAR Features (3 columns) ⭐ PRIMARY INPUTS
    'har_features': ['har_daily_vol', 'har_weekly_vol', 'har_monthly_vol'],
    
    # Return Features (8 columns)
    'returns': ['log_return', 'percentage_return', 'return_lag_1', 'return_lag_5',
                'return_lag_10', 'return_lag_22', 'return_mean_5', 'return_std_5'],
    
    # Price Features (6 columns)
    'prices': ['log_close', 'log_high', 'log_low', 'log_open', 
               'close_to_open', 'high_low_ratio'],
    
    # Technical Indicators (12 columns)
    'technical': ['rsi_14', 'rsi_30', 'ma_5', 'ma_10', 'ma_22',
                 'bollinger_upper', 'bollinger_middle', 'bollinger_lower', 'bollinger_width',
                 'momentum_5', 'roc_5'],
    
    # Volume Features (5 columns)
    'volume': ['volume_ma_5', 'volume_ma_22', 'volume_ratio', 'volume_change', 'volume_std_5'],
    
    # Temporal Features (8 columns)
    'temporal': ['day_of_week', 'is_monday', 'is_friday', 'day_of_month',
                 'is_month_start', 'is_month_end', 'quarter', 'is_quarter_end'],
    
    # Target Variables (4 columns) ⭐ FORECASTING TARGETS
    'targets': ['target_1d', 'target_5d', 'target_10d', 'target_22d'],
    
    # Advanced Features (8 columns - optional)
    'advanced': ['realized_semi_variance', 'realized_bipower_variation',
                 'vol_of_vol_5', 'vol_of_vol_22', 'jump_indicator', 'gap_indicator']
}
```

**TOTAL: 59 COLUMNS** (51 standard + 8 optional advanced)

---

## 🎯 **FEATURE SELECTION FOR MODEL INPUT**

### **Baseline HAR-R Model (Minimal Inputs):**
```python
BASELINE_FEATURES = [
    'har_daily_vol',    # Primary input 1
    'har_weekly_vol',   # Primary input 2
    'har_monthly_vol'   # Primary input 3 ✅
]
```

### **Enhanced HAR-R Model (Extended Inputs):**
```python
ENHANCED_FEATURES = [
    # Core HAR features
    'har_daily_vol', 'har_weekly_vol', 'har_monthly_vol',
    
    # Additional returns
    'return_lag_1', 'return_lag_5', 'return_lag_10', 'return_lag_22',
    
    # Volume indicators
    'volume_ratio',
    
    # Technical indicators
    'rsi_14', 'ma_22', 'bollinger_width'
]
```

### **Neural Network Models (Full Feature Set):**
```python
NEURAL_FEATURES = [
    # All HAR + returns
    'har_daily_vol', 'har_weekly_vol', 'har_monthly_vol',
    'log_return', 'return_lag_1', 'return_lag_5', 'return_lag_10', 'return_lag_22',
    
    # Price features
    'close_to_open', 'high_low_ratio',
    
    # Technical indicators
    'rsi_14', 'ma_5', 'ma_10', 'ma_22', 'bollinger_width',
    
    # Volume features
    'volume_ratio', 'volume_change',
    
    # Temporal features
    'day_of_week', 'is_month_end',
    
    # Volatility measures
    'garman_klass_vol', 'vol_of_vol_5'
]
```

---

## 🔧 **DATA PROCESSING PIPELINE**

### **Complete Processing Function:**

```python
def process_stock_data(raw_df, ticker_name):
    """
    Complete data processing pipeline from raw OHLCV to feature-rich dataset.
    
    Args:
        raw_df: Raw OHLCV DataFrame
        ticker_name: Stock symbol (e.g., 'VCB')
        
    Returns:
        DataFrame: Complete processed dataset with 51+ features
    """
    
    df = raw_df.copy()
    
    # 1. Identification
    df['ticker'] = ticker_name
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Volatility Measures
    df = calculate_volatility_measures(df)
    
    # 3. HAR Features ⭐ PRIMARY
    df = create_har_features(df, volatility_col='parkinson_vol')
    
    # 4. Return Features
    df = calculate_return_features(df)
    
    # 5. Price Features
    df = calculate_price_features(df)
    
    # 6. Technical Indicators
    df = calculate_technical_indicators(df)
    
    # 7. Volume Features
    df = calculate_volume_features(df)
    
    # 8. Temporal Features
    df = calculate_temporal_features(df)
    
    # 9. Target Variables ⭐ FORECASTING TARGETS
    df = create_target_variables(df, volatility_col='parkinson_vol')
    
    # 10. Advanced Features (optional)
    # df = calculate_advanced_features(df)
    
    # 11. Data Quality Checks
    df = perform_data_quality_checks(df)
    
    # 12. Stock ID assignment
    df['stock_id'] = encode_ticker(ticker_name)
    
    return df
```

---

## 📋 **IMPLEMENTATION PRIORITY**

### **Sprint 1: Core Features (Required)**
```python
SPRINT1_FEATURES = [
    'parkinson_vol',           # Primary volatility measure
    'har_daily_vol',           # HAR feature 1
    'har_weekly_vol',          # HAR feature 2
    'har_monthly_vol',         # HAR feature 3 ✅
    'target_1d', 'target_5d',  # Forecast targets
    'target_10d', 'target_22d' # ✅ Updated to 22 days
]
```

### **Sprint 2: Extended Features**
```python
SPRINT2_FEATURES = [
    'log_return', 'return_lag_1', 'return_lag_5',  # Returns
    'volume_ratio', 'rsi_14', 'ma_22',               # Indicators
    'day_of_week', 'is_month_end'                     # Temporal
]
```

### **Sprint 3+: Advanced Features**
```python
ADVANCED_FEATURES = [
    'garman_klass_vol', 'bollinger_width',           # Advanced volatility
    'vol_of_vol_5', 'jump_indicator'                 # Volatility of volatility
]
```

---

## ✅ **FINAL ANSWER TO YOUR QUESTION**

**"Xử lý dữ liệu có thêm các cột nào?"**

**🎯 Answer: From 5 raw OHLCV columns → 51+ engineered columns**

**Critical additions:**
1. **parkinson_vol** - Primary volatility measure
2. **har_daily_vol, har_weekly_vol, har_monthly_vol** - HAR inputs ⭐
3. **target_1d, target_5d, target_10d, target_22d** - Forecast targets ⭐
4. **20+ additional features** cho model enhancement

**Total: 59 engineered features** from 5 raw columns!

Ready to implement data processing pipeline! 🚀