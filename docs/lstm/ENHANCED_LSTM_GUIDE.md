# Enhanced LSTM Architecture for Rich Feature Set

## 🎯 Overview

Enhanced LSTM architecture designed specifically for **18 enriched features** including:
- Realized volatility (RV_5, RV_10, RV_20, RV_30)
- Multiple volatility estimators (Parkinson, Garman-Klass, Close-to-Close)
- Calendar effects (day of week, month end, Tet holidays)
- Overnight returns and log transformations

## 📊 Feature Analysis

### Feature Groups (18 Total)

```
1. PRICE FEATURES (5)
   - open, high, low, close, volume
   - Raw OHLCV data for price movements

2. LOG FEATURES (2)
   - log_close: Log-transformed close prices
   - log_returns: Daily log returns

3. REALIZED VOLATILITY (4)
   - RV_5: 5-day realized volatility
   - RV_10: 10-day realized volatility
   - RV_20: 20-day realized volatility
   - RV_30: 30-day realized volatility

4. VOLATILITY ESTIMATORS (3)
   - parkinson: Parkinson volatility estimator
   - gk: Garman-Klass volatility estimator
   - close_to_close: Close-to-close volatility

5. CALENDAR FEATURES (5)
   - day_of_week: Day of week (0-6)
   - is_monday: Monday indicator
   - is_friday: Friday indicator
   - is_month_end: Month-end indicator
   - is_tet_period: Tet holiday period indicator

6. SPECIAL FEATURES (1)
   - overnight: Overnight returns (close-to-open)
```

## 🏗️ Enhanced Architecture

### Architecture Diagram

![Enhanced LSTM Architecture](enhanced_lstm_architecture.png)

### Detailed Architecture Flow

```
INPUT LAYER (18 features)
    ↓
┌─────────────────────────────────────────────┐
│  FEATURE-SPECIFIC EMBEDDINGS                │
│  ┌────────────┐  ┌────────────┐             │
│  │ Price Embed│  │  Vol Embed  │             │
│  │  OHLCV→16D │  │  RV+Est→32D │             │
│  └────────────┘  └────────────┘             │
│  ┌────────────┐  ┌────────────┐             │
│  │ Cal Embed  │  │ Spec Embed  │             │
│  │ Calendar→8D │  │ Over→4D    │             │
│  └────────────┘  └────────────┘             │
│                                             │
│  COMBINED: 16+32+8+4+2 = 62D               │
│  PROJECT → 128D                             │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  LSTM LAYER 1 (Bidirectional)              │
│  Input: 128D → Hidden: 128×2 = 256D        │
│  Dropout: 0.3                               │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  LSTM LAYER 2 (Bidirectional)              │
│  Input: 256D → Hidden: 128×2 = 256D        │
│  Dropout: 0.3                               │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  MULTI-HEAD ATTENTION (4 heads)             │
│  Focus: Important temporal patterns          │
│  Output: 256D (context-aware)                │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  OUTPUT LAYERS (Progressive Reduction)     │
│  256D → 64D → 16D → 1D                     │
│  ReLU + Dropout + BatchNorm                 │
└─────────────────────────────────────────────┘
    ↓
PREDICTION: Non-negative volatility
```

### Key Improvements vs Basic LSTM

| Aspect | Basic LSTM | Enhanced LSTM | Improvement |
|--------|-----------|---------------|-------------|
| **Features** | 3 HAR features | 18 enriched features | **6× more** |
| **Input Processing** | Direct input | Feature embeddings | **Learned representations** |
| **LSTM Direction** | Unidirectional | Bidirectional | **Past + Future context** |
| **Attention** | None | Multi-head (4 heads) | **Focus on important patterns** |
| **Hidden Size** | 64 units | 128 units | **2× capacity** |
| **Parameters** | ~50K | ~150K | **3× capacity** |
| **Regularization** | Dropout only | Multiple techniques | **Much better** |

## 🔧 Implementation Guide

### Step 1: Data Preparation

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load your enriched data
data = pd.read_csv('enhanced_volatility_data.csv')

# Prepare features and target
feature_cols = [
    'open', 'high', 'low', 'close', 'volume',     # Price (5)
    'log_close', 'log_returns',                    # Log (2)
    'RV_5', 'RV_10', 'RV_20', 'RV_30',           # RV (4)
    'parkinson', 'gk', 'close_to_close',          # Estimators (3)
    'day_of_week', 'is_monday', 'is_friday',     # Calendar (4)
    'is_month_end', 'is_tet_period',             # More calendar (2)
    'overnight'                                   # Special (1)
]

# Extract features and target
features = data[feature_cols].values
target = data['RV_5'].shift(-5)  # 5-day ahead prediction

# Remove NaN values
valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(target))
features = features[valid_mask]
target = target[valid_mask]

# CRITICAL: Scale both inputs AND outputs
feature_scaler = StandardScaler()
features_scaled = feature_scaler.fit_transform(features)

target_scaler = StandardScaler()
target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

print(f"Features shape: {features_scaled.shape}")
print(f"Target shape: {target_scaled.shape}")
```

### Step 2: Sequence Creation

```python
def create_sequences(features, target, seq_length=30):
    """Create sequences for LSTM training."""
    X_sequences = []
    y_sequences = []
    
    for i in range(seq_length, len(features)):
        X_seq = features[i-seq_length:i]  # 30-day lookback
        y_seq = target[i]                  # Predict current volatility
        
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
    
    return np.array(X_sequences), np.array(y_sequences)

# Create sequences
X_seq, y_seq = create_sequences(features_scaled, target_scaled, seq_length=30)

print(f"Sequences created: {len(X_seq)}")
print(f"X shape: {X_seq.shape}")  # (n_samples, 30, 18)
print(f"y shape: {y_seq.shape}")  # (n_samples,)
```

### Step 3: Model Training

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from enhanced_lstm_architecture import EnhancedVolatilityLSTM

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_seq)
y_tensor = torch.FloatTensor(y_seq)

# Create dataloaders
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model
model = EnhancedVolatilityLSTM(
    input_size=18,
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
    bidirectional=True,
    use_attention=True
)

# Training configuration
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Training loop
num_epochs = 200
patience = 15
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch).squeeze()
        loss = criterion(predictions, y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()
    
    val_loss /= len(test_loader)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Save best model
        torch.save(model.state_dict(), 'best_enhanced_lstm.pth')
    else:
        epochs_without_improvement += 1
    
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
```

### Step 4: Prediction and Evaluation

```python
# Load best model
model.load_state_dict(torch.load('best_enhanced_lstm.pth'))
model.eval()

# Generate predictions
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_seq[-len(test_dataset):])
    predictions_scaled = model(X_test_tensor).squeeze().numpy()

# CRITICAL: Inverse transform predictions
predictions = target_scaler.inverse_transform(
    predictions_scaled.reshape(-1, 1)
).flatten()

# Get actual values
y_actual = target[-len(predictions):]

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_actual, predictions))
mae = mean_absolute_error(y_actual, predictions)

print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")

# Calculate QLIKE Loss
def qlike_loss(y_true, y_pred, epsilon=1e-8):
    y_pred = np.maximum(y_pred, epsilon)
    y_true = np.maximum(y_true, epsilon)
    ratio = y_true / y_pred
    qlike = ratio - np.log(ratio) - 1
    return np.mean(qlike)

qlike = qlike_loss(y_actual, predictions)
print(f"QLIKE Loss: {qlike:.6f}")
```

## 📈 Expected Performance

### Performance Targets

With the enhanced architecture and rich features, we expect significant improvements:

| Metric | Basic LSTM | Enhanced LSTM (Expected) | HAR-R Baseline |
|--------|-----------|-------------------------|----------------|
| **QLIKE Loss** | 16,033 ❌ | < 0.5 ✅ | 0.55 |
| **RMSE** | NaN ❌ | < 0.0003 ✅ | 0.0004 |
| **Directional Acc** | N/A | > 55% ✅ | 49.4% |
| **Theil U** | N/A | < 0.85 ✅ | 0.86 |
| **Training Time** | 30s/stock | 60s/stock | < 1s/stock |

### Why Enhanced LSTM Will Work Better

#### 1. **Rich Feature Set**
```
Basic: 3 HAR features (daily, weekly, monthly averages)
Enhanced: 18 features including:
  - 4 Realized volatility measures (different timeframes)
  - 3 Volatility estimators (different calculation methods)
  - 5 Calendar effects (capture seasonality)
  - Price + log features (capture trends)
  - Overnight returns (gap risk)
```

#### 2. **Feature Embeddings**
```python
Basic: Direct input → LSTM
Enhanced: Feature-specific embeddings → LSTM
  - Learn optimal representations for each feature group
  - Allow model to discover relationships between features
  - Better than raw inputs
```

#### 3. **Attention Mechanism**
```python
Basic: Fixed temporal weighting
Enhanced: Multi-head attention
  - Learn which timesteps are important
  - Focus on critical periods (e.g., high volatility days)
  - Capture long-term dependencies
```

#### 4. **Bidirectional Processing**
```python
Basic: Unidirectional (past → future)
Enhanced: Bidirectional (past ↔ future)
  - Capture future context (e.g., upcoming calendar effects)
  - Better understanding of temporal patterns
```

#### 5. **Proper Scaling**
```python
Basic: Features scaled, target NOT scaled → MISMATCH
Enhanced: BOTH features AND target scaled → MATCH
  - Model learns in scaled space
  - Predictions evaluated in scaled space
  - Inverse-transform for final evaluation
```

## 🎯 Configuration Details

### Recommended Hyperparameters

```python
ENHANCED_LSTM_CONFIG = {
    # Architecture
    'input_size': 18,              # 18 features
    'hidden_size': 128,            # 128 hidden units
    'num_layers': 2,               # 2 LSTM layers
    'dropout': 0.3,                # 30% dropout
    'bidirectional': True,         # Bidirectional LSTM
    'use_attention': True,         # Multi-head attention
    
    # Training
    'learning_rate': 0.001,        # Higher LR (better architecture)
    'num_epochs': 200,             # More epochs
    'patience': 15,                # More patience
    'batch_size': 32,
    'seq_length': 30,              # 30-day lookback
    
    # Regularization
    'weight_decay': 1e-5,         # L2 regularization
    'gradient_clip': 1.0,          # Gradient clipping
    
    # Data processing
    'feature_scaling': 'StandardScaler',      # Scale features
    'target_scaling': 'StandardScaler',       # Scale target
    'sequence_length': 30,                    # 30-day windows
    'train_test_split': 0.8,                  # 80/20 split
    
    # Scheduling
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_patience': 5,
    'scheduler_factor': 0.5
}
```

### Sequence Length Recommendation

```python
# Recommended sequence lengths based on feature set:

30 days:  # Default (1 month)
  - Captures monthly patterns
  - Balances context vs overfitting
  - Good for RV_30 features

44 days:  # ~2 months
  - Captures quarterly patterns
  - Better for seasonality
  - More context for calendar effects

60 days:  # ~3 months
  - Captures quarterly patterns
  - Maximum recommended
  - Risk of overfitting beyond this

# Avoid very long sequences (>90 days)
  - Too much historical noise
  - Diminishing returns
  - Increased computational cost
```

## 🔍 Feature Importance Analysis

### Expected Feature Importance Ranking

Based on volatility forecasting literature:

```
1. HIGH IMPORTANCE (Core predictors):
   - RV_5, RV_10 (Short-term realized volatility)
   - log_returns (Daily returns)
   - parkinson, gk (Volatility estimators)
   
2. MEDIUM IMPORTANCE (Context features):
   - RV_20, RV_30 (Long-term volatility)
   - overnight (Gap risk)
   - close_to_close (Daily range)
   
3. LOW IMPORTANCE (Auxiliary features):
   - Calendar effects (day_of_week, is_month_end)
   - OHLCV raw prices
   - is_tet_period (Holiday effects)
```

### Attention Weight Analysis

The multi-head attention mechanism allows us to analyze which timesteps are most important:

```python
# Analyze attention weights
def get_attention_weights(model, X_sequence):
    """Extract attention weights for interpretation."""
    model.eval()
    
    with torch.no_grad():
        # Get LSTM outputs
        lstm_out, _ = model.lstm(model.input_projection(X_sequence))
        
        # Get attention weights
        attention_output, attention_weights = model.attention(
            lstm_out, lstm_out, lstm_out
        )
    
    return attention_weights

# Use to understand which days are most important
attention_weights = get_attention_weights(model, X_sample)
print(f"Attention weights shape: {attention_weights.shape}")
```

## ⚠️ Common Pitfalls to Avoid

### 1. Scaling Mismatch (CRITICAL)
```python
# WRONG:
features_scaled = scaler.fit_transform(features)
predictions = model(features_scaled)  # Scaled predictions
evaluation(y_raw, predictions)  # MISMATCH!

# CORRECT:
features_scaled = feature_scaler.fit_transform(features)
target_scaled = target_scaler.fit_transform(target)
predictions_scaled = model(features_scaled)
predictions = target_scaler.inverse_transform(predictions_scaled)
evaluation(y_raw, predictions)  # MATCH!
```

### 2. Data Leakage
```python
# WRONG:
# Shuffle data before creating sequences
shuffled_indices = np.random.permutation(len(data))

# CORRECT:
# Maintain temporal order
# Create sequences chronologically only
```

### 3. Target Variable Selection
```python
# WRONG:
target = data['close']  # Prices, not volatility

# CORRECT:
target = data['RV_5']  # Realized volatility
# OR target = data['parkinson']  # Volatility estimator
```

### 4. Sequence Length
```python
# WRONG:
seq_length = 5  # Too short for meaningful patterns

# CORRECT:
seq_length = 30  # Capture monthly patterns
seq_length = 60  # Capture quarterly patterns
seq_length = 90  # Maximum recommended
```

## 🚀 Next Steps

### Phase 1: Implementation (Week 1-2)
1. ✅ Implement enhanced data processing
2. ✅ Build Enhanced LSTM architecture
3. ✅ Train on single stock (validate approach)
4. ✅ Compare with HAR-R baseline

### Phase 2: Optimization (Week 3-4)
1. Tune hyperparameters (hidden size, dropout, seq_length)
2. Feature importance analysis
3. Attention weight interpretation
4. Cross-validation across time periods

### Phase 3: Production (Week 5-6)
1. Train on all 30 stocks
2. Deploy best model
3. Monitor performance
4. Compare with alternative models (Random Forest, XGBoost)

## 📊 Comparison with Alternatives

### Enhanced LSTM vs Other Models

| Model | Pros | Cons | When to Use |
|-------|------|------|-------------|
| **Enhanced LSTM** | Captures complex temporal patterns, attention mechanism | Complex, slower training | Rich features, complex patterns |
| **HAR-R Linear** | Simple, fast, interpretable | Limited to linear patterns | Baseline, limited features |
| **Random Forest** | Handles non-linearity, feature importance | No temporal modeling | Tabular data, no seq dependence |
| **XGBoost** | State-of-art performance | No temporal modeling | Tabular data, competitions |
| **GARCH** | Specialized for volatility | Single time series | Pure volatility clustering |

## 🎯 Success Criteria

### Target Metrics

```python
SUCCESS_METRICS = {
    'primary_metrics': {
        'QLIKE': '< 0.5 (better than HAR-R)',
        'RMSE': '< 0.0003',
        'Directional_Accuracy': '> 55%'
    },
    
    'secondary_metrics': {
        'MAE': '< 0.0002',
        'R²': '> 0.3',
        'Theil_U': '< 0.85 (better than random walk)'
    },
    
    'training_metrics': {
        'Convergence': '< 50 epochs',
        'Stability': 'No NaN losses',
        'Overfitting': 'Train/val loss gap < 20%'
    }
}
```

### Evaluation Checklist

```python
CHECKLIST = {
    'data_quality': [
        'No missing values in features',
        'Proper scaling applied',
        'Temporal order maintained',
        'No data leakage'
    ],
    
    'model_performance': [
        'QLIKE < 0.5',
        'RMSE < 0.0003',
        'Directional accuracy > 55%',
        'Stable training (no NaN)'
    ],
    
    'implementation': [
        'Feature embeddings working',
        'Attention mechanism functional',
        'Proper input/output scaling',
        'Gradient clipping effective'
    ]
}
```

---

## 📝 Conclusion

The **Enhanced LSTM Architecture** is specifically designed to leverage your rich feature set:

### Key Advantages
1. **18 features** vs 3 in basic version (6× more information)
2. **Feature embeddings** learn optimal representations
3. **Attention mechanism** focuses on important patterns
4. **Bidirectional LSTM** captures past + future context
5. **Proper scaling** eliminates the mismatch problem
6. **Multiple regularization** techniques prevent overfitting

### Expected Outcome
With proper implementation, the enhanced LSTM should significantly outperform both the basic LSTM and potentially the HAR-R baseline, achieving:
- **QLIKE Loss: < 0.5** (vs 16,033 in basic LSTM)
- **Directional Accuracy: > 55%** (vs 49.4% in HAR-R)
- **Stable training** without NaN losses

The enhanced architecture transforms LSTM from a "failed experiment" to a **promising approach** for volatility forecasting with rich feature sets.

---

**Generated:** 2026-06-17  
**Architecture:** Enhanced LSTM with 18 Features + Attention  
**Diagram:** enhanced_lstm_architecture.png  
**Implementation:** enhanced_lstm_architecture.py