# LSTM Improvement Strategies to Beat HAR Baseline

**Date:** 2026-06-18  
**Objective:** Improve LSTM architecture to outperform HAR baseline  
**Current Status:** Analyzing improvement opportunities

---

## Current Performance Comparison

### Baseline Results (Expected)

| Model | RMSE | Directional Accuracy | Complexity |
|-------|------|----------------------|------------|
| **HAR-R** | 0.018 - 0.022 | 52% - 56% | Low |
| **Simple LSTM** | 0.016 - 0.020 | 54% - 58% | Medium |
| **LSTM-HAR** | 0.015 - 0.019 | 56% - 60% | Medium-High |

**Goal:** Achieve **RMSE < 0.017** and **Directional Accuracy > 58%**

---

## 🎯 Strategy 1: Attention Mechanism ⭐ HIGH IMPACT

### Concept
Add attention layer to focus on important time steps in volatility patterns.

### Architecture
```
Input (batch, seq_len, features)
    ↓
LSTM Layers (hidden states)
    ↓
Attention Layer (weighted combination)
    ↓
Fully Connected
    ↓
Output (batch, 1)
```

### Implementation
```python
class AttentiveLSTM(nn.Module):
    """LSTM with attention mechanism for volatility prediction."""

    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Calculate attention weights
        attention_weights = torch.softmax(
            self.attention(lstm_out), dim=1
        )  # (batch, seq_len, 1)

        # Apply attention weights
        context_vector = torch.sum(
            attention_weights * lstm_out, dim=1
        )  # (batch, hidden_size)

        # Output
        output = self.relu(self.fc(context_vector))
        return output
```

### Benefits
- ✅ Focuses on important time periods (e.g., market crashes)
- ✅ Learns which historical patterns matter most
- ✅ Interpretable (can visualize attention weights)
- ✅ Expected improvement: **5-10% better RMSE**

---

## 🎯 Strategy 2: Bidirectional LSTM ⭐ HIGH IMPACT

### Concept
Use bidirectional LSTM to capture both past and future context within window.

### Architecture
```
Input (batch, seq_len, features)
    ↓
Bidirectional LSTM
    ├→ Forward LSTM (past → future)
    └→ Backward LSTM (future → past)
    ↓
Concatenate Forward + Backward
    ↓
Fully Connected
    ↓
Output (batch, 1)
```

### Implementation
```python
class BiLSTM(nn.Module):
    """Bidirectional LSTM for volatility prediction."""

    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2,
            bidirectional=True
        )

        # Output layer (hidden_size * 2 for bidirectional)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)

        # Extract last timestep
        last_output = lstm_out[:, -1, :]

        # Output
        output = self.relu(self.fc(last_output))
        return output
```

### Benefits
- ✅ Captures patterns from both directions
- ✅ Better understanding of context
- ✅ Expected improvement: **3-7% better RMSE**

---

## 🎯 Strategy 3: Multi-Scale Feature Pyramid ⭐ HIGH IMPACT

### Concept
Process features at multiple time scales and combine them.

### Architecture
```
Input (batch, seq_len, features)
    ↓
┌─────────────────────────────────────┐
│ Multi-Scale Feature Extraction      │
├─────────────────────────────────────┤
│ Scale 1: Raw daily (seq_len=22)     │
│ Scale 2: Weekly avg (seq_len=5)      │
│ Scale 3: Monthly avg (seq_len=3)     │
└─────────────────────────────────────┘
    ↓
Concatenate Multi-Scale Features
    ↓
LSTM Processing
    ↓
Output (batch, 1)
```

### Implementation
```python
class MultiScaleLSTM(nn.Module):
    """LSTM with multi-scale feature extraction."""

    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()

        # Separate LSTM for each scale
        self.lstm_daily = nn.LSTM(input_size, hidden_size,
                                  batch_first=True)
        self.lstm_weekly = nn.LSTM(input_size, hidden_size//2,
                                   batch_first=True)
        self.lstm_monthly = nn.LSTM(input_size, hidden_size//2,
                                    batch_first=True)

        # Combine all scales
        total_hidden = hidden_size + hidden_size//2 + hidden_size//2
        self.fc = nn.Linear(total_hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Original sequence
        daily_out, _ = self.lstm_daily(x)
        daily_last = daily_out[:, -1, :]

        # Downsample for weekly (every 5 days)
        weekly_x = x[:, ::5, :]
        weekly_out, _ = self.lstm_weekly(weekly_x)
        weekly_last = weekly_out[:, -1, :]

        # Downsample for monthly (every 7 days approx)
        monthly_x = x[:, ::7, :]
        monthly_out, _ = self.lstm_monthly(monthly_x)
        monthly_last = monthly_out[:, -1, :]

        # Concatenate
        combined = torch.cat([daily_last, weekly_last, monthly_last], dim=1)

        # Output
        output = self.relu(self.fc(combined))
        return output
```

### Benefits
- ✅ Captures multi-scale patterns explicitly
- ✅ Similar to HAR but with LSTM learning
- ✅ Expected improvement: **5-12% better RMSE**

---

## 🎯 Strategy 4: Enhanced Feature Engineering ⭐ MEDIUM IMPACT

### Concept
Add more sophisticated features beyond HAR features.

### New Features

#### 1. Momentum Features
```python
# Price momentum
momentum_5d = volatility[t] / volatility[t-5] - 1
momentum_10d = volatility[t] / volatility[t-10] - 1
momentum_22d = volatility[t] / volatility[t-22] - 1
```

#### 2. Volatility of Volatility (VoV)
```python
# How much volatility itself is changing
vov_5d = volatility.rolling(5).std()
vov_10d = volatility.rolling(10).std()
```

#### 3. Lag Features
```python
# Historical volatility at different lags
lag_1 = volatility[t-1]
lag_5 = volatility[t-5]
lag_22 = volatility[t-22]
```

#### 4. Range-Based Features
```python
# Parkinson-based range features
range_daily = high / low
range_weekly_avg = (high / low).rolling(5).mean()
```

### Implementation
```python
def create_enhanced_features(parkinson_volatility, high, low):
    """Create enhanced features for LSTM."""
    features = pd.DataFrame()

    # HAR features (existing)
    features['har_daily'] = parkinson_volatility.rolling(1).mean()
    features['har_weekly'] = parkinson_volatility.rolling(5).mean()
    features['har_monthly'] = parkinson_volatility.rolling(22).mean()

    # Momentum features
    features['momentum_5d'] = parkinson_volatility / parkinson_volatility.shift(5) - 1
    features['momentum_10d'] = parkinson_volatility / parkinson_volatility.shift(10) - 1

    # Volatility of Volatility
    features['vov_5d'] = parkinson_volatility.rolling(5).std()
    features['vov_10d'] = parkinson_volatility.rolling(10).std()

    # Lag features
    features['lag_1'] = parkinson_volatility.shift(1)
    features['lag_5'] = parkinson_volatility.shift(5)

    # Range features
    features['range_daily'] = high / low
    features['range_weekly'] = (high / low).rolling(5).mean()

    return features.dropna()
```

### Benefits
- ✅ Richer feature set
- ✅ Captures more market dynamics
- ✅ Expected improvement: **3-8% better RMSE**

---

## 🎯 Strategy 5: Residual Connections ⭐ MEDIUM IMPACT

### Concept
Add skip connections to improve gradient flow and enable deeper networks.

### Architecture
```
Input (batch, seq_len, features)
    ↓
LSTM Layer 1 + Residual
    ↓
LSTM Layer 2 + Residual
    ↓
LSTM Layer 3 + Residual
    ↓
Fully Connected
    ↓
Output (batch, 1)
```

### Implementation
```python
class ResidualLSTM(nn.Module):
    """LSTM with residual connections."""

    def __init__(self, input_size=3, hidden_size=64, num_layers=3):
        super().__init__()

        # LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_size,
                   hidden_size, batch_first=True)
            for i in range(num_layers)
        ])

        # Projection layers for residual connections
        self.projections = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Extract last dimension for residual
        residual_input = x[:, -1, :]  # (batch, input_size)

        for i, lstm in enumerate(self.lstm_layers):
            # LSTM forward pass
            lstm_out, _ = lstm(x)

            # Extract last timestep
            last_out = lstm_out[:, -1, :]  # (batch, hidden_size)

            # Project residual input
            projected_residual = self.projections[i](residual_input)

            # Add residual connection
            last_out = last_out + projected_residual

            # Update for next layer
            residual_input = last_out.unsqueeze(1)  # (batch, 1, hidden_size)
            x = lstm_out

        # Output
        output = self.relu(self.fc(last_out))
        return output
```

### Benefits
- ✅ Better gradient flow
- ✅ Enables deeper networks
- ✅ Reduces vanishing gradient problem
- ✅ Expected improvement: **2-5% better RMSE**

---

## 🎯 Strategy 6: Ensemble Methods ⭐ HIGH IMPACT

### Concept
Combine multiple LSTM models with different architectures.

### Architecture
```
Model 1: LSTM-HAR (current)
Model 2: Bidirectional LSTM
Model 3: Attentive LSTM
    ↓
Weighted Average / Stacking
    ↓
Final Prediction
```

### Implementation
```python
class LSTMEnsemble:
    """Ensemble of multiple LSTM models."""

    def __init__(self, models, weights=None):
        """
        Args:
            models: List of trained LSTM models
            weights: Optional weights for averaging (default: equal weights)
        """
        self.models = models
        self.weights = weights if weights else [1.0/len(models)] * len(models)

    def predict(self, x):
        """Make ensemble prediction."""
        predictions = []

        for model in self.models:
            pred = model(x)
            predictions.append(pred)

        # Weighted average
        ensemble_pred = sum(
            w * p for w, p in zip(self.weights, predictions)
        )

        return ensemble_pred
```

### Training Strategy
```python
# Train individual models
model1 = HARVolatilityLSTM()
model2 = BiLSTM()
model3 = AttentiveLSTM()

# Train each model
train_model(model1, train_loader)
train_model(model2, train_loader)
train_model(model3, train_loader)

# Create ensemble
ensemble = LSTMEnsemble([model1, model2, model3])
```

### Benefits
- ✅ Reduces overfitting
- ✅ Combines diverse model strengths
- ✅ More robust predictions
- ✅ Expected improvement: **5-15% better RMSE**

---

## 🎯 Strategy 7: Hyperparameter Optimization ⭐ MEDIUM IMPACT

### Key Hyperparameters to Tune

#### 1. Architecture
- **hidden_size**: [32, 64, 128, 256]
- **num_layers**: [1, 2, 3, 4]
- **dropout**: [0.1, 0.2, 0.3, 0.4]

#### 2. Training
- **learning_rate**: [0.0001, 0.001, 0.01]
- **batch_size**: [32, 64, 128]
- **weight_decay**: [1e-6, 1e-5, 1e-4]

#### 3. Sequence
- **seq_length**: [10, 22, 30, 44]

### Optimization Methods

#### Grid Search
```python
param_grid = {
    'hidden_size': [32, 64, 128],
    'learning_rate': [0.001, 0.01],
    'dropout': [0.2, 0.3]
}

# Test all combinations
best_model = grid_search(param_grid)
```

#### Random Search
```python
# Sample random combinations
best_model = random_search(param_distributions, n_iter=50)
```

#### Bayesian Optimization
```python
import optuna

def objective(trial):
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    learning_rate = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.4)

    model = HARVolatilityLSTM(hidden_size=hidden_size, dropout=dropout)
    # Train and evaluate
    return validation_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

### Benefits
- ✅ Find optimal configuration
- ✅ Significant performance gains
- ✅ Expected improvement: **3-10% better RMSE**

---

## 🎯 Strategy 8: Learning Rate Scheduling ⭐ LOW-MEDIUM IMPACT

### Concept
Use adaptive learning rate schedules for better convergence.

### Methods

#### 1. Cosine Annealing
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)
```

#### 2. OneCycleLR
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, epochs=50, steps_per_epoch=len(train_loader)
)
```

#### 3. Warmup + Decay
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

### Benefits
- ✅ Faster convergence
- ✅ Better final performance
- ✅ Expected improvement: **1-3% better RMSE**

---

## 🎯 Strategy 9: Data Augmentation ⭐ LOW-MEDIUM IMPACT

### Concept
Generate additional training samples through data augmentation.

### Methods

#### 1. Jittering (Noise Injection)
```python
def jitter_data(X, noise_level=0.01):
    """Add small random noise to input data."""
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise
```

#### 2. Scaling
```python
def scale_data(X, scale_range=(0.95, 1.05)):
    """Randomly scale sequences."""
    scale = np.random.uniform(*scale_range)
    return X * scale
```

#### 3. Time Shifting
```python
def shift_time(X, max_shift=2):
    """Randomly shift sequences in time."""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(X, shift, axis=1)
```

### Implementation
```python
# During training
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        # Apply augmentation
        X_augmented = jitter_data(X_batch.numpy())
        X_augmented = torch.FloatTensor(X_augmented)

        # Train with augmented data
        predictions = model(X_augmented)
        loss = criterion(predictions, y_batch)
```

### Benefits
- ✅ More training data
- ✅ Reduces overfitting
- ✅ Expected improvement: **1-4% better RMSE**

---

## 🎯 Strategy 10: Regularization Techniques ⭐ LOW-MEDIUM IMPACT

### Concept
Apply various regularization techniques to prevent overfitting.

### Methods

#### 1. Dropout
```python
# Increase dropout rate
self.lstm = nn.LSTM(..., dropout=0.3)  # Increase from 0.2
```

#### 2. L1/L2 Regularization
```python
# Add L1 regularization
l1_penalty = sum(torch.norm(p, 1) for p in model.parameters())
loss = criterion(predictions, targets) + lambda1 * l1_penalty

# Add L2 regularization (weight decay)
optimizer = torch.optim.Adam(model.parameters(),
                           lr=0.001,
                           weight_decay=1e-4)
```

#### 3. Early Stopping
```python
# Already implemented - tune patience
patience = [5, 10, 15, 20]  # Test different values
```

#### 4. Batch Normalization
```python
class BatchNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]

        # Apply batch normalization
        last_out = self.bn(last_out)

        output = self.fc(last_out)
        return output
```

### Benefits
- ✅ Reduces overfitting
- ✅ Better generalization
- ✅ Expected improvement: **2-6% better RMSE**

---

## 📊 Prioritized Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
1. ✅ **Enhanced Features** (Strategy 4) - Add momentum, VoV, lag features
2. ✅ **Hyperparameter Tuning** (Strategy 7) - Optuna for 50 trials
3. ✅ **Learning Rate Scheduling** (Strategy 8) - OneCycleLR

**Expected Improvement:** 8-15% better RMSE

### Phase 2: Architecture Improvements (3-5 days)
1. ✅ **Bidirectional LSTM** (Strategy 2) - Easy to implement
2. ✅ **Attention Mechanism** (Strategy 1) - High impact
3. ✅ **Residual Connections** (Strategy 5) - Enable deeper networks

**Expected Improvement:** 10-20% better RMSE

### Phase 3: Advanced Techniques (5-7 days)
1. ✅ **Multi-Scale Features** (Strategy 3) - Explicit multi-scale learning
2. ✅ **Ensemble Methods** (Strategy 6) - Combine multiple models

**Expected Improvement:** 15-25% better RMSE

### Phase 4: Fine-tuning (2-3 days)
1. ✅ **Data Augmentation** (Strategy 9) - Reduce overfitting
2. ✅ **Regularization** (Strategy 10) - Optimize dropout, weight decay

**Expected Improvement:** 3-8% better RMSE

---

## 🎯 Recommended Starting Point

### Immediate Action: Enhanced LSTM-HAR

Combine **Strategies 1, 2, and 4** for maximum impact:

```python
class EnhancedLSTM(nn.Module):
    """
    Enhanced LSTM with:
    - Bidirectional processing
    - Attention mechanism
    - Enhanced features (momentum, VoV, lags)
    """

    def __init__(self, input_size=8, hidden_size=64):
        super().__init__()

        # Input: 8 features instead of 3
        # HAR (3) + Momentum (2) + VoV (2) + Lag (1)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.2,
                            bidirectional=True)

        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x)

        # Attention weights
        attention_weights = torch.softmax(
            self.attention(lstm_out), dim=1
        )

        # Weighted context
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Output
        output = self.relu(self.fc(context))
        return output
```

### Expected Performance
- **RMSE:** 0.013 - 0.016 (vs HAR-R's 0.018-0.022)
- **Directional Accuracy:** 60% - 65% (vs HAR-R's 52%-56%)
- **Improvement:** **20-30% better than HAR-R**

---

## 📚 Implementation Guide

### Step 1: Enhanced Features
```bash
# Create enhanced feature module
cd src/lstm_har_baseline/
# Add enhanced feature creation to features.py
```

### Step 2: Model Architecture
```bash
# Create enhanced model
# Add enhanced_lstm.py to src/lstm_har_baseline/
```

### Step 3: Training
```bash
# Train enhanced model
python -m src.lstm_har_baseline.train_enhanced
```

### Step 4: Evaluation
```bash
# Compare all models
python -m src.experiment.compare_models
```

---

## 🔬 Validation Strategy

### Cross-Validation
- Use time-series cross-validation
- Test on different time periods
- Validate robustness across stocks

### Metrics to Track
1. **RMSE** - Primary metric (lower is better)
2. **Directional Accuracy** - Secondary metric (higher is better)
3. **QLIKE Loss** - Academic standard (lower is better)
4. **R² Score** - Explained variance (higher is better)

### Success Criteria
- ✅ RMSE < 0.017 (25% better than HAR-R)
- ✅ Directional Accuracy > 58% (5% better than HAR-R)
- ✅ QLIKE Loss < HAR-R baseline
- ✅ Consistent across different time periods

---

## 📖 References

### Academic Papers
1. **Attention in LSTM:** "Attention Is All You Need" (Vaswani et al., 2017)
2. **Bidirectional LSTM:** "Bidirectional LSTM" (Graves & Schmidhuber, 2005)
3. **HAR Model:** "HAR-RV: A Heterogeneous Autoregressive Model" (Corsi, 2009)
4. **Volatility Forecasting:** "Volatility Forecasting" (various recent papers)

### Project Documentation
- `CLAUDE.md` - Project rules
- `docs/lstm/LSTM_HAR_BASELINE_GUIDE.md` - LSTM-HAR baseline
- `docs/lstm/ENHANCED_LSTM_GUIDE.md` - LSTM enhancements

---

## 🎓 Next Steps

1. ✅ **Implement Enhanced Features** (Strategy 4) - Immediate impact
2. ✅ **Try Bidirectional LSTM** (Strategy 2) - Easy win
3. ✅ **Add Attention Mechanism** (Strategy 1) - High impact
4. ✅ **Create Ensemble** (Strategy 6) - Best final performance
5. ✅ **Hyperparameter Tuning** (Strategy 7) - Optimize everything

---

**Document Date:** 2026-06-18  
**Status:** Ready for Implementation  
**Priority:** HIGH - Beat HAR Baseline

---

*Last Updated: 2026-06-18*  
*Version: 1.0*  
*Author: Stock Volatility Prediction Team*