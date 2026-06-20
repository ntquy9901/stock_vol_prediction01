# LSTM-HAR Baseline Guide

**Date:** 2026-06-18  
**Baseline:** LSTM-HAR (LSTM with Heterogeneous Autoregressive Features)  
**Status:** Ready for Training

---

## Overview

LSTM-HAR baseline combines LSTM architecture with HAR (Heterogeneous Autoregressive) features for improved volatility forecasting. Instead of using raw Parkinson volatility as input, this baseline uses engineered HAR features that capture multi-scale temporal patterns.

---

## Architecture

### Input Features (3 HAR Features)
```
1. har_daily_vol:   1-day rolling average  (daily pattern)
2. har_weekly_vol:  5-day rolling average  (weekly pattern)
3. har_monthly_vol: 22-day rolling average (monthly pattern)
```

### Model Architecture
```
Input: (batch, seq_length=22, features=3)
    ↓
LSTM Layer 1: 64 hidden units + Dropout(0.2)
    ↓
LSTM Layer 2: 64 hidden units + Dropout(0.2)
    ↓
Fully Connected: 64 → 1
    ↓
ReLU Activation
    ↓
Output: (batch, 1) - Volatility prediction
```

### Model Parameters
- **Total Parameters:** ~33,000 (vs ~17,000 for Simple LSTM)
- **Input Dimension:** 3 (HAR features)
- **Hidden Size:** 64
- **Layers:** 2
- **Dropout:** 0.2

---

## Key Differences from Simple LSTM

| Aspect | Simple LSTM | LSTM-HAR |
|--------|-------------|----------|
| **Input** | Raw Parkinson volatility | HAR features (daily, weekly, monthly) |
| **Features** | 1 feature | 3 features |
| **Architecture** | 1 layer, 128 hidden units | 2 layers, 64 hidden units |
| **Parameters** | ~17,000 | ~33,000 |
| **Complexity** | Lower | Higher (with dropout) |
| **Pattern Capture** | Sequential only | Multi-scale temporal |

---

## Advantages of HAR Features

### 1. Multi-Scale Temporal Patterns
```python
# Captures patterns at different time horizons
har_daily_vol:   Short-term patterns (1 day)
har_weekly_vol:  Medium-term patterns (5 days)
har_monthly_vol: Long-term patterns (22 days)
```

### 2. Reduced Noise
- Rolling averages smooth out daily noise
- More stable input signals
- Better generalization

### 3. Financial Intuition
- HAR features are well-established in volatility literature
- Capture different investment horizons
- Align with domain knowledge

### 4. Improved Learning
- Multiple features provide richer information
- Model can learn interactions between time scales
- Better representation of volatility dynamics

---

## File Structure

```
src/lstm_har_baseline/
├── __init__.py              # Package initialization
├── features.py              # HAR feature engineering
├── dataset.py               # HAR dataset class
├── model.py                 # LSTM-HAR model architecture
└── train.py                 # Training logic

train_lstm_har.py            # Root-level training script
```

---

## Usage

### Quick Start

```bash
# Train LSTM-HAR model
python train_lstm_har.py
```

### Advanced Usage

```python
from src.lstm_har_baseline import (
    HARVolatilityLSTM,
    HARVolatilityDataset,
    create_har_features
)

# Create HAR features
har_features = create_har_features(parkinson_volatility)

# Create dataset
dataset = HARVolatilityDataset('data/processed', seq_length=22)

# Create model
model = HARVolatilityLSTM(hidden_size=64, num_layers=2, dropout=0.2)

# Train (see train.py for full training loop)
```

---

## Training Configuration

### Hyperparameters
```python
seq_length = 22              # Input window size
forecast_horizon = 5         # 5-day ahead prediction
batch_size = 64
learning_rate = 0.001
num_epochs = 50
patience = 10                # Early stopping
hidden_size = 64
num_layers = 2
dropout = 0.2
weight_decay = 1e-5
```

### Data Split
- **Train:** 80% of pooled data
- **Test:** 20% of pooled data
- **Split Method:** Random split with seed=42

### Optimization
- **Optimizer:** Adam
- **Loss Function:** MSE
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Gradient Clipping:** max_norm=1.0
- **Mixed Precision:** Enabled for GPU training

---

## Expected Performance

### Baseline Comparison

| Model | Expected RMSE | Expected Dir Acc | Complexity |
|-------|---------------|------------------|------------|
| HAR-R | 0.018 - 0.022 | 52% - 56% | Low |
| Simple LSTM | 0.016 - 0.020 | 54% - 58% | Medium |
| **LSTM-HAR** | **0.015 - 0.019** | **56% - 60%** | Medium-High |

### Success Criteria
- ✅ **RMSE:** < 0.20 (target: < 0.019)
- ✅ **Directional Accuracy:** > 55% (target: > 58%)
- ✅ **QLIKE Loss:** Lower than baselines
- ✅ **Learning Curves:** No severe overfitting

---

## HAR Features Explained

### Formula
```python
har_daily_vol = volatility.rolling(1).mean()   # Daily average
har_weekly_vol = volatility.rolling(5).mean()  # Weekly average
har_monthly_vol = volatility.rolling(22).mean() # Monthly average
```

### Interpretation
- **har_daily_vol:** Current volatility level
- **har_weekly_vol:** Short-term trend (5 trading days)
- **har_monthly_vol:** Long-term trend (~22 trading days/month)

### Why 22 Days?
- ~22 trading days per month
- Academic standard for monthly window
- Captures monthly seasonality
- Balances signal-to-noise ratio

---

## Results Organization

Training results will be saved to:
```
results/lstm_har_YYYY-MM-DD_HHMMSS/
├── best_lstm_har_model.pth              # Best model checkpoint
├── lstm_har_results.json                 # Test metrics
├── learning_curves_epoch_*.png           # Learning curves (every 10 epochs)
└── learning_curves_final.png             # Final learning curves
```

---

## Monitoring Training

### Key Metrics to Watch
1. **Training/Validation Loss Gap**
   - Small gap (< 10%): Good generalization
   - Large gap (> 20%): Potential overfitting

2. **Convergence Speed**
   - Should converge within 20-30 epochs
   - Slow convergence: Check learning rate

3. **Learning Curves**
   - Both train and val should decrease
   - Val loss increasing: Overfitting (stop early)

### Example Output
```
Epoch 10/50 (12.3s) - Train Loss: 0.003245, Val Loss: 0.004123
Epoch 20/50 (11.8s) - Train Loss: 0.002876, Val Loss: 0.003912
Epoch 30/50 (12.1s) - Train Loss: 0.002654, Val Loss: 0.003845
💾 Learning curves saved: results/lstm_har_.../learning_curves_epoch_30.png
```

---

## Troubleshooting

### Common Issues

#### 1. High Training Loss (> 0.01)
**Cause:** Data scaling issues or wrong feature extraction
**Solution:** Check HAR feature calculation and scaling

#### 2. Overfitting (val loss increasing)
**Cause:** Model too complex or insufficient data
**Solution:** 
- Increase dropout (0.2 → 0.3)
- Reduce hidden size (64 → 32)
- Add more regularization

#### 3. Slow Convergence
**Cause:** Learning rate too low
**Solution:** Increase learning rate (0.001 → 0.005)

#### 4. NaN Values in Training
**Cause:** Invalid HAR features or division by zero
**Solution:** Check data validation and feature extraction

---

## Next Steps

### Immediate Tasks
1. ✅ **Train LSTM-HAR model**
   ```bash
   python train_lstm_har.py
   ```

2. ⏳ **Compare with baselines**
   - Compare RMSE, MAE, R², directional accuracy
   - Analyze learning curves
   - Check overfitting

3. ⏳ **Hyperparameter tuning**
   - Test different hidden sizes (32, 64, 128)
   - Test different dropout rates (0.1, 0.2, 0.3)
   - Test different learning rates

### Future Enhancements
- **Attention mechanism:** Focus on important time steps
- **Residual connections:** Improve gradient flow
- **Multi-head output:** Predict multiple horizons (1, 5, 10, 22 days)
- **Ensemble methods:** Combine with HAR-R baseline

---

## References

### Academic Papers
1. **HAR Model:** Corsi (2009) - "A Simple Long Memory Model of Realized Volatility"
2. **LSTM for Volatility:** Various recent applications
3. **Parkinson Estimator:** Parkinson (1980)

### Project Documentation
- `CLAUDE.md` - Project rules and guidelines
- `docs/lstm/ENHANCED_LSTM_GUIDE.md` - LSTM enhancements
- `docs/project/REFACTOR_SUMMARY.md` - Project organization

---

## Contact

**Project:** Stock Volatility Prediction - VN30  
**Date:** 2026-06-18  
**Status:** Development Phase - LSTM-HAR Baseline Ready

---

*Last Updated: 2026-06-18*  
*Version: 1.0*  
*Author: Stock Volatility Prediction Team*