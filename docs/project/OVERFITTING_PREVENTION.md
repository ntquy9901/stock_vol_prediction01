# Overfitting Prevention - Stock Volatility Prediction

**Project:** Stock Volatility Prediction - VN30  
**Purpose:** Mandatory anti-overfitting rules cho ALL models  
**Version:** 1.0  
**Date:** 2026-06-21

---

## Overview

**Overfitting** là vấn đề CRITICAL trong volatility forecasting. Mô hình overfit sẽ có performance tốt trên tập train nhưng fail thảm hại trên thực tế (test data/unseen data).

**Document này là MANDATORY cho tất cả models trong project:**
- LSTM Baseline
- LSTM-HAR
- Enhanced LSTM-HAR
- TimesFM LoRA
- LSTM-GAT Hybrid (future)

---

## 1. Data-Centric Techniques (Priority 1)

**Data-centric methods are MOST effective** - luôn ưu tiên trước khi can thiệp model.

### 1.1 Data Augmentation cho Time Series ⭐

**Mandatory cho tất cả LSTM variants:**

```python
# ✅ CORRECT - Time series augmentation
def augment_time_series(series, augment_factor=2):
    """Jittering + Scaling cho volatility series"""
    augmented = []
    for _ in range(augment_factor):
        # Add Gaussian noise (jittering)
        noise = np.random.normal(0, 0.01, len(series))
        jittered = series + noise * series.std()
        
        # Random scaling
        scale = np.random.uniform(0.95, 1.05)
        scaled = series * scale
        
        augmented.extend([jittered, scaled])
    
    return np.array(augmented)
```

**Applied to:**
- `src/lstm_baseline/` 
- `src/lstm_har_baseline/`
- `src/lstm_har_enhanced/`

### 1.2 Outlier Removal (Mandatory) ⭐

**Critical:** Volatility data chứa nhiều outliers (market crashes, flash crashes).

```python
# ✅ CORRECT - Outlier removal
def remove_volatility_outliers(volatility_series, n_std=3):
    """Remove outliers beyond n_std"""
    mean = volatility_series.mean()
    std = volatility_series.std()
    
    # Cap outliers at n_std
    capped = volatility_series.copy()
    capped[capped > mean + n_std * std] = mean + n_std * std
    capped[capped < mean - n_std * std] = mean - n_std * std
    
    return capped
```

**Applied to:** ALL preprocessing pipelines.

### 1.3 Label Smoothing (Optional but Recommended)

```python
# Smooth labels to prevent overconfident predictions
def smooth_labels(y, smoothing=0.1):
    """Apply label smoothing"""
    return y * (1 - smoothing) + 0.5 * smoothing
```

---

## 2. Model-Centric Techniques (Priority 2)

### 2.1 Early Stopping (MANDATORY) ⭐⭐⭐

**CRITICAL:** ALL models MUST use early stopping.

```python
# ✅ CORRECT - Early stopping implementation
from pytorch_lightning.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,           # Standard cho project
    mode='min',
    verbose=True,
    restore_best_weights=True  # CRITICAL!
)
```

**Standard Hyperparameters:**
- `patience=15` cho ALL LSTM variants
- `patience=10` cho TimesFM (foundation model hội tụ nhanh hơn)

**Documentation:** See `CLAUDE.md` section "Standard Hyperparameters"

### 2.2 L2 Regularization (Weight Decay) (MANDATORY) ⭐⭐

**CRITICAL:** ALL models MUST use weight decay.

```python
# ✅ CORRECT - Weight decay
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5      # MANDATORY
)
```

**Standard Values:**
- `weight_decay=1e-5` cho LSTM variants
- `weight_decay=1e-4` cho TimesFM (foundation model cần regularization mạnh hơn)

### 2.3 Dropout (MANDATORY cho LSTM) ⭐⭐⭐

**CRITICAL:** ALL LSTM models MUST use dropout.

```python
# ✅ CORRECT - Dropout in LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=2,
            dropout=dropout      # MANDATORY: 0.2
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

**Standard Values:**
- `dropout=0.2` cho LSTM layers
- `dropout=0.3` cho fully connected layers

### 2.4 Layer Normalization (Recommended) ⭐

```python
# ✅ Add layer normalization
class LSTMWithLayerNorm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)  # Add this
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)  # Normalize
        return out[:, -1, :]
```

---

## 3. Architecture-Specific Techniques

### 3.1 Cho LSTM Models ⭐⭐⭐

**CRITICAL:** Standard Dropout phá vỡ temporal dependencies trong LSTM.

#### Use Recurrent Dropout (NOT Standard Dropout)

```python
# ❌ WRONG - Standard dropout breaks temporal memory
self.lstm = nn.LSTM(input_size, hidden_size, dropout=0.2)

# ✅ CORRECT - Recurrent dropout (if using custom LSTM implementation)
# PyTorch LSTM's dropout parameter IS recurrent dropout internally
```

**Note:** PyTorch's `nn.LSTM(dropout=0.2)` applies dropout BETWEEN LSTM layers (not within), which is correct cho multi-layer LSTM.

#### Spatial Dropout cho Sequence Data

```python
# ✅ CORRECT - Drop entire channels (features) instead of random elements
class SpatialDropout(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
    def forward(self, x):
        if not self.training:
            return x
        # x: (batch, seq_len, features)
        mask = torch.bernoulli(torch.ones(x.size(0), 1, x.size(2)) * (1 - self.dropout)).to(x.device)
        return x * mask / (1 - self.dropout)
```

### 3.2 Cho Graph Neural Networks (LSTM-GAT Hybrid) ⭐⭐⭐

#### DropEdge (MANDATORY cho GNN)

```python
# ✅ CORRECT - DropEdge for GAT
class DropEdgeGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, edge_drop=0.3):
        super().__init__()
        self.gat = GATConv(in_features, out_features, dropout=dropout)
        self.edge_drop = edge_drop  # Drop edges
        
    def forward(self, x, edge_index):
        # Randomly drop edges
        if self.training:
            num_edges = edge_index.size(1)
            mask = torch.bernoulli(torch.ones(num_edges) * (1 - self.edge_drop)).bool()
            edge_index = edge_index[:, mask]
            
        return self.gat(x, edge_index)
```

**Standard Values:**
- `edge_drop=0.3` cho dynamic graph construction
- `node_drop=0.2` cho node-level dropout

### 3.3 Cho TimesFM (Foundation Model) ⭐⭐

**Note:** Foundation models có built-in regularization,但仍 cần:

#### Gradient Clipping (MANDATORY)

```python
# ✅ CORRECT - Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### LoRA-Specific Regularization

```python
# ✅ CORRECT - LoRA with regularization
lora_config = {
    'r': 8,              # Rank (lower = more regularization)
    'lora_alpha': 16,    # Scaling factor
    'lora_dropout': 0.1  # Dropout cho LoRA layers
}
```

---

## 4. Training Techniques

### 4.1 Learning Rate Scheduling (MANDATORY) ⭐⭐

**CRITICAL:** ALL models MUST use learning rate scheduler.

```python
# ✅ CORRECT - ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,      # Reduce LR by half
    patience=5,     # Wait 5 epochs before reducing
    verbose=True
)
```

**Alternative (cosine annealing):**
```python
# ✅ Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,         # Restart every 10 epochs
    T_mult=2         # Double period after each restart
)
```

### 4.2 Gradient Clipping (MANDATORY) ⭐⭐

**CRITICAL:** ALL models MUST use gradient clipping.

```python
# ✅ CORRECT - Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Why:** Prevent exploding gradients trong LSTM training.

### 4.3 Batch Size Tuning

**Standard Values:**
- `batch_size=32` cho LSTM variants
- `batch_size=64` cho TimesFM (foundation model)

**Note:** Larger batches → more stable gradients but less regularization effect.

---

## 5. Cross-Validation Strategy (CRITICAL)

### 5.1 Time Series Split (MANDATORY) ⭐⭐⭐

**CRITICAL:** NEVER use standard K-Fold cho time series!

```python
# ❌ WRONG - Standard K-Fold causes data leakage
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)

# ✅ CORRECT - TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

**Why:** Standard K-Fold shuffles data → future information leaks into training.

### 5.2 Walk-Forward Validation (Recommended cho Production)

```python
# ✅ CORRECT - Walk-forward validation
def walk_forward_validation(data, window_size=252, test_size=22):
    """Expanding window validation cho time series"""
    predictions = []
    for i in range(window_size, len(data) - test_size, test_size):
        train = data[:i]
        test = data[i:i+test_size]
        
        model.fit(train)
        pred = model.predict(test)
        predictions.append(pred)
    
    return predictions
```

---

## 6. Model Complexity Control

### 6.1 Architecture Selection Guidelines

**Dataset Size vs Model Complexity:**

| Dataset Size | Recommended Architecture |
|--------------|--------------------------|
| < 1000 samples | Linear regression, simple LSTM (1 layer) |
| 1000-5000 | LSTM-HAR (2 layers) |
| 5000-10000 | Enhanced LSTM-HAR + dropout |
| > 10000 | LSTM-GAT Hybrid, TimesFM LoRA |

**Current Project:** ~5000 samples → Enhanced LSTM-HAR is appropriate.

### 6.2 Hidden Size Tuning

```python
# ✅ CORRECT - Hidden size based on data complexity
# Small dataset (< 1000 samples)
hidden_size = 32

# Medium dataset (1000-5000 samples)
hidden_size = 64  # Current project

# Large dataset (> 5000 samples)
hidden_size = 128
```

**Rule of thumb:** `hidden_size = min(64, num_features * 4)`

---

## 7. Monitoring & Detection

### 7.1 Learning Curves (MANDATORY) ⭐⭐⭐

**CRITICAL:** Plot learning curves EVERY 10 epochs.

```python
# ✅ CORRECT - Plot learning curves
if (epoch + 1) % 10 == 0:
    plot_learning_curves(
        train_losses, 
        val_losses, 
        save_path=f'results/learning_curves_epoch_{epoch+1}.png'
    )
```

**Signs of Overfitting:**
- ✅ Train loss ↓, Val loss ↑ (classic overfitting)
- ✅ Train loss << Val loss (large gap > 0.05)
- ✅ Val loss increases after plateau

**Signs of Underfitting:**
- ✅ Train loss và Val loss cả hai đều high
- ✅ Train loss và Val loss converge nhưng không improve

### 7.2 Metrics Comparison (MANDATORY) ⭐⭐⭐

**CRITICAL:** Compare validation vs test metrics.

```python
# ✅ CORRECT - Check val-test gap
val_rmse = 0.18
test_rmse = 0.25
gap = test_rmse - val_rmse

if gap > 0.05:  # 5% gap threshold
    print(f"WARNING: Large val-test gap ({gap:.4f}) - possible overfitting!")
```

**Thresholds:**
- Gap < 0.02: Good generalization
- Gap 0.02-0.05: Acceptable
- Gap > 0.05: Overfitting warning

---

## 8. Mandatory Checklist (ALL Models)

### Before Training

- [ ] Data augmentation applied (if dataset size < 5000)
- [ ] Outliers removed (n_std=3)
- [ ] Temporal split verified (NOT random split)
- [ ] Early stopping configured (patience=15)
- [ ] Weight decay set (1e-5 cho LSTM, 1e-4 cho TimesFM)
- [ ] Dropout configured (0.2 cho LSTM layers)
- [ ] Learning rate scheduler configured
- [ ] Gradient clipping enabled (max_norm=1.0)

### During Training

- [ ] Learning curves plotted every 10 epochs
- [ ] Val loss monitored for overfitting signs
- [ ] Checkpoints saved at best val loss
- [ ] Gradients clipped every batch
- [ ] LR reduced on plateau

### After Training

- [ ] Val-test metrics gap computed
- [ ] Gap < 0.05 (generalization check)
- [ ] Learning curves reviewed for overfitting patterns
- [ ] Model evaluated on ALL 6 mandatory metrics
- [ ] Results compared to baseline

---

## 9. Quick Reference Implementation

### Complete Training Loop with All Anti-Overfitting Techniques

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 1. Model with dropout
class VolatilityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=2,
            batch_first=True,
            dropout=dropout  # Recurrent dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# 2. Training setup
model = VolatilityLSTM(input_size=3, hidden_size=64, dropout=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
criterion = nn.MSELoss()

# 3. Early stopping
best_val_loss = float('inf')
patience_counter = 0
patience = 15

# 4. Training loop
for epoch in range(70):
    # Train
    model.train()
    train_loss = 0
    for batch in train_loader:
        x, y = batch
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Plot learning curves every 10 epochs
    if (epoch + 1) % 10 == 0:
        plot_learning_curves(train_losses, val_losses)
```

---

## 10. Architecture-Specific Examples

### 10.1 LSTM-GAT Hybrid Anti-Overfitting

```python
class LSTM_GAT_Hybrid(nn.Module):
    def __init__(self, num_stocks=30, hidden_size=64, num_heads=4):
        super().__init__()
        # Temporal encoder with dropout
        self.lstm = nn.LSTM(
            input_size=22,  # HAR + technical indicators
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Spatial encoder with edge dropout
        self.gat1 = GATConv(hidden_size, hidden_size, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_size * num_heads, hidden_size, heads=1, dropout=0.2)
        
        # Edge dropout rate
        self.edge_drop = 0.3
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, edge_index):
        # Temporal encoding
        temporal_out, _ = self.lstm(x)
        temporal_features = temporal_out[:, -1, :]  # Last timestep
        
        # Edge dropout for graph
        if self.training:
            num_edges = edge_index.size(1)
            mask = torch.bernoulli(torch.ones(num_edges) * (1 - self.edge_drop)).bool()
            edge_index = edge_index[:, mask]
        
        # Spatial encoding
        spatial_features = self.gat1(temporal_features, edge_index)
        spatial_features = F.elu(spatial_features)
        spatial_features = self.gat2(spatial_features, edge_index)
        
        # Fusion
        combined = torch.cat([temporal_features, spatial_features], dim=-1)
        out = self.fusion(combined)
        out = self.layer_norm(out)
        out = self.dropout(out)
        
        return out
```

### 10.2 TimesFM LoRA Anti-Overfitting

```python
# LoRA configuration with built-in regularization
lora_config = {
    'r': 8,              # Low rank = strong regularization
    'lora_alpha': 16,    # Scaling
    'lora_dropout': 0.1, # Dropout cho LoRA
    'target_modules': ['q_proj', 'v_proj'],  # Attention only
}

# Training with gradient clipping
for batch in train_loader:
    outputs = model(**batch)
    loss = outputs.loss
    
    loss.backward()
    
    # Gradient clipping (MANDATORY cho foundation models)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
```

---

## 11. Troubleshooting Guide

### Problem: Still Overfitting After Applying All Techniques

**Symptoms:**
- Train loss << Val loss (gap > 0.1)
- Val loss increases after plateau
- Test metrics significantly worse than val

**Solutions (in order):**

1. **Reduce model complexity**
   - Decrease hidden_size: 128 → 64 → 32
   - Reduce num_layers: 3 → 2
   - Remove unnecessary features

2. **Increase regularization strength**
   - Increase dropout: 0.2 → 0.3 → 0.4
   - Increase weight_decay: 1e-5 → 1e-4 → 1e-3
   - Add L1 regularization (feature selection)

3. **Get more data**
   - Data augmentation (factor 2-3x)
   - Collect more historical data
   - Add more stocks

4. **Use simpler architecture**
   - LSTM-GAT → LSTM-HAR → Linear HAR

### Problem: Underfitting

**Symptoms:**
- Train loss và Val loss cả hai đều high
- Model fails to learn training data

**Solutions:**

1. **Increase model capacity**
   - Increase hidden_size
   - Add more layers
   - Reduce dropout

2. **Train longer**
   - Increase epochs
   - Reduce early stopping patience
   - Use learning rate warmup

3. **Feature engineering**
   - Add more features
   - Use better features
   - Feature selection

---

## 12. References & Resources

**Internal Project Docs:**
- `CLAUDE.md` - Project rules
- `docs/project/TEMPORAL_SPLIT_EVALUATION.md` - Evaluation methodology
- `docs/project/LSTM_GAT_ARCHITECTURE.md` - LSTM-GAT architecture

**External Resources:**
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [DropEdge: Node Dropout for Graph Neural Networks](https://arxiv.org/abs/1907.07674)
- [TimesFM Foundation Model Paper](https://arxiv.org/abs/2310.10666)

---

## 13. Version History

**v1.0 (2026-06-21):**
- Initial version
- Mandatory techniques defined
- Architecture-specific guidelines added
- Complete examples provided

---

**Last Updated:** 2026-06-21  
**Maintainer:** Project Team  
**Status:** ACTIVE - Mandatory cho all models
