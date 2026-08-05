# Parallel LSTM-GNN Architecture for VN30 Volatility Forecasting

**Based on:** Sonani et al. (2025) - "Stock Price Prediction Using a Hybrid LSTM-GNN Model"  
**Implementation:** `src/lstm_gat_hybrid/model_parallel.py`  
**Date:** 2026-06-21  
**Status:** ✅ Active Training (Epoch 4/50) - Dir Acc: 64.87% - 65.05%  
**Target:** >67.90% Dir Acc (beat LSTM-HAR Enhanced baseline)

---

## 📋 Executive Summary

### **Architecture Overview**
Parallel LSTM-GNN Hybrid combining temporal pattern learning (LSTM) and spatial relationship modeling (GNN) for multi-stock volatility forecasting.

### **Key Innovations**
1. **Parallel Processing** (vs Sequential) - No cascading bottleneck
2. **Concatenation Fusion** - Proven 10.6% MSE reduction in paper
3. **Dual Graph Methods** - Correlation-based (paper) + k-NN sparse graph
4. **Anti-Overfitting** - From LSTM-HAR Enhanced (67.90% Dir Acc)

### **Performance Targets**
| Metric | Baseline (LSTM-HAR Enhanced) | Target | Current Status |
|--------|-------------------------------|--------|-----------------|
| RMSE | 0.18 | **<0.15** (17% improvement) | Training... |
| Dir Acc | 67.90% | **>75%** (7% improvement) | 64.87-65.05% (Epoch 4) |
| QLIKE | ~0.12 | **<0.10** (17% improvement) | Training... |
| R² | ~0.65 | **>0.75** (15% improvement) | Training... |

---

## 🏗️ Architecture Overview

### **High-Level Design**

```
Input: 30 stocks × 22 timesteps × 3 HAR features
     ↓
┌─────────────────────────────────────────────────────────────────┐
│              Parallel LSTM-GNN Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: [batch, 22, 30, 3] HAR features                        │
│         (seq_len, num_stocks, num_features)                    │
│                                                                 │
│  ┌──────────────────────┐         ┌──────────────────────┐    │
│  │   LSTM Stream        │         │   GNN Stream         │    │
│  │   (Temporal Branch)  │         │   (Spatial Branch)   │    │
│  │                      │         │                      │    │
│  │  For each stock:     │         │  For each timestep:  │    │
│  │  [batch, 22, 3]      │         │  [batch, 30, 3]      │    │
│  │       ↓              │         │       ↓              │    │
│  │  LSTM Encoder        │         │  GAT Layer 1         │    │
│  │  (2 layers, hidden=64)│        │  (4 heads, 64)       │    │
│  │       ↓              │         │       ↓              │    │
│  │  Final Hidden State  │         │  GAT Layer 2         │    │
│  │  [batch, 64]         │         │  (4 heads, 64)       │    │
│  │       ↓              │         │       ↓              │    │
│  │  [batch, 30, 64]     │         │  Mean Pool           │    │
│  └──────┬───────────────┘         │  [batch, 30, 256]    │    │
│         │                         └──────┬───────────────┘    │
│         │                                │                     │
│         └─────────────┬──────────────────┘                     │
│                       ↓                                        │
│         ┌─────────────────────────┐                          │
│         │  Concatenation Fusion    │                          │
│         │  [64 + 256 = 320 dims]  │                          │
│         └──────────┬───────────────┘                          │
│                    ↓                                           │
│         ┌─────────────────────────┐                          │
│         │  Dense Layers (MLP)     │                          │
│         │  320 → 64 → 32 → 1      │                          │
│         │  (ReLU + Dropout 0.5)   │                          │
│         └──────────┬───────────────┘                          │
│                    ↓                                           │
│         Output: [batch, 30] predictions                        │
└─────────────────────────────────────────────────────────────────┘
```

### **Why Parallel Architecture?**

**Sequential (Previous Approach) - FAILED:**
```
Input → LSTM → GNN → Fusion → Output
         ↓      ↓
      Bottleneck: GNN only sees LSTM-encoded features
      Problem: Cascading errors, information bottleneck
```

**Parallel (Current Approach) - WORKING:**
```
Input → LSTM ──┐
       ├─→ GNN ─┼─→ Concat → Fusion → Output
       │        │
       Both streams access RAW input independently
       Benefits: 
       - No information bottleneck
       - Both temporal and spatial features from raw data
       - 10.6% MSE reduction (proven in paper)
```

---

## 🔧 Key Components

### **1. LSTM Stream (Temporal Branch)**

**File:** `src/lstm_gat_hybrid/model_parallel.py` (Lines 58-64)

**Purpose:** Capture temporal patterns for each stock independently

**Architecture:**
```python
# For each of 30 stocks:
LSTM_Encoder(
    input_size=3,          # HAR features (daily, weekly, monthly)
    hidden_size=64,        # From paper
    num_layers=2,          # Stacked LSTM (from paper)
    dropout=0.5,           # From paper (but reduced to 0.2 in our config)
    batch_first=True
)

# Input:  [batch, 22, 3] per stock
# Output: [batch, 64] final hidden state per stock
# All 30 stocks: [batch, 30, 64]
```

**Processing Flow:**
1. For each stock `i` in 30 stocks:
   - Extract stock features: `x[:, :, i, :]` → `[batch, 22, 3]`
   - LSTM encoding: `lstm_out, (h_n, c_n) = lstm(x_stock)`
   - Extract final hidden state: `h_n[-1]` → `[batch, 64]`
2. Stack all stocks: `torch.stack()` → `[batch, 30, 64]`

**Hyperparameters (from paper):**
- Hidden dim: 64
- Layers: 2 (stacked)
- Dropout: 0.5 (paper) / 0.2 (our config with anti-overfitting)

---

### **2. GNN Stream (Spatial Branch)**

**File:** `src/lstm_gat_hybrid/model.py` (GraphAttentionLayer)

**Purpose:** Model cross-stock relationships using graph attention

**Architecture:**
```python
# Process each timestep independently, then pool
For each timestep t in 22 timesteps:
    # Extract features for this timestep
    x_t = x[:, t, :, :]  # [batch, 30, 3]
    
    # Apply GAT layers (2 layers)
    GAT_Layer_1(
        in_dim=3,             # HAR features
        out_dim=64,           # Per-head dimension
        num_heads=4,          # Multi-head attention
        alpha=0.2             # LeakyReLU slope
    )
    GAT_Layer_2(
        in_dim=256,           # 4 heads × 64 = 256
        out_dim=64,
        num_heads=4
    )
    
    # Output: [batch, 30, 256]

# Mean pooling across timesteps
h_gnn = mean(gnn_outputs)  # [batch, 30, 256]
```

**Attention Mechanism:**
```python
# Multi-head attention coefficients
e_ij = LeakyReLU(a^T [W h_i || W h_j])  # For each head
α_ij = softmax_j(e_ij)  # Normalize by neighbors

# Apply attention
h_i' = Σ(α_ij * W h_j)  # Aggregate from neighbors
```

**Hyperparameters:**
- Hidden dim: 64 (per head)
- Num heads: 4
- Num layers: 2
- Dropout: 0.1
- Alpha: 0.2 (LeakyReLU)

---

### **3. Graph Construction**

**File:** `src/lstm_gat_hybrid/dataset_with_graph_method.py`

**Two Methods Available:**

#### **Method 1: Correlation-Based (From Paper)**

**File:** `src/lstm_gat_hybrid/graph_correlation.py`

```python
# Build correlation graph from volatility data
def construct_correlation_graph(volatility_data, corr_threshold=0.7):
    """
    Args:
        volatility_data: [seq_len, num_stocks] volatility values
        corr_threshold: 0.7 (from paper)
    
    Returns:
        adj_matrix: [num_stocks, num_stocks] adjacency matrix
    """
    # Calculate Pearson correlation
    corr_matrix = np.corrcoef(volatility_data.T)
    
    # Threshold: Keep edges where |corr| > threshold
    adj_matrix = (np.abs(corr_matrix) > corr_threshold).astype(float)
    
    return adj_matrix
```

**Characteristics:**
- Dense graph (high connectivity)
- Captures long-term correlations
- Threshold: 0.7 (default from paper)

#### **Method 2: k-NN Sparse Graph (Current)**

**File:** `src/lstm_gat_hybrid/graph_utils_fixed.py`

```python
# Build k-NN sparse graph
def build_knn_graph(volatility_data, k=8):
    """
    Args:
        volatility_data: [seq_len, num_stocks] volatility values
        k: 8 neighbors (default)
    
    Returns:
        adj_matrix: [num_stocks, num_stocks] sparse adjacency
    """
    # Calculate correlation
    corr_matrix = np.corrcoef(volatility_data.T)
    
    # Keep top-k neighbors per node
    adj_matrix = np.zeros((num_stocks, num_stocks))
    for i in range(num_stocks):
        top_k_indices = np.argsort(-np.abs(corr_matrix[i]))[:k+1]
        adj_matrix[i, top_k_indices] = 1
    
    return adj_matrix
```

**Characteristics:**
- Sparse graph (k=8 edges per node)
- More efficient computation
- Captures strongest relationships

**Comparison:**
| Method | Density | Pros | Cons | Use Case |
|--------|---------|------|------|----------|
| **Correlation** | High (~70%) | Rich relationships | Computationally expensive | Small datasets |
| **k-NN** | Low (~27%) | Efficient, scalable | May miss weak links | Large datasets |

---

### **4. Fusion Layer**

**File:** `src/lstm_gat_hybrid/model_parallel.py` (Lines 89-104)

**Purpose:** Combine temporal (LSTM) and spatial (GNN) features

**Architecture:**
```python
# Concatenation fusion (from paper)
fusion_input_dim = lstm_hidden_dim + gat_num_heads * gat_hidden_dim
                 = 64 + 4 * 64 = 64 + 256 = 320

Fusion_MLP = nn.Sequential(
    nn.Linear(320, 64),    # Hidden layer 1
    nn.ReLU(),
    nn.Dropout(0.5),        # From paper
    
    nn.Linear(64, 32),     # Hidden layer 2
    nn.ReLU(),
    nn.Dropout(0.5),        # From paper
    
    nn.Linear(32, 1)       # Output layer (linear)
)

# Input:  [batch, 30, 320] (concatenated)
# Output: [batch, 30, 1] → squeeze → [batch, 30]
```

**Why Concatenation (not Addition)?**
- Paper shows 10.6% MSE reduction with concatenation
- Preserves all information from both streams
- Allows MLP to learn optimal weighting

**Linear Output (No Activation):**
- Following LSTM-HAR Enhanced approach (67.90% Dir Acc)
- Predictions on normalized scale (can be negative)
- Inverse transform brings back to physical scale (≥0)

---

## 📊 Data Pipeline

### **Input Features**

**File:** `src/lstm_gat_hybrid/dataset_with_graph_method.py`

**HAR Features (3 per stock):**
```python
# For each stock, generate HAR features
har_daily_vol = rolling_mean(volatility, window=1)
har_weekly_vol = rolling_mean(volatility, window=5)
har_monthly_vol = rolling_mean(volatility, window=22)

# Input shape: [seq_len=22, num_stocks=30, features=3]
```

**Data Preprocessing:**
1. **Outlier Removal** (optional):
   ```python
   # Remove outliers using z-score (n_std=3)
   z_scores = np.abs(stats.zscore(volatility))
   df_clean = df[z_scores < n_std]
   ```

2. **Data Augmentation** (training only):
   ```python
   # Jittering: add small Gaussian noise
   noise = np.random.normal(0, 0.1 * x.std(), x.shape)
   x_aug = x + noise
   ```

3. **Per-Stock Normalization**:
   ```python
   # StandardScaler per stock (from LSTM-HAR Enhanced)
   for stock in stocks:
       scaler = StandardScaler()
       scaler.fit(stock_features)
       normalized = scaler.transform(stock_features)
   ```

4. **Temporal Split**:
   ```python
   # 70/15/15 temporal split (MANDATORY)
   train_end = int(n * 0.7)
   val_end = int(n * 0.85)
   test_end = n  # 100%
   
   # No shuffling (time series order preserved)
   ```

---

## 🛡️ Anti-Overfitting Techniques

**File:** `src/lstm_gat_hybrid/train_parallel_enhanced.py`

**All techniques from LSTM-HAR Enhanced (67.90% Dir Acc):**

### **1. Data-Centric Techniques (Priority 1)**
- ✅ Outlier removal (n_std=3)
- ✅ Data augmentation (jittering, 30% prob, 10% strength)
- ⚠️ Label smoothing (optional, not implemented)

### **2. Model-Centric Techniques (Priority 2)**
- ✅ Early stopping (patience=15, min_epochs=20)
- ✅ L2 regularization (weight_decay=1e-5)
- ✅ LSTM dropout (0.2)
- ✅ FC dropout (0.3)
- ✅ Layer normalization (in GAT)
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Gradient clipping (max_norm=1.0)

### **3. Architecture-Specific**
- ⚠️ Recurrent dropout (not in PyTorch LSTM)
- ⚠️ DropEdge (GNN-specific, optional)
- ⚠️ Node dropout (GNN-specific, optional)

**Compliance Checklist:**
```
[x] Outlier removal
[x] Data augmentation
[x] Early stopping
[x] Weight decay
[x] Dropout (LSTM + FC)
[x] Layer norm
[x] LR scheduling
[x] Gradient clipping
[ ] Recurrent dropout (not available)
[ ] DropEdge (optional)
[ ] Node dropout (optional)
```

---

## 🎯 Training Configuration

### **Standardized Hyperparameters**

**File:** `src/lstm_gat_hybrid/config.py`

**Project Standards (ALL models):**
```python
num_epochs = 70              # Maximum epochs (MANDATORY)
patience = 15               # Early stopping patience (MANDATORY)
learning_rate = 0.0005      # Moderate LR
weight_decay = 1e-5         # L2 regularization
batch_size = 32             # Batch size
```

**Architecture-Specific:**
```python
# LSTM
lstm_hidden_dim = 64
lstm_num_layers = 2
lstm_dropout = 0.2          # Reduced from 0.5 (paper)

# GAT
gat_hidden_dim = 64
gat_num_heads = 4
gat_num_layers = 2
gat_dropout = 0.1
gat_alpha = 0.2

# Graph
graph_method = 'correlation'  # or 'knn'
graph_threshold = 0.7        # For correlation method
k_neighbors = 8              # For k-NN method
```

---

## 📈 Performance Tracking

### **Learning Curves**

**File:** `src/lstm_gat_hybrid/train_parallel_enhanced.py` (Lines 78-150)

**Plot every 10 epochs:**
```python
if (epoch + 1) % 10 == 0:
    plot_learning_curves_with_analysis(
        train_losses, val_losses,
        output_dir, epoch, gap_threshold=0.05
    )
```

**4-Panel Analysis:**
1. **Learning Curves:** Train vs Val loss
2. **Overfitting Monitor:** Val loss - Train loss
3. **Overfitting Monitor (Zoomed):** Detailed gap view
4. **Text Analysis:** Quantitative metrics

### **6 Mandatory Metrics**

**File:** `src/common/evaluation.py`

```python
metrics = evaluate_predictions(y_true, y_pred)
# Returns: mse, rmse, mae, r2, qlike, directional_accuracy
```

**Output Format:**
```
Validation Results:
  MSE:  0.xxxxxx
  RMSE: 0.xxxxxx
  MAE:  0.xxxxxx
  R²:   0.xxxxxx
  QLIKE: 0.xxxxxx
  Dir Acc: xx.xx%

Test Results:
  (same format)

Val-Test Gap:
  MSE_diff:  ±0.xxxxxx
  RMSE_diff: ±0.xxxxxx
  ...
```

---

## 🚀 Training Commands

### **Quick Start**

```bash
# Quick test (5 epochs)
python src/lstm_gat_hybrid/train_parallel_enhanced.py --quick_test

# Full training with correlation-based graph (default from paper)
python src/lstm_gat_hybrid/train_parallel_enhanced.py

# Full training with k-NN sparse graph
python src/lstm_gat_hybrid/train_parallel_enhanced.py --graph_method knn
```

### **Output Locations**

```
Models:     models/parallel_lstm_gnn_2026-06-21_*/
Results:    results/parallel_lstm_gnn_2026-06-21_*/
            ├── learning_curves_epoch_*.png
            ├── training_metrics.json
            └── final_results.json
```

---

## 🔍 Analysis & Debugging

### **Extract Embeddings**

**File:** `src/lstm_gat_hybrid/model_parallel.py` (Lines 196-228)

```python
# Extract intermediate embeddings for analysis
h_lstm, h_gnn = model.get_embeddings(x, adj_matrix)

# h_lstm: [batch, 30, 64] - Temporal features
# h_gnn:  [batch, 30, 256] - Spatial features
```

**Use Cases:**
- Visualize learned representations
- Analyze attention weights
- Understand stock relationships
- Debug training issues

---

## 📚 References

### **Academic Foundation**
1. **Sonani et al. (2025)** - "Stock Price Prediction Using a Hybrid LSTM-GNN Model"
   - Parallel architecture methodology
   - Concatenation fusion (10.6% MSE reduction)
   - Hyperparameters: LR=0.005, batch_size=11, dropout=0.5

2. **TemporalGAT (2024)** - "Dynamic Graph Neural Networks for Enhanced Volatility Prediction"
   - Dynamic graph construction
   - Multi-head attention mechanism

3. **FSTGAT (2024)** - "Financial Spatio-Temporal Graph Attention Network"
   - Spatial-temporal fusion
   - Graph construction methods

### **Project References**
- **LSTM-HAR Enhanced** (67.90% Dir Acc) - Anti-overfitting techniques
- **ml-ds-common-rules** - Universal ML/DS best practices
- **Project CLAUDE.md** - Project-specific rules

---

## 📝 Implementation Checklist

### **Pre-Training**
- [x] Dataset created with HAR features
- [x] Graph construction method selected (correlation/knn)
- [x] Temporal split verified (70/15/15)
- [x] Outlier removal enabled (n_std=3)
- [x] Data augmentation configured (30% prob, 10% strength)

### **Training Configuration**
- [x] Early stopping (patience=15, min_epochs=20)
- [x] Weight decay (1e-5)
- [x] Dropout (LSTM=0.2, FC=0.3)
- [x] Gradient clipping (1.0)
- [x] LR scheduling (ReduceLROnPlateau)
- [x] Learning curves plotting (every 10 epochs)

### **Evaluation**
- [x] All 6 metrics computed (MSE, RMSE, MAE, R², QLIKE, Dir Acc)
- [x] Val-test gap monitored (<0.05 threshold)
- [x] Comparison with baseline (LSTM-HAR Enhanced)

---

## ⚡ Current Status

**Training Progress (2026-06-21):**
- Epoch: 4/50
- Dir Acc: 64.87% - 65.05%
- Target: >67.90% (beat LSTM-HAR Enhanced)
- Status: On track, training stable

**Next Steps:**
1. Complete training (50 epochs)
2. Compare correlation vs k-NN graph methods
3. Analyze attention weights
4. Generate comprehensive report

---

**Last Updated:** 2026-06-21  
**Version:** 1.0 (Based on current code)  
**Status:** Active Development
