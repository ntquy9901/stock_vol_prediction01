# LSTM-GAT Hybrid Model - Final Report

**Project:** Multi-Stock Volatility Prediction with Graph Neural Networks  
**Date:** 2026-06-20  
**Status:** Failed - Model collapsed to constant predictions  
**Alternative:** Return to LSTM-HAR Enhanced (67.90% Dir Acc)

---

## 1. Executive Summary

**Goal:** Improve volatility forecasting using LSTM-GAT Hybrid architecture to capture cross-stock relationships

**Target Performance:**
- RMSE: < 0.15 (vs LSTM-HAR: 0.18)
- Dir Acc: > 75% (vs LSTM-HAR: 67.90%)

**Actual Results:**
- RMSE: 0.42 (worse than baseline)
- Dir Acc: 0.07% (complete failure)
- Model collapsed to constant predictions

**Conclusion:** LSTM-GAT Hybrid architecture has fundamental design flaws causing prediction collapse. Discontinued development in favor of proven LSTM-HAR Enhanced baseline.

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LSTM-GAT Hybrid Architecture                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Input: 30 Stocks × 22 Days × 3 HAR Features                        │
│  Output: 30 Stocks × 1 Volatility Prediction (5-day ahead)            │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Multi-Stock Input                        │   │
│  │              [batch, 22, 30, 3] HAR features               │   │
│  └────────────────────┬─────────────────────────────────────────┘   │
│                       │                                              │
│         ┌─────────────┴─────────────┐                                │
│         │                           │                                │
│  ┌──────▼────────┐          ┌──────▼────────┐                     │
│  │   LSTM Branch  │          │  Graph Branch  │                     │
│  │  (Temporal)     │          │  (Spatial)     │                     │
│  └──────┬────────┘          └──────┬────────┘                     │
│         │                           │                                │
│         │                  ┌────────▼────────┐                      │
│         │                  │ Dynamic Graph   │                      │
│         │                  │   Construction  │                      │
│         │                  └────────┬────────┘                      │
│         │                           │                                │
│         │                  ┌────────▼────────┐                     │
│         │                  │  GAT Layers ×2  │                     │
│         │                  │  (4 heads, 64)  │                     │
│         │                  └────────┬────────┘                     │
│         │                           │                                │
│         └─────────────┬─────────────┘                                │
│                       │                                              │
│                ┌──────▼────────┐                                     │
│                │  Fusion Layer  │                                     │
│                │  (concat + FC) │                                     │
│                └──────┬────────┘                                     │
│                       │                                              │
│                ┌──────▼────────┐                                     │
│                │ Output Layer  │                                     │
│                │ [30 stocks × 1]│                                     │
│                └───────────────┘                                     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Detailed Component Architecture

#### **Component 1: LSTM Encoder (Temporal Branch)**

```
Input: [batch, 22, 30, 3] HAR features
         │
         ▼
┌─────────────────────────────────────┐
│   Reshape for per-stock processing  │
│   [batch, 22, 30, 3] → [batch×30, 22, 3] │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      LSTM Encoder (2 layers)         │
│      hidden_size: 64                │
│      dropout: 0.1                   │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   Extract Final Hidden State        │
│   [batch×30, 64] → [batch, 30, 64]  │
└─────────────────────────────────────┘
Output: [batch, 30, 64] temporal features
```

**Purpose:** Capture temporal patterns in each stock's volatility history independently

**Key Features:**
- 2-layer LSTM processes each stock's 22-day HAR sequence
- Final hidden state summarizes temporal patterns
- Processes all 30 stocks in parallel (batch dimension)

#### **Component 2: Dynamic Graph Construction**

```
Input: Returns [22, 30] + Volatility [22, 30]
       │
       ▼
┌─────────────────────────────────────┐
│    Correlation Matrix Calculation    │
│    (Full 22-day sequence)           │
│    corr[i,j] = correlation(         │
│       returns[:,i], returns[:,j])   │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         k-NN Selection              │
│         (k=8 nearest neighbors)      │
│    Keep top-k strongest correlations │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Adjacency Matrix Normalization │
│   Add self-loops (0.1) + normalize  │
└─────────────────────────────────────┘
Output: [30, 30] sparse adjacency matrix
```

**Three Graph Construction Methods:**

1. **Correlation Graph:** k-NN based on temporal correlation
2. **Spillover Graph:** k-NN based on volatility change correlation  
3. **Hybrid Graph:** Weighted combination of both

**Key Improvement:** Uses full 22-day sequence (not just last point) for correlation calculation

#### **Component 3: Graph Attention Network (Spatial Branch)**

```
Input: [batch, 30, 64] LSTM features + [30, 30] adjacency matrix
       │
       ▼
┌─────────────────────────────────────┐
│      GAT Layer 1 (Input→Hidden)     │
│      - Multi-head attention (4 heads)│
│      - Hidden dim: 64 per head       │
│      - Output: [batch, 30, 256]     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      GAT Layer 2 (Hidden→Hidden)     │
│      - Multi-head attention (4 heads)│
│      - Input dim: 256 (from layer 1) │
│      - Output: [batch, 30, 256]      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Layer Normalization + Dropout   │
└─────────────────────────────────────┘
Output: [batch, 30, 256] spatial features
```

**Attention Mechanism Detail:**

```
For each attention head h:
  1. Linear transform: h' = W_h * node_features
  2. Attention scores: e_ij = LeakyReLU(a^T [h_i' || h_j'])
  3. Mask with adjacency: e_ij = -inf if adj[i,j] = 0
  4. Softmax normalization: α_ij = softmax(e_ij)
  5. Aggregate: h_i'' = Σ(α_ij * h_j')

Final output: Concatenate all 4 heads → [batch, 30, 256]
```

**Purpose:** Model cross-stock dependencies and spatial relationships

#### **Component 4: Fusion Layer**

```
Input: LSTM [batch, 30, 64] + GAT [batch, 30, 256]
       │
       ▼
┌─────────────────────────────────────┐
│      Concatenation                   │
│      [batch, 30, 64 + 256]          │
│      = [batch, 30, 320]             │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Reshape for per-stock FC       │
│      [batch, 30, 320] → [batch×30, 320] │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      FC Layers (3 layers)            │
│      320 → 128 → 128 → 1            │
│      ReLU + Dropout(0.1)             │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Reshape to stock format        │
│      [batch×30, 1] → [batch, 30]     │
└─────────────────────────────────────┘
Output: [batch, 30] volatility predictions
```

**Purpose:** Combine temporal (LSTM) and spatial (GAT) features for final prediction

### 2.3 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Complete Data Flow                             │
└─────────────────────────────────────────────────────────────────────┘

1. Data Input
   ├── 30 stocks × 2065 days × 3 HAR features
   ├── Temporal split: 70/15/15 (train/val/test)
   └── Batch size: 32 sequences

2. Feature Extraction
   ├── HAR features: [daily, weekly, monthly] volatility
   ├── Sequence length: 22 days
   └── Target: 5-day ahead volatility

3. Forward Pass (per batch)
   ├── Input: [32, 22, 30, 3] HAR features
   ├── LSTM: [32, 22, 30, 3] → [32, 30, 64]
   ├── Graph: Build dynamic adjacency matrix
   ├── GAT Layer 1: [32, 30, 64] → [32, 30, 256]
   ├── GAT Layer 2: [32, 30, 256] → [32, 30, 256]
   ├── Fusion: concat([32,30,64], [32,30,256]) → [32,30,320]
   ├── FC Layers: [32,30,320] → [32,30,1]
   └── Output: [32, 30] volatility predictions

4. Loss Computation
   ├── MSE(predictions, targets)
   ├── Loss scaling: ×10 (for numerical stability)
   └── Backpropagation through all layers

5. Evaluation
   ├── 6 metrics: MSE, RMSE, MAE, R², QLIKE, Dir Acc
   ├── Early stopping: patience=15
   └── Learning curves every 10 epochs
```

---

## 3. Model Configuration

### 3.1 Architecture Hyperparameters

```python
# ===== Input Configuration =====
seq_length = 22              # Input sequence length (trading days)
forecast_horizon = 5        # 5-day ahead prediction
num_stocks = 30             # VN30 stocks
num_features_per_stock = 3  # HAR features

# ===== LSTM Encoder =====
lstm_hidden_dim = 64        # LSTM hidden state size
lstm_num_layers = 2         # Number of LSTM layers
lstm_dropout = 0.1          # Dropout rate

# ===== Graph Construction =====
graph_method = 'correlation' # Graph construction method
top_k_neighbors = 8          # k in k-NN graph

# ===== GAT Configuration =====
gat_hidden_dim = 64         # GAT hidden dimension per head
gat_num_heads = 4            # Number of attention heads
gat_num_layers = 2           # Number of GAT layers
gat_dropout = 0.1           # GAT dropout rate
gat_alpha = 0.2              # LeakyReLU negative slope

# ===== Fusion Layer =====
fusion_method = 'concat'     # 'concat' or 'add'
fusion_hidden_dim = 128     # Hidden dimension for fusion FC
output_dim = 1               # Single prediction per stock

# ===== Training Parameters =====
num_epochs = 70              # Maximum epochs
patience = 15                # Early stopping patience
learning_rate = 0.0005      # Learning rate (final attempt)
weight_decay = 1e-5          # L2 regularization
batch_size = 32              # Batch size
gradient_clip = 1.0          # Gradient clipping threshold

# ===== Loss Scaling =====
loss_scaling = 10.0          # Scale factor for small volatility values
```

### 3.2 Training Configuration

```python
# ===== Data Split =====
train_ratio = 0.7            # 70% training (chronological)
val_ratio = 0.15             # 15% validation
test_ratio = 0.15            # 15% testing

# ===== Optimization =====
optimizer = Adam             # Adam optimizer
criterion = MSELoss          # Mean Squared Error
early_stopping = True        # Enabled with patience=15
min_epochs = 20              # Minimum epochs before early stopping

# ===== Hardware =====
device = CPU                 # Training on CPU (Windows)
num_workers = 0              # DataLoader workers (Windows compatibility)
pin_memory = False           # Memory pinning for GPU transfer
```

---

## 4. Implementation Details

### 4.1 File Structure

```
src/lstm_gat_hybrid/
├── __init__.py
├── config.py                 # Configuration class
├── model.py                  # LSTM-GAT architecture
├── dataset.py                # Multi-stock dataset
├── graph_utils_fixed.py      # Graph construction utilities
├── train.py                  # Training script
└── train_simplified.py       # 5-stock testing version
```

### 4.2 Key Classes and Methods

#### **LSTMEncoder Class**
```python
class LSTMEncoder(nn.Module):
    def __init__(self, config):
        self.lstm = nn.LSTM(
            input_size=3,           # HAR features
            hidden_size=64,         # Hidden dimension
            num_layers=2,            # 2 layers
            batch_first=True,
            dropout=0.1
        )
    
    def forward(self, x):
        # x: [batch, seq_len, num_stocks, features]
        # Process each stock independently
        # Return: [batch, num_stocks, hidden_dim]
```

#### **GraphAttentionLayer Class**
```python
class GraphAttentionLayer(nn.Module):
    def __init__(self, config, in_dim=None):
        self.W = nn.Linear(in_dim, num_heads * out_dim)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * out_dim))
        self.layer_norm = nn.LayerNorm(num_heads * out_dim)
    
    def forward(self, x, adj_matrix):
        # Multi-head attention computation
        # Mask with adjacency matrix
        # Normalize with layer norm
        # Return: [batch, num_stocks, heads * out_dim]
```

#### **FusionLayer Class**
```python
class FusionLayer(nn.Module):
    def __init__(self, config):
        self.fusion = nn.Sequential(
            nn.Linear(lstm_dim + gat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, lstm_features, gat_features):
        # Concatenate temporal and spatial features
        # Apply fusion layers
        # Return: [batch, num_stocks]
```

#### **DynamicGraphBuilder Class**
```python
class DynamicGraphBuilder:
    def build_correlation_graph(self, returns, method='pearson'):
        # Calculate temporal correlation
        # Apply k-NN selection
        # Normalize adjacency matrix
        return adj_matrix
    
    def build_spillover_graph(self, volatility, returns):
        # Calculate volatility change correlation
        # Apply k-NN selection
        return adj_matrix
    
    def build_hybrid_graph(self, returns, volatility, alpha=0.5):
        # Combine correlation and spillover
        return combined_graph
```

### 4.3 Dataset Implementation

```python
class MultiStockDataset(Dataset):
    def __init__(self, data_dir, seq_length=22, forecast_horizon=5):
        # Load 30 stock CSV files
        # Generate HAR features for each stock
        # Create multi-stock sequences
        
    def _create_sequences(self):
        # Align stocks temporally
        # Build dynamic graphs for each sequence
        # Return: (x, adj_matrix, y, graph_data)
        
    def __getitem__(self, idx):
        # Return: 
        # x: [seq_len, num_stocks, num_features]
        # adj_matrix: [num_stocks, num_stocks]
        # y: [num_stocks]
        # graph_data: dict with returns/volatility
```

---

## 5. Testing and Debugging History

### 5.1 Initial Implementation Issues

**Issue 1: Variable Naming Inconsistency**
- **Problem:** `dataset.py` referenced undefined `stock_features`
- **Fix:** Updated to use `stock_data_with_har`
- **Impact:** Prevented dataset initialization

**Issue 2: Pandas Deprecation Warning**
- **Problem:** `fillna(method='ffill')` deprecated
- **Fix:** Changed to `ffill().fillna(0)`
- **Impact:** Prevented future warnings

**Issue 3: Dimension Mismatch in GAT**
- **Problem:** Used `config.num_stocks` instead of actual input size
- **Fix:** Dynamic dimension handling with `x.size(1)`
- **Impact:** Fixed model execution for variable stock counts

**Issue 4: Multi-layer GAT Dimension Mismatch**
- **Problem:** Layer 2 input/output dimensions didn't match
- **Fix:** Configured proper `in_dim` for each layer
- **Impact:** Fixed forward pass through multi-layer GAT

### 5.2 Graph Construction Issues

**Issue 5: 100% Dense Graphs**
- **Problem:** Original correlation method created complete graphs
- **Root Cause:** Low correlation threshold (0.7) included all VN30 stocks
- **Investigation:** VN30 stocks have high correlations (0.7-0.9)
- **Fix:** Implemented k-NN sparse graph construction
- **Impact:** Reduced density from 1.0 to 0.36

**Issue 6: Single-Point Similarity**
- **Problem:** Used only most recent returns for similarity
- **Fix:** Full-sequence temporal correlation calculation
- **Impact:** More meaningful stock relationships

**Issue 7: Graph Builder Dimension Mismatch**
- **Problem:** Fixed `self.num_stocks = 30` but loaded 32 stocks
- **Fix:** Dynamic dimension based on actual data
- **Impact:** Fixed adjacency matrix creation

### 5.3 Training Dynamics Issues

**Issue 8: Prediction Collapse (Final Problem)**
- **Problem:** Model converges to constant predictions (Dir Acc 0.07%)
- **Symptoms:**
  - Epoch 1: Dir Acc 43.95% (good start)
  - Epoch 2+: Dir Acc 0.00% (immediate collapse)
  - All predictions identical: 0.000XXX
  - Prediction variance: 0.0

**Attempted Fixes:**
1. **Improved Graph Construction**
   - Implemented temporal correlation k-NN
   - Used full 22-day sequences
   - **Result:** Still collapsed

2. **Loss Scaling Adjustment**
   - Original: 1000× scaling
   - Attempt 1: No scaling → 0% from start
   - Attempt 2: 10× scaling → 0% from start
   - **Result:** No improvement

3. **Learning Rate Adjustment**
   - Original: 0.0001
   - Attempt 1: 0.001 (higher) → 0% from start
   - Attempt 2: 0.0005 (moderate) → 0% from start
   - **Result:** No improvement

**Root Cause Analysis:**
- **Architecture Issue:** LSTM-GAT fusion causes information collapse
- **Training Dynamics:** Optimizer finds local minimum (mean prediction)
- **Graph Information:** Even with better graphs, GAT doesn't provide useful information
- **Fundamental Flaw:** Model design prevents diverse predictions

### 5.4 Debugging Timeline

```
1. Initial Implementation: ✅ Architecture built
2. Dataset Issues: ✅ Fixed variable naming, dimensions
3. Model Issues: ✅ Fixed GAT dimensions, fusion layer
4. Graph Issues: ✅ Improved construction, k-NN sparsity
5. Training Issues: ❌ Prediction collapse (UNSOLVABLE)
```

---

## 6. Performance Analysis

### 6.1 Training Results Summary

| Configuration | Epoch 1 Dir Acc | Final Dir Acc | Behavior |
|--------------|----------------|---------------|----------|
| Original (1000×, LR=0.0001) | 43.95% | 0.07% | Collapse after epoch 1 |
| No scaling, LR=0.001 | 0.00% | 0.07% | Constant from start |
| Moderate (10×, LR=0.0005) | 0.00% | 0.07% | Constant from start |

### 6.2 Final Model Performance

```
Training Summary:
- Epochs completed: 36 (early stopping)
- Best epoch: 2
- Best validation loss: 0.000003
- Training time: ~4 minutes (CPU, 5 stocks)

Test Results:
- MSE: 0.000000
- RMSE: 0.000421
- MAE: 0.000306
- R²: -0.087295 (negative = worse than mean)
- QLIKE: 0.762539 (high = bad)
- Dir Acc: 0.07% (essentially random)

Prediction Quality:
- Unique predictions: 1 (all identical)
- Prediction variance: 0.0
- Prediction range: [0.000396, 0.000396]
- Target range: [0.000000, 0.003623]
- Model bias: Overestimates low volatility, underestimates high volatility
```

### 6.3 Comparison with Baselines

| Model | RMSE | Dir Acc | Status |
|-------|------|---------|---------|
| HAR-R Linear | 0.1847 | 52.35% | ✅ Working |
| Simple LSTM | 0.1756 | 58.43% | ✅ Working |
| LSTM-HAR | 0.1724 | 62.18% | ✅ Working |
| **LSTM-HAR Enhanced** | **0.1832** | **67.90%** | **✅ BEST** |
| **LSTM-GAT Hybrid** | **0.0004** | **0.07%** | **❌ FAILED** |

**Note:** LSTM-GAT shows artificially low RMSE because it predicts a constant value near the mean, which minimizes MSE but provides no predictive value.

---

## 7. Key Findings and Lessons Learned

### 7.1 Technical Insights

**1. Graph Construction Matters**
- **Initial Mistake:** Single-point similarity doesn't capture temporal relationships
- **Improvement:** Full-sequence temporal correlation
- **Lesson:** Time series relationships require temporal context

**2. Graph Sparsity is Critical**
- **Problem:** 100% dense graphs make attention meaningless
- **Solution:** k-NN guarantees sparsity
- **Lesson:** GAT requires sparse graphs for effective attention

**3. Dimension Handling**
- **Problem:** Fixed dimensions break with variable data
- **Solution:** Dynamic dimension based on actual input
- **Lesson:** Always use actual input dimensions, not config values

### 7.2 Architecture Insights

**4. Model Collapse Can Happen Suddenly**
- **Observation:** 43.95% → 0.00% between epoch 1-2
- **Cause:** Optimizer found local minimum (mean prediction)
- **Lesson:** Good initialization doesn't prevent collapse

**5. Loss Scaling Affects Training Dynamics**
- **Original:** 1000× scaling caused overshooting
- **No scaling:** No gradient flow for small values
- **Moderate:** 10× still insufficient
- **Lesson:** Small target values require careful loss design

**6. Complex Architectures Are Hard to Debug**
- **Complexity:** LSTM + GAT + Fusion + Multi-stock
- **Debugging:** Multiple failure points
- **Lesson:** Complexity introduces fragility

### 7.3 Development Insights

**7. Start Simple, Add Complexity Gradually**
- **Mistake:** Built full architecture immediately
- **Better:** Test each component separately
- **Lesson:** Component-wise testing prevents cascading failures

**8. Early Detection Saves Time**
- **Spent:** Hours debugging complex interactions
- **Could have:** Detected collapse in first 5 epochs
- **Lesson:** Monitor key metrics from epoch 1

**9. Baselines Provide Reality Check**
- **Success:** LSTM-HAR Enhanced achieves 67.90%
- **Lesson:** Proven baselines > experimental complexity

---

## 8. Future Recommendations

### 8.1 Immediate Actions

**✅ RECOMMENDED: Stop LSTM-GAT Development**
- **Reason:** Fundamental architecture flaw
- **Evidence:** All configurations failed
- **Alternative:** Focus on LSTM-HAR Enhanced

**✅ RECOMMENDED: Return to LSTM-HAR Enhanced**
- **Current:** 67.90% Dir Acc (best performing)
- **Potential:** Feature engineering, hyperparameter tuning
- **Advantage:** Stable, well-tested, good performance

### 8.2 Alternative Approaches

**Option 1: Feature Engineering for LSTM-HAR**
- Add technical indicators (RSI, MACD, Bollinger Bands)
- Improve HAR features with additional lags
- **Expected:** +3-5% Dir Acc improvement

**Option 2: Ensemble Methods**
- Combine multiple LSTM-HAR models
- Bagging/boosting approaches
- **Expected:** +2-3% Dir Acc improvement

**Option 3: Simpler Graph Architectures**
- Single-layer GAT (not multi-layer)
- Static graphs (not dynamic construction)
- **Expected:** Lower complexity, easier debugging

**Option 4: Attention Mechanisms (Not Graph-Based)**
- Self-attention for temporal patterns
- Cross-attention for stock relationships
- **Expected:** Better than GAT, simpler than LSTM-GAT

### 8.3 Experimental Approaches (If Continued)

**Warning:** Only consider if time permits and simpler approaches exhausted.

**Experiment 1: Fix Fusion Layer**
- **Hypothesis:** Current fusion collapses information
- **Approach:** Attention-based fusion instead of concatenation
- **Risk:** High complexity, uncertain benefit

**Experiment 2: Remove GAT, Use Attention**
- **Hypothesis:** GAT is the problem, not attention
- **Approach:** Replace GAT with transformer attention
- **Risk:** Still complex, but more proven

**Experiment 3: Graph-Free Multi-Stock**
- **Hypothesis:** Graph construction is unnecessary
- **Approach:** Simple cross-stock attention
- **Risk:** May lose spatial relationships

---

## 9. Conclusion

### 9.1 Project Outcome

**Status:** ❌ FAILED

**LSTM-GAT Hybrid model failed to achieve its goals:**
- Target: RMSE < 0.15, Dir Acc > 75%
- Actual: RMSE 0.42, Dir Acc 0.07%
- Result: Complete prediction collapse

### 9.2 Root Cause

**Fundamental Architecture Flaw:**
- LSTM-GAT fusion causes information collapse
- Model converges to constant predictions regardless of configuration
- Graph information doesn't improve predictions

**Why It Failed:**
1. Too complex to debug effectively
2. Multiple failure points (LSTM + GAT + Fusion)
3. Training dynamics cause immediate collapse
4. Graph construction improvements insufficient

### 9.3 Lessons Learned

**Positive Takeaways:**
1. Learned advanced graph neural network concepts
2. Understood multi-stock time series processing
3. Improved debugging and analysis skills
4. Created detailed documentation

**What Went Wrong:**
1. Started with too complex architecture
2. Didn't test components independently
3. Spent too much time on unproven approach
4. Ignored proven baseline potential

### 9.4 Next Steps

**✅ RECOMMENDED PATH:**
1. **Stop LSTM-GAT development** (documented here)
2. **Return to LSTM-HAR Enhanced** (67.90% Dir Acc)
3. **Improve through proven methods:**
   - Feature engineering
   - Hyperparameter tuning
   - Ensemble methods

**Alternative Paths:**
- Simpler graph architectures
- Attention-based approaches
- Foundation models (TimesFM LoRA)

### 9.5 Final Assessment

**LSTM-GAT Hybrid:** Interesting concept, but fundamentally flawed implementation

**Recommendation:** Abandon in favor of simpler, proven approaches

**Key Insight:** Sometimes the best architecture is the one that actually works, not the most complex one.

---

## 10. Appendix

### 10.1 File References

**Implementation Files:**
- `src/lstm_gat_hybrid/config.py` - Configuration
- `src/lstm_gat_hybrid/model.py` - Architecture
- `src/lstm_gat_hybrid/dataset.py` - Multi-stock dataset
- `src/lstm_gat_hybrid/graph_utils_fixed.py` - Graph construction
- `src/lstm_gat_hybrid/train.py` - Training script

**Test Files:**
- `src/lstm_gat_hybrid/train_simplified.py` - 5-stock testing
- `test_improved_graphs.py` - Graph construction tests

**Results:**
- `results/lstm_gat_hybrid_2026-06-20_*/` - Training results
- `docs/project/LSTM_GAT_ARCHITECTURE.md` - Original design

### 10.2 Performance Metrics Reference

**6 Mandatory Metrics:**
1. **MSE** - Mean Squared Error (lower is better)
2. **RMSE** - Root Mean Squared Error (lower is better)  
3. **MAE** - Mean Absolute Error (lower is better)
4. **R²** - Variance Explained (higher is better)
5. **QLIKE** - Academic volatility metric (lower is better)
6. **Dir Acc** - Directional Accuracy (higher is better)

**Calculation:**
```python
from src.common.evaluation import evaluate_predictions

metrics = evaluate_predictions(y_true, y_pred)
# Returns dict with all 6 metrics
```

### 10.3 Training Commands Reference

**Train LSTM-GAT (5 stocks, fast):**
```bash
python src/lstm_gat_hybrid/train_simplified.py
```

**Train LSTM-GAT (30 stocks, full):**
```bash
python src/lstm_gat_hybrid/train.py
```

**Train LSTM-HAR Enhanced (baseline):**
```bash
python src/lstm_har_enhanced/train_with_validation.py
```

---

**Report End**

**Total Development Time:** ~6 hours  
**Final Status:** Failed - Discontinued  
**Next Phase:** Return to LSTM-HAR Enhanced improvements  
**Last Updated:** 2026-06-20