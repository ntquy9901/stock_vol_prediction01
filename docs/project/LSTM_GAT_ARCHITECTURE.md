# LSTM-GAT Hybrid Architecture for VN30 Volatility Forecasting

**Based on:** SOTA Papers (2023-2025)
- [TemporalGAT: Dynamic Graph Neural Networks for Enhanced Volatility Prediction](https://arxiv.org/html/2410.16858v1) (arXiv, 2024)
- [FSTGAT: Financial Spatio-Temporal Graph Attention Network](https://www.mdpi.com/2073-8994/17/8/1344) (MDPI Symmetry, 2024)
- [STGAT: Spatial-Temporal Graph Attention Neural Network](https://github.com/RuizheF/STGAT) (MDPI Applied Sciences, 2025)

**Date:** 2026-06-19
**Status:** Architecture Design Phase
**Target:** Improve 5-day volatility forecast accuracy for VN30 stocks

---

## 🎯 MOTIVATION

### Current Limitations
1. **LSTM-only models** treat each stock independently, missing cross-stock relationships
2. **HAR features** only capture temporal patterns, not spatial dependencies
3. **VN30 stocks** have strong correlations and spillover effects
4. **Current best:** Enhanced LSTM-HAR with 67.90% directional accuracy

### Proposed Solution
**Temporal Graph Attention Network (TemporalGAT)** combining:
- **LSTM branches** for temporal feature learning (per-stock)
- **Graph Attention layers** for spatial relationship modeling (cross-stock)
- **Dynamic graph construction** based on correlation & volatility spillover
- **Multi-head attention** for adaptive weight learning

---

## 🏗️ ARCHITECTURE OVERVIEW

### High-Level Design

```
Input: 30 stocks × 22 features (HAR + technical) × T timesteps
    ↓
[Per-Stock LSTM Branches] → Temporal Features (30 × hidden_dim)
    ↓
[Dynamic Graph Construction] → Adjacency Matrix A_t (30 × 30)
    ↓
[Graph Attention Layers] → Spatial-Temporal Features (30 × output_dim)
    ↓
[Temporal Fusion] → Final Predictions (30 × 1)
```

### Key Components

#### **1. Per-Stock LSTM Encoder (Temporal Branch)**
```python
class LSTMTemporalEncoder(nn.Module):
    """LSTM branches for each stock's temporal patterns."""

    def __init__(self, input_dim=22, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (batch, seq_len, num_stocks, features)
        # Process each stock independently
        stocks_output = []
        for i in range(x.shape[2]):  # For each stock
            stock_x = x[:, :, i, :]  # (batch, seq_len, features)
            lstm_out, _ = self.lstm(stock_x)  # (batch, seq_len, hidden)
            stocks_output.append(lstm_out[:, -1, :])  # Last timestep

        return torch.stack(stocks_output, dim=1)  # (batch, stocks, hidden)
```

#### **2. Dynamic Graph Construction**
```python
class DynamicGraphBuilder:
    """Build dynamic graph based on correlation and volatility spillover."""

    def __init__(self, num_stocks=30, top_k=10, threshold=0.5):
        self.num_stocks = num_stocks
        self.top_k = top_k  # Keep top-k edges per node
        self.threshold = threshold  # Minimum correlation threshold

    def build_graph(self, temporal_features, returns_data):
        """
        Args:
            temporal_features: (batch, num_stocks, hidden_dim)
            returns_data: (seq_len, num_stocks) - for correlation

        Returns:
            adjacency_matrix: (batch, num_stocks, num_stocks)
        """
        batch_size = temporal_features.shape[0]
        adjacency_matrices = []

        for b in range(batch_size):
            # 1. Calculate rolling correlation (recent window)
            corr_matrix = self._calculate_correlation(returns_data)

            # 2. Calculate volatility spillover (Diebold-Yilmaz)
            spillover_matrix = self._calculate_spillover(returns_data)

            # 3. Combine: A = α*corr + (1-α)*spillover
            combined = 0.6 * corr_matrix + 0.4 * spillover_matrix

            # 4. Apply threshold and keep top-k
            combined[combined < self.threshold] = 0
            combined = self._keep_top_k(combined, self.top_k)

            # 5. Normalize
            adj_matrix = combined / (combined.sum(dim=1, keepdim=True) + 1e-8)
            adjacency_matrices.append(adj_matrix)

        return torch.stack(adjacency_matrices)

    def _calculate_correlation(self, returns):
        """Calculate Pearson correlation matrix."""
        # Using recent window (e.g., last 20 days)
        return torch.corrcoef(returns.T)

    def _calculate_spillover(self, returns):
        """Calculate Diebold-Yilmaz spillover index."""
        # Simplified spillover calculation
        var_cov = torch.cov(returns.T)
        std_dev = torch.sqrt(torch.diag(var_cov))
        corr = var_cov / (std_dev.unsqueeze(1) * std_dev.unsqueeze(0))
        return torch.abs(corr)

    def _keep_top_k(self, matrix, k):
        """Keep only top-k connections per node."""
        values, indices = torch.topk(matrix, k, dim=1)
        sparse_matrix = torch.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            sparse_matrix[i, indices[i]] = values[i]
        return sparse_matrix
```

#### **3. Graph Attention Network (Spatial Branch)**
```python
class GraphAttentionLayer(nn.Module):
    """Graph Attention layer for spatial relationship modeling."""

    def __init__(self, input_dim=128, output_dim=64, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        # Linear transformations for each head
        self.W = nn.Parameter(torch.randn(num_heads, input_dim, self.head_dim))
        self.a = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim, 1))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h, adjacency_matrix):
        """
        Args:
            h: Node features (batch, num_stocks, input_dim)
            adjacency_matrix: (batch, num_stocks, num_stocks)

        Returns:
            h_prime: Updated features (batch, num_stocks, output_dim)
        """
        batch_size, num_stocks = h.shape[0], h.shape[1]

        # Multi-head attention
        heads_output = []
        for head in range(self.num_heads):
            # Linear transformation
            Wh = torch.matmul(h, self.W[head])  # (batch, stocks, head_dim)

            # Attention mechanism
            Wh1 = torch.matmul(Wh, self.a[head][:self.head_dim, :])  # (batch, stocks, 1)
            Wh2 = torch.matmul(Wh, self.a[head][self.head_dim:, :])  # (batch, stocks, 1)

            # Broadcast add for all pairs
            e = Wh1.unsqueeze(2) + Wh2.unsqueeze(1)  # (batch, stocks, stocks, 1)
            e = self.leaky_relu(e.squeeze(-1))  # (batch, stocks, stocks)

            # Mask with adjacency matrix
            e = e.masked_fill(adjacency_matrix == 0, -1e9)

            # Softmax normalization
            attention = F.softmax(e, dim=2)  # (batch, stocks, stocks)
            attention = self.dropout(attention)

            # Apply attention to neighbors
            h_prime = torch.matmul(attention, Wh)  # (batch, stocks, head_dim)
            heads_output.append(h_prime)

        # Concatenate heads
        output = torch.cat(heads_output, dim=2)  # (batch, stocks, output_dim)
        return output


class MultiLayerGAT(nn.Module):
    """Multi-layer Graph Attention Network."""

    def __init__(self, input_dim=128, hidden_dim=64, output_dim=32,
                 num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            GraphAttentionLayer(input_dim, hidden_dim, num_heads, dropout)
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            )

        # Output layer
        if num_layers > 1:
            self.layers.append(
                GraphAttentionLayer(hidden_dim, output_dim, num_heads=1, dropout=dropout)
            )

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adjacency_matrix):
        """
        Args:
            h: Node features (batch, num_stocks, input_dim)
            adjacency_matrix: (batch, num_stocks, num_stocks)

        Returns:
            h_out: (batch, num_stocks, output_dim)
        """
        for i, layer in enumerate(self.layers):
            h = layer(h, adjacency_matrix)
            if i < len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
```

#### **4. Temporal-Spatial Fusion & Prediction Head**
```python
class LSTMGATHybrid(nn.Module):
    """Complete LSTM-GAT hybrid model for volatility forecasting."""

    def __init__(self, input_dim=22, lstm_hidden=128, gat_hidden=64,
                 gat_output=32, num_lstm_layers=2, num_gat_layers=2,
                 num_gat_heads=4, num_stocks=30, dropout=0.2):
        super().__init__()

        # Temporal branch (LSTM)
        self.lstm_encoder = LSTMTemporalEncoder(
            input_dim, lstm_hidden, num_lstm_layers
        )

        # Spatial branch (GAT)
        self.gat_layers = MultiLayerGAT(
            lstm_hidden, gat_hidden, gat_output,
            num_gat_layers, num_gat_heads, dropout
        )

        # Graph builder
        self.graph_builder = DynamicGraphBuilder(num_stocks, top_k=10)

        # Fusion & prediction
        self.fusion = nn.Sequential(
            nn.Linear(gat_output + lstm_hidden, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Single output: volatility prediction
        )

    def forward(self, x, returns_data):
        """
        Args:
            x: Input features (batch, seq_len, num_stocks, input_dim)
            returns_data: Returns for graph construction (seq_len, num_stocks)

        Returns:
            predictions: (batch, num_stocks, 1)
        """
        # 1. Temporal encoding (LSTM)
        temporal_features = self.lstm_encoder(x)  # (batch, stocks, lstm_hidden)

        # 2. Build dynamic graph
        adjacency_matrix = self.graph_builder.build_graph(
            temporal_features, returns_data
        )  # (batch, stocks, stocks)

        # 3. Spatial attention (GAT)
        spatial_features = self.gat_layers(temporal_features, adjacency_matrix)
        # (batch, stocks, gat_output)

        # 4. Fusion
        combined = torch.cat([temporal_features, spatial_features], dim=2)
        # (batch, stocks, lstm_hidden + gat_output)

        # 5. Prediction
        predictions = self.fusion(combined)  # (batch, stocks, 1)

        return predictions
```

---

## 📊 DATA PIPELINE

### Input Data Structure

```python
# Feature dimensions
NUM_STOCKS = 30  # VN30 stocks
SEQ_LENGTH = 20  # Lookback window
NUM_FEATURES = 22  # HAR (3) + Technical (19)

# Input tensor shape
INPUT_SHAPE = (batch_size, SEQ_LENGTH, NUM_STOCKS, NUM_FEATURES)

# Features breakdown
FEATURES = {
    # HAR features (3)
    'har_daily_vol': 0,
    'har_weekly_vol': 1,
    'har_monthly_vol': 2,

    # Technical indicators (19)
    'rsi': 3,
    'macd': 4,
    'bollinger_upper': 5,
    'bollinger_lower': 6,
    'volume_ma': 7,
    'volume_ratio': 8,
    'return_lag_1': 9,
    'return_lag_5': 10,
    'return_lag_10': 11,
    'volatility_lag_1': 12,
    'volatility_lag_5': 13,
    'momentum': 14,
    'stochastic_k': 15,
    'stochastic_d': 16,
    'atr': 17,
    'cci': 18,
    'williams_r': 19,
    'obv': 20,
    'adx': 21
}
```

### Graph Construction Features

```python
# For dynamic graph construction
RETURNS_DATA = {
    'shape': (seq_len, NUM_STOCKS),
    'calculation': 'log_returns = log(close_t / close_{t-1})',
    'purpose': 'correlation & spillover calculation'
}
```

---

## 🔧 TRAINING CONFIGURATION

### Hyperparameters

```python
CONFIG = {
    # Model architecture
    'input_dim': 22,
    'lstm_hidden': 128,
    'gat_hidden': 64,
    'gat_output': 32,
    'num_lstm_layers': 2,
    'num_gat_layers': 2,
    'num_gat_heads': 4,

    # Training
    'num_epochs': 70,
    'patience': 15,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,

    # Graph construction
    'num_stocks': 30,
    'top_k_neighbors': 10,
    'correlation_threshold': 0.5,

    # Regularization
    'dropout': 0.2,
    'label_smoothing': 0.1,

    # Temporal split
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15
}
```

### Loss Function

```python
# Primary: MSE (stable for training)
criterion = nn.MSELoss()

# Evaluation: All 6 metrics
from src.common.evaluation import evaluate_predictions
metrics = evaluate_predictions(y_true, y_pred)
# MSE, RMSE, MAE, R², QLIKE, Dir Acc
```

---

## 📈 EXPECTED IMPROVEMENTS

### Performance Targets

| Metric | Current (Enhanced LSTM-HAR) | Target (LSTM-GAT) | Improvement |
|--------|----------------------------|-------------------|-------------|
| **RMSE** | ~0.18 | **< 0.15** | 17% ↓ |
| **Dir Acc** | 67.90% | **> 75%** | 7% ↑ |
| **QLIKE** | ~0.12 | **< 0.10** | 17% ↓ |
| **R²** | ~0.65 | **> 0.75** | 15% ↑ |

### Key Advantages

1. **Spatial Relationships:** Captures cross-stock correlations & spillovers
2. **Dynamic Adaptation:** Graph structure evolves with market conditions
3. **Attention Mechanism:** Learns adaptive importance weights
4. **Multi-Scale Learning:** LSTM (temporal) + GAT (spatial)
5. **Interpretability:** Attention weights show influential stocks

---

## 🚀 IMPLEMENTATION ROADMAP

### Phase 1: Data Preparation (Week 1)
- [ ] Extract technical indicators for all 30 stocks
- [ ] Calculate returns for graph construction
- [ ] Create DataLoader with spatial structure
- [ ] Implement graph construction utilities

### Phase 2: Model Development (Week 2)
- [ ] Implement LSTMTemporalEncoder
- [ ] Implement DynamicGraphBuilder
- [ ] Implement GraphAttentionLayer
- [ ] Implement complete LSTMGATHybrid model

### Phase 3: Training & Evaluation (Week 3)
- [ ] Train with temporal split (70/15/15)
- [ ] Hyperparameter tuning (learning rate, heads, layers)
- [ ] Compare against Enhanced LSTM-HAR baseline
- [ ] Generate learning curves and metrics

### Phase 4: Analysis & Deployment (Week 4)
- [ ] Analyze attention weights (stock influence network)
- [ ] Visualize dynamic graph evolution
- [ ] Ablation studies (LSTM-only vs GAT-only vs Hybrid)
- [ ] Deploy best model for production

---

## 📚 REFERENCES

### SOTA Papers (2023-2025)

1. **[TemporalGAT: Dynamic Graph Neural Networks for Enhanced Volatility Prediction](https://arxiv.org/html/2410.16858v1)**
   - Pulikandala Nithish Kumar et al. (arXiv, 2024)
   - Combines GCN + GAT for volatility spillover modeling
   - 15-year study on 8 global indices
   - **Key insight:** Dynamic graphs + attention improves mid-term forecasts

2. **[FSTGAT: Financial Spatio-Temporal Graph Attention Network](https://www.mdpi.com/2073-8994/17/8/1344)**
   - MDPI Symmetry (2024)
   - Addresses non-stationary financial systems
   - **Key insight:** Overcomes single-sequence LSTM limitations

3. **[STGAT: Spatial-Temporal Graph Attention Neural Network](https://github.com/RuizheF/STGAT)**
   - MDPI Applied Sciences (2025)
   - Combines graph attention with temporal convolution
   - **Code available:** [GitHub](https://github.com/RuizheF/STGAT)

4. **[Deep Learning for Financial Time Series Prediction: A State-of-the-Art Review](https://www.sciencedirect.com/org/science/article/pii/S152614922300125X)**
   - ScienceDirect (2023)
   - Comprehensive review of GNN applications in finance

5. **[Graph-Based Stock Volatility Forecasting with Effective Transfer](https://www.mdpi.com/2504-3110/9/6/339)**
   - MDPI Mathematics (2024)
   - Pearson-correlation-based graph attention network
   - Captures inter-stock relationships

---

## 🔬 RESEARCH QUESTIONS

### Key Questions to Investigate

1. **Optimal Graph Construction:**
   - What's the best correlation threshold for VN30 stocks?
   - How many top-k neighbors per stock?
   - Correlation vs spillover: What's the optimal mix?

2. **Architecture Design:**
   - How many GAT layers? (suggested: 2-3)
   - How many attention heads? (suggested: 4-8)
   - LSTM hidden dimension vs GAT dimension ratio?

3. **Temporal Dynamics:**
   - How often to update graph structure? (daily? weekly?)
   - What lookback window for correlation calculation?
   - Rolling vs expanding window for graph features?

4. **Interpretability:**
   - Which stocks are most influential in the network?
   - How does attention weight distribution change over time?
   - Can we identify market regimes from graph structure?

---

**Status:** Architecture design complete, ready for implementation
**Next Steps:** Data preparation → Model development → Training & evaluation

---

**Sources:**
- [TemporalGAT Paper](https://arxiv.org/html/2410.16858v1)
- [FSTGAT Paper](https://www.mdpi.com/2073-8994/17/8/1344)
- [STGAT GitHub](https://github.com/RuizheF/STGAT)
- [Volatility Forecasting Review](https://www.sciencedirect.com/org/science/article/pii/S152614922300125X)
- [Graph-Based Volatility](https://www.mdpi.com/2504-3110/9/6/339)