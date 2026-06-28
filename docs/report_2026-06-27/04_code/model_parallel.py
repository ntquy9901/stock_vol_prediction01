"""
Parallel LSTM-GNN Model for Multi-Stock Volatility Prediction

Based on: Sonani et al. (2025) - "Stock Price Prediction Using a Hybrid LSTM-GNN Model"
Architecture: Parallel processing (LSTM temporal + GNN spatial) → Concatenation fusion

Key improvements over sequential architecture:
- No cascading bottleneck (both streams access raw input independently)
- Concatenation fusion (proven to work: 10.6% MSE reduction in paper)
- Paper's hyperparameters: LR=0.005, batch_size=11, dropout=0.5
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .config import LSTMGATConfig


class ParallelLSTMGNN(nn.Module):
    """
    Parallel LSTM-GNN Hybrid for Multi-Stock Volatility Prediction

    Architecture:
        Input → [LSTM Stream] ───┐
                 (Temporal)     ├──→ [Concatenate] → [Dense] → Prediction
        Input → [GNN Stream] ───┘
                 (Relational)

    Based on Sonani et al. (2025) hybrid integration methodology

    Args:
        config: LSTMGATConfig object with hyperparameters
    """

    def __init__(self, config: LSTMGATConfig):
        super(ParallelLSTMGNN, self).__init__()

        self.config = config
        self.num_stocks = config.num_stocks
        self.num_features_per_stock = config.num_features_per_stock  # HAR features: 3
        self.lstm_hidden_dim = config.lstm_hidden_dim  # 64 (from paper)
        self.gat_hidden_dim = config.gat_hidden_dim  # 64 (from paper)
        self.num_lstm_layers = config.lstm_num_layers  # 2 (stacked, from paper)
        self.num_gat_layers = config.gat_num_layers  # 2 (from paper)

        print(f"[ParallelLSTMGNN] Initializing parallel architecture:")
        print(f"  - LSTM hidden dim: {self.lstm_hidden_dim}")
        print(f"  - LSTM layers: {self.num_lstm_layers}")
        print(f"  - GAT hidden dim: {self.gat_hidden_dim}")
        print(f"  - GAT num heads: {config.gat_num_heads}")
        print(f"  - GAT layers: {self.num_gat_layers}")
        print(f"  - Fusion input dim: {self.lstm_hidden_dim + config.gat_num_heads * self.gat_hidden_dim}")

        # =====================================================================
        # Stream 1: LSTM Encoder (Temporal)
        # =====================================================================
        self.lstm_encoder = nn.LSTM(
            input_size=self.num_features_per_stock,  # HAR features (3)
            hidden_size=self.lstm_hidden_dim,        # 64 (from paper)
            num_layers=self.num_lstm_layers,          # 2 stacked layers
            batch_first=True,
            dropout=config.lstm_dropout if self.num_lstm_layers > 1 else 0  # 0.5 from paper
        )

        # =====================================================================
        # Stream 2: GAT Layers (Spatial/Relational)
        # =====================================================================
        # Import GAT layer from existing implementation
        from .model import GraphAttentionLayer

        # GAT layers: each layer outputs [batch, num_stocks, num_heads * gat_hidden_dim]
        # First layer: input dim = num_features_per_stock (3)
        # Second layer: input dim = num_heads * gat_hidden_dim (4 * 64 = 256)
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                config,
                in_dim=(config.gat_num_heads * self.gat_hidden_dim) if i > 0 else self.num_features_per_stock
            )
            for i in range(self.num_gat_layers)  # 2 layers (from paper)
        ])

        # =====================================================================
        # Fusion Layer: Concatenation + MLP (from paper)
        # =====================================================================
        # GAT outputs [batch, num_stocks, num_heads * gat_hidden_dim]
        fusion_input_dim = self.lstm_hidden_dim + config.gat_num_heads * self.gat_hidden_dim

        self.fusion = nn.Sequential(
            # First hidden layer
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # From paper

            # Second hidden layer
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),  # From paper

            # Output layer (scalar prediction)
            nn.Linear(32, 1)
            # Note: No activation (linear) - following LSTM-HAR Enhanced approach
            # Predictions will be normalized, then inverse_transformed for evaluation
        )

        print(f"[ParallelLSTMGNN] Total parameters: {self.count_parameters():,}")
        print(f"[ParallelLSTMGNN] Trainable parameters: {self.count_trainable_parameters():,}")

    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with parallel processing (no cascading bottleneck)

        Args:
            x: Input features [batch_size, seq_len, num_stocks, num_features]
            adj_matrix: Adjacency matrix [batch_size, num_stocks, num_stocks]

        Returns:
            predictions: Predicted volatility [batch_size, num_stocks]
        """
        batch_size, seq_len, num_stocks, num_features = x.shape

        # =====================================================================
        # Stream 1: LSTM Processing (Temporal)
        # =====================================================================
        # Process each stock independently through LSTM
        # Output: [batch_size, num_stocks, lstm_hidden_dim]

        lstm_embeddings_list = []

        for stock_idx in range(num_stocks):
            # Extract features for this stock: [batch_size, seq_len, num_features]
            x_stock = x[:, :, stock_idx, :]

            # LSTM encoding
            # lstm_out: [batch_size, seq_len, lstm_hidden_dim]
            # h_n: [num_layers, batch_size, lstm_hidden_dim]
            lstm_out, (h_n, c_n) = self.lstm_encoder(x_stock)

            # Use final hidden state from last layer: [batch_size, lstm_hidden_dim]
            h_stock_final = h_n[-1]

            lstm_embeddings_list.append(h_stock_final)

        # Stack: [batch_size, num_stocks, lstm_hidden_dim]
        h_lstm = torch.stack(lstm_embeddings_list, dim=1)

        # =====================================================================
        # Stream 2: GNN Processing (Spatial/Relational)
        # =====================================================================
        # Process each timestep independently, then pool across time
        # Output: [batch_size, num_stocks, gat_hidden_dim]

        gnn_embeddings_list = []

        for t in range(seq_len):
            # Extract features for this timestep: [batch_size, num_stocks, num_features]
            x_t = x[:, t, :, :]

            # Apply GAT layers
            h_t = x_t
            for gat_layer in self.gat_layers:
                h_t = gat_layer(h_t, adj_matrix)

            # h_t: [batch_size, num_stocks, gat_hidden_dim]
            gnn_embeddings_list.append(h_t)

        # Pool across timesteps: Mean pooling
        # Stack: [batch_size, seq_len, num_stocks, gat_hidden_dim]
        gnn_stack = torch.stack(gnn_embeddings_list, dim=1)

        # Mean pool: [batch_size, num_stocks, gat_hidden_dim]
        h_gnn = gnn_stack.mean(dim=1)

        # =====================================================================
        # Fusion: Concatenate + MLP (from paper methodology)
        # =====================================================================

        # Concatenate embeddings along feature dimension
        # [batch_size, num_stocks, lstm_hidden_dim + gat_hidden_dim]
        h_fused = torch.cat([h_lstm, h_gnn], dim=2)

        # Predict through dense layers
        # [batch_size, num_stocks, 1] -> squeeze -> [batch_size, num_stocks]
        predictions = self.fusion(h_fused).squeeze(-1)

        return predictions

    def get_embeddings(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract intermediate embeddings for analysis

        Args:
            x: Input features [batch_size, seq_len, num_stocks, num_features]
            adj_matrix: Adjacency matrix [batch_size, num_stocks, num_stocks]

        Returns:
            h_lstm: LSTM embeddings [batch_size, num_stocks, lstm_hidden_dim]
            h_gnn: GNN embeddings [batch_size, num_stocks, gat_hidden_dim]
        """
        batch_size, seq_len, num_stocks, num_features = x.shape

        # LSTM stream
        lstm_embeddings_list = []
        for stock_idx in range(num_stocks):
            x_stock = x[:, :, stock_idx, :]
            lstm_out, (h_n, c_n) = self.lstm_encoder(x_stock)
            lstm_embeddings_list.append(h_n[-1])
        h_lstm = torch.stack(lstm_embeddings_list, dim=1)

        # GNN stream
        gnn_embeddings_list = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            h_t = x_t
            for gat_layer in self.gat_layers:
                h_t = gat_layer(h_t, adj_matrix)
            gnn_embeddings_list.append(h_t)
        h_gnn = torch.stack(gnn_embeddings_list, dim=1).mean(dim=1)

        return h_lstm, h_gnn


def create_parallel_lstm_gat_model(config: LSTMGATConfig) -> ParallelLSTMGNN:
    """
    Factory function to create Parallel LSTM-GNN model

    Args:
        config: LSTMGATConfig object

    Returns:
        model: ParallelLSTMGNN instance
    """
    model = ParallelLSTMGNN(config)
    return model


if __name__ == '__main__':
    # Quick test
    print("Testing Parallel LSTM-GNN Model...")

    # Create config
    config = LSTMGATConfig()
    config.num_stocks = 5  # Small test
    config.num_features_per_stock = 3  # HAR features

    # Create model
    model = create_parallel_lstm_gat_model(config)

    # Create dummy data
    batch_size = 2
    seq_len = 22
    num_stocks = 5
    num_features = 3

    x = torch.randn(batch_size, seq_len, num_stocks, num_features)
    adj_matrix = torch.randn(batch_size, num_stocks, num_stocks)

    # Forward pass
    predictions = model(x, adj_matrix)

    print(f"Input shape: {x.shape}")
    print(f"Adjacency shape: {adj_matrix.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")

    # Test embedding extraction
    h_lstm, h_gnn = model.get_embeddings(x, adj_matrix)
    print(f"LSTM embeddings shape: {h_lstm.shape}")
    print(f"GNN embeddings shape: {h_gnn.shape}")

    print("✅ Parallel LSTM-GNN model test passed!")
