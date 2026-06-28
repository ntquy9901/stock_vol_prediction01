"""
LSTM-GAT Hybrid Model for Multi-Stock Volatility Prediction

Architecture:
1. Per-stock LSTM encoder: Captures temporal patterns for each stock
2. Dynamic graph construction: Builds stock relationship graph
3. Graph Attention Network: Models cross-stock dependencies
4. Fusion layer: Combines temporal and spatial features
5. Output layer: Predicts volatility for all stocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .config import LSTMGATConfig


class LSTMEncoder(nn.Module):
    """
    LSTM encoder for temporal feature learning

    Processes each stock's time series independently to capture temporal patterns
    """

    def __init__(self, config: LSTMGATConfig):
        """
        Initialize LSTM encoder

        Args:
            config: LSTM-GAT configuration
        """
        super(LSTMEncoder, self).__init__()

        self.config = config
        self.num_stocks = config.num_stocks
        self.input_dim = config.num_features_per_stock
        self.hidden_dim = config.lstm_hidden_dim
        self.num_layers = config.lstm_num_layers
        self.dropout = config.lstm_dropout

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM encoder

        Args:
            x: Input tensor [batch, seq_len, num_stocks, num_features]

        Returns:
            lstm_output: [batch, num_stocks, hidden_dim] encoded features
        """
        batch_size, seq_len, num_stocks, num_features = x.shape

        # Reshape for LSTM: process each stock separately
        # [batch, seq_len, num_stocks, num_features] -> [batch * num_stocks, seq_len, num_features]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size * num_stocks, seq_len, num_features)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Get final hidden state for each stock
        # h_n: [num_layers, batch * num_stocks, hidden_dim]
        final_hidden = h_n[-1]  # [batch * num_stocks, hidden_dim]

        # Reshape back: [batch * num_stocks, hidden_dim] -> [batch, num_stocks, hidden_dim]
        final_hidden = final_hidden.reshape(batch_size, num_stocks, self.hidden_dim)

        return final_hidden


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network layer for spatial dependency modeling

    Uses multi-head attention to model relationships between stocks
    """

    def __init__(self, config: LSTMGATConfig, in_dim: int = None):
        """
        Initialize GAT layer

        Args:
            config: LSTM-GAT configuration
            in_dim: Input dimension (if None, uses lstm_hidden_dim)
        """
        super(GraphAttentionLayer, self).__init__()

        self.config = config
        self.num_stocks = config.num_stocks
        # Use provided in_dim or default to lstm_hidden_dim
        self.in_dim = in_dim if in_dim is not None else config.lstm_hidden_dim
        self.out_dim = config.gat_hidden_dim
        self.num_heads = config.gat_num_heads
        self.dropout = config.gat_dropout
        self.alpha = config.gat_alpha

        # Linear projections for attention
        self.W = nn.Linear(self.in_dim, self.num_heads * self.out_dim, bias=False)

        # Attention coefficients
        self.a = nn.Parameter(torch.Tensor(self.num_heads, 2 * self.out_dim))

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.num_heads * self.out_dim)

        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        # Initialize parameters
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GAT layer

        Args:
            x: Node features [batch, num_stocks, in_dim]
            adj_matrix: Adjacency matrix [num_stocks, num_stocks]

        Returns:
            output: Updated node features [batch, num_stocks, num_heads * out_dim]
        """
        batch_size = x.size(0)
        num_stocks = x.size(1)  # Use actual input shape instead of config

        # ✅ FIX: Add self-loops to prevent isolated nodes causing NaN in softmax
        # If a node has no edges (all masked to -inf), softmax over all -inf produces NaN
        device = x.device
        if adj_matrix.dim() == 2:
            # Single adj matrix: [num_stocks, num_stocks]
            adj_matrix = adj_matrix.clone()  # Avoid modifying original
            adj_matrix.fill_diagonal_(1.0)  # Add self-loops
        elif adj_matrix.dim() == 3:
            # Batched adj matrix: [batch, num_stocks, num_stocks]
            adj_matrix = adj_matrix.clone()
            for i in range(adj_matrix.size(0)):
                adj_matrix[i].fill_diagonal_(1.0)

        # Linear projection
        # x: [batch, num_stocks, in_dim] -> [batch, num_stocks, num_heads * out_dim]
        h = self.W(x)

        # Reshape for multi-head attention
        # [batch, num_stocks, num_heads * out_dim] -> [batch, num_stocks, num_heads, out_dim]
        h = h.view(batch_size, num_stocks, self.num_heads, self.out_dim)

        # Self-attention mechanism
        # Compute attention coefficients
        # a: [num_heads, 2 * out_dim]
        a1 = self.a[:, :self.out_dim]   # [num_heads, out_dim]
        a2 = self.a[:, self.out_dim:]   # [num_heads, out_dim]

        # Broadcast attention computation
        # h: [batch, num_stocks, num_heads, out_dim]
        # a1: [num_heads, out_dim] -> [1, 1, num_heads, out_dim]
        attn_input1 = torch.sum(h * a1.unsqueeze(0).unsqueeze(0), dim=-1)  # [batch, num_stocks, num_heads]
        attn_input2 = torch.sum(h * a2.unsqueeze(0).unsqueeze(0), dim=-1)  # [batch, num_stocks, num_heads]

        # Compute attention scores: e_ij = LeakyReLU(a1^T h_i + a2^T h_j)
        # attn_input1: [batch, num_stocks, num_heads] -> [batch, num_stocks, 1, num_heads]
        attn_input1 = attn_input1.unsqueeze(2)
        # attn_input2: [batch, num_stocks, num_heads] -> [batch, num_stocks, 1, num_heads]
        attn_input2 = attn_input2.unsqueeze(2)
        # Broadcast for attention calculation
        # attn_input1: [batch, num_stocks, 1, num_heads] -> [batch, num_stocks, num_stocks, num_heads]
        attention_scores = attn_input1 + attn_input2.transpose(1, 2)

        # Apply LeakyReLU
        attention_scores = F.leaky_relu(attention_scores, negative_slope=self.alpha)

        # Mask attention scores with adjacency matrix
        # adj_matrix: [batch, num_stocks, num_stocks] (batched by DataLoader)
        if adj_matrix.dim() == 3:  # [batch, num_stocks, num_stocks]
            # Expand for masking with attention_scores [batch, num_stocks, num_stocks, num_heads]
            adj_matrix_expanded = adj_matrix.unsqueeze(-1)  # [batch, num_stocks, num_stocks, 1]
        else:  # [num_stocks, num_stocks]
            # Single adjacency matrix for all samples in batch
            adj_matrix_expanded = adj_matrix.unsqueeze(0).unsqueeze(-1)  # [1, num_stocks, num_stocks, 1]

        attention_scores = attention_scores.masked_fill(adj_matrix_expanded == 0, float('-inf'))

        # Apply softmax to get attention coefficients
        attention_coeffs = F.softmax(attention_scores, dim=2)  # [batch, num_stocks, num_stocks, num_heads]

        # Apply attention to neighbors
        # h: [batch, num_stocks, num_heads, out_dim]
        # attention_coeffs: [batch, num_stocks, num_stocks, num_heads]
        # Need to broadcast: for each head, compute weighted sum
        h_prime = torch.zeros_like(h)

        for head in range(self.num_heads):
            h_head = h[:, :, head, :]  # [batch, num_stocks, out_dim]

            # Extract attention for this head
            # attention_coeffs: [batch, num_stocks, num_stocks, num_heads]
            attn_head = attention_coeffs[:, :, :, head]  # [batch, num_stocks, num_stocks]

            # Compute weighted sum: h_i' = sum_j (alpha_ij * h_j)
            h_prime_head = torch.bmm(attn_head, h_head)  # [batch, num_stocks, out_dim]
            h_prime[:, :, head, :] = h_prime_head

        # Layer normalization and dropout
        h_prime = h_prime.reshape(batch_size, num_stocks, -1)  # [batch, num_stocks, num_heads * out_dim]
        h_prime = self.layer_norm(h_prime)
        h_prime = self.dropout_layer(h_prime)

        return h_prime


class FusionLayer(nn.Module):
    """
    Fusion layer to combine temporal (LSTM) and spatial (GAT) features
    """

    def __init__(self, config: LSTMGATConfig):
        """
        Initialize fusion layer

        Args:
            config: LSTM-GAT configuration
        """
        super(FusionLayer, self).__init__()

        self.config = config
        self.lstm_dim = config.lstm_hidden_dim
        self.gat_dim = config.gat_hidden_dim * config.gat_num_heads
        self.hidden_dim = config.fusion_hidden_dim
        self.output_dim = config.output_dim
        self.num_stocks = config.num_stocks

        if config.fusion_method == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(self.lstm_dim + self.gat_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.output_dim)  # Output 1 prediction per stock
            )
        elif config.fusion_method == 'add':
            # Ensure dimensions match for addition
            assert self.lstm_dim == self.gat_dim, "Dimensions must match for addition fusion"
            self.fusion = nn.Sequential(
                nn.Linear(self.lstm_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.output_dim)  # Output 1 prediction per stock
            )
        else:
            raise ValueError(f"Unknown fusion method: {config.fusion_method}")

    def forward(self, lstm_features: torch.Tensor, gat_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse temporal and spatial features

        Args:
            lstm_features: [batch, num_stocks, lstm_dim]
            gat_features: [batch, num_stocks, gat_dim]

        Returns:
            predictions: [batch, num_stocks * output_dim]
        """
        batch_size = lstm_features.size(0)

        if self.config.fusion_method == 'concat':
            # Concatenate features
            fused = torch.cat([lstm_features, gat_features], dim=-1)  # [batch, num_stocks, lstm_dim + gat_dim]
        else:
            # Add features
            fused = lstm_features + gat_features

        # Reshape to process all stocks as batch: [batch, num_stocks, features] -> [batch * num_stocks, features]
        batch_size, num_stocks, feature_dim = fused.shape
        fused_flat = fused.reshape(batch_size * num_stocks, feature_dim)

        # Apply fusion layers
        output_flat = self.fusion(fused_flat)  # [batch * num_stocks, output_dim]

        # Reshape back to [batch, num_stocks]
        output = output_flat.reshape(batch_size, num_stocks)

        return output


class LSTMGATHybrid(nn.Module):
    """
    LSTM-GAT Hybrid model for multi-stock volatility prediction

    Architecture:
    1. Per-stock LSTM encoder (temporal patterns)
    2. Dynamic graph construction (stock relationships)
    3. Graph Attention Network layers (spatial dependencies)
    4. Fusion layer (combine temporal + spatial)
    5. Output layer (volatility predictions)
    """

    def __init__(self, config: LSTMGATConfig):
        """
        Initialize LSTM-GAT Hybrid model

        Args:
            config: LSTM-GAT configuration
        """
        super(LSTMGATHybrid, self).__init__()

        self.config = config
        self.num_stocks = config.num_stocks
        self.gat_num_layers = config.gat_num_layers

        # LSTM Encoder (temporal branch)
        self.lstm_encoder = LSTMEncoder(config)

        # Graph Attention Network (spatial branch)
        # Create GAT layers with matching input/output dimensions
        self.gat_layers = nn.ModuleList()
        for i in range(config.gat_num_layers):
            if i == 0:
                # First layer: LSTM hidden dim -> GAT hidden dim
                gat_layer = GraphAttentionLayer(config)
            else:
                # Subsequent layers: output of previous layer -> next layer
                # Input dimension should be output of previous layer
                in_dim = config.gat_hidden_dim * config.gat_num_heads
                gat_layer = GraphAttentionLayer(config, in_dim=in_dim)

            self.gat_layers.append(gat_layer)

        # Fusion layer
        self.fusion = FusionLayer(config)

        # Initialize weights
        self._initialize_weights()

    def forward(
        self,
        x: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through LSTM-GAT Hybrid

        Args:
            x: Input features [batch, seq_len, num_stocks, num_features]
            adj_matrix: Adjacency matrix [batch, num_stocks, num_stocks]

        Returns:
            predictions: [batch, num_stocks * output_dim]
        """
        # Temporal encoding with LSTM
        lstm_features = self.lstm_encoder(x)  # [batch, num_stocks, lstm_hidden_dim]

        # Spatial processing with GAT
        gat_features = lstm_features  # Start with LSTM features

        for gat_layer in self.gat_layers:
            gat_features = gat_layer(gat_features, adj_matrix)

        # Fusion and prediction
        predictions = self.fusion(lstm_features, gat_features)  # [batch, num_stocks * output_dim]

        return predictions

    def _initialize_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


def create_lstm_gat_model(config: LSTMGATConfig = None) -> LSTMGATHybrid:
    """
    Factory function to create LSTM-GAT Hybrid model

    Args:
        config: LSTM-GAT configuration (uses default if None)

    Returns:
        LSTMGATHybrid model
    """
    if config is None:
        config = LSTMGATConfig()

    model = LSTMGATHybrid(config)
    return model
