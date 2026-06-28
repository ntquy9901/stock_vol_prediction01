"""
Fusion Layer: Temporal-Spatial Feature Combination

Combines temporal (LSTM-HAR) and spatial (GAT) features through
concatenation followed by MLP layers for final volatility prediction.

Architecture:
    Input: Concatenation of temporal and spatial features
    MLP: 3 fully connected layers with ReLU and dropout
    Output: 5-day ahead volatility prediction

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FusionLayer(nn.Module):
    """
    Fusion layer combining temporal and spatial features.

    Uses concatenation strategy to preserve all information from
    both branches, followed by MLP for non-linear combination.

    Args:
        temporal_dim: Temporal feature dimension (default: 64)
        spatial_dim: Spatial feature dimension (default: 64)
        hidden_dim: Hidden dimension for MLP (default: 128)
        num_layers: Number of MLP layers (default: 3)
        dropout: Dropout probability (default: 0.2)
    """

    def __init__(
        self,
        temporal_dim: int = 64,
        spatial_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super(FusionLayer, self).__init__()

        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input dimension (concatenation)
        input_dim = temporal_dim + spatial_dim

        # Build MLP layers
        layers = []
        current_dim = input_dim

        # Hidden layers
        for i in range(num_layers):
            if i == 0:
                # First layer: input -> hidden_dim
                layers.append(nn.Linear(current_dim, hidden_dim))
            elif i < num_layers - 1:
                # Middle layers: hidden_dim -> hidden_dim
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                # Last layer: hidden_dim -> hidden_dim // 2
                layers.append(nn.Linear(hidden_dim, hidden_dim // 2))

            current_dim = layers[-1].out_features

        self.mlp_layers = nn.ModuleList(layers)

        # Output layer (single prediction)
        self.output_layer = nn.Linear(hidden_dim // 2, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        temporal_features: torch.Tensor,
        spatial_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through fusion layer.

        Args:
            temporal_features: Temporal features (batch, num_stocks, temporal_dim)
            spatial_features: Spatial features (batch, num_stocks, spatial_dim)

        Returns:
            predictions: Volatility predictions (batch, num_stocks, 1)
        """
        # Concatenate temporal and spatial features
        x = torch.cat([temporal_features, spatial_features], dim=-1)
        # shape: (batch, num_stocks, temporal_dim + spatial_dim)

        # Apply MLP layers
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Output layer
        predictions = self.output_layer(x)
        # shape: (batch, num_stocks, 1)

        return predictions


class AttentionFusionLayer(nn.Module):
    """
    Attention-based fusion layer for weighted combination.

    Uses attention mechanism to learn optimal weighting between
    temporal and spatial features dynamically.

    Args:
        temporal_dim: Temporal feature dimension (default: 64)
        spatial_dim: Spatial feature dimension (default: 64)
        hidden_dim: Hidden dimension (default: 128)
        dropout: Dropout probability (default: 0.2)
    """

    def __init__(
        self,
        temporal_dim: int = 64,
        spatial_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        super(AttentionFusionLayer, self).__init__()

        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim

        # Attention weights
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)
        self.spatial_proj = nn.Linear(spatial_dim, hidden_dim)

        # Attention scoring
        self.attention_scorer = nn.Linear(hidden_dim, 1)

        # Output layers
        self.fc1 = nn.Linear(temporal_dim + spatial_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        temporal_features: torch.Tensor,
        spatial_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through attention fusion layer.

        Args:
            temporal_features: Temporal features (batch, num_stocks, temporal_dim)
            spatial_features: Spatial features (batch, num_stocks, spatial_dim)

        Returns:
            predictions: Volatility predictions (batch, num_stocks, 1)
        """
        # Project features
        temporal_proj = self.temporal_proj(temporal_features)  # (batch, stocks, hidden)
        spatial_proj = self.spatial_proj(spatial_features)      # (batch, stocks, hidden)

        # Compute attention scores
        temporal_score = self.attention_scorer(temporal_proj)   # (batch, stocks, 1)
        spatial_score = self.attention_scorer(spatial_proj)     # (batch, stocks, 1)

        # Softmax normalization
        attention_weights = F.softmax(torch.cat([temporal_score, spatial_score], dim=-1), dim=-1)
        # shape: (batch, stocks, 2)

        # Weighted combination
        temporal_weight = attention_weights[:, :, 0:1]  # (batch, stocks, 1)
        spatial_weight = attention_weights[:, :, 1:2]   # (batch, stocks, 1)

        weighted_temporal = temporal_weight * temporal_features
        weighted_spatial = spatial_weight * spatial_features

        # Concatenate weighted features
        x = torch.cat([weighted_temporal, weighted_spatial], dim=-1)
        # shape: (batch, stocks, temporal_dim + spatial_dim)

        # MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output
        predictions = self.output(x)
        # shape: (batch, stocks, 1)

        return predictions


# Test the fusion layer
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FUSION LAYER - TEST")
    print("="*60)

    # Test basic FusionLayer
    print("\n[1] Testing FusionLayer")
    batch_size = 4
    num_stocks = 30
    temporal_dim = 64
    spatial_dim = 64

    fusion = FusionLayer(
        temporal_dim=temporal_dim,
        spatial_dim=spatial_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2
    )

    # Create dummy features
    temporal_features = torch.randn(batch_size, num_stocks, temporal_dim)
    spatial_features = torch.randn(batch_size, num_stocks, spatial_dim)

    # Forward pass
    predictions = fusion(temporal_features, spatial_features)

    print(f"  Temporal features shape: {temporal_features.shape}")
    print(f"  Spatial features shape: {spatial_features.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Expected: ({batch_size}, {num_stocks}, 1)")
    print(f"  [OK] Test passed!")

    # Test AttentionFusionLayer
    print("\n[2] Testing AttentionFusionLayer")
    attention_fusion = AttentionFusionLayer(
        temporal_dim=temporal_dim,
        spatial_dim=spatial_dim,
        hidden_dim=128,
        dropout=0.2
    )

    # Forward pass
    attention_predictions = attention_fusion(temporal_features, spatial_features)

    print(f"  Predictions shape: {attention_predictions.shape}")
    print(f"  Expected: ({batch_size}, {num_stocks}, 1)")
    print(f"  [OK] Test passed!")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
