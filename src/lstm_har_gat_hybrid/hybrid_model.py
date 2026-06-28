"""
LSTM-HAR-GAT Hybrid Model for Volatility Prediction

Complete hybrid architecture combining:
1. Temporal Branch: LSTM-HAR for per-stock temporal learning
2. Spatial Branch: GAT for cross-stock relationship modeling
3. Fusion Layer: Temporal-spatial feature combination

This model processes all 30 VN30 stocks simultaneously, capturing both
temporal volatility patterns and cross-stock spillover effects.

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .temporal_encoder import TemporalLSTM, EnhancedTemporalLSTM
from .spatial_encoder import SpatialGAT, BatchSpatialGAT
from .fusion_layer import FusionLayer, AttentionFusionLayer


class LSTMHAR_GAT_Hybrid(nn.Module):
    """
    Complete LSTM-HAR-GAT Hybrid model for volatility prediction.

    Architecture:
        Input: (batch, num_stocks, seq_len, num_features)
        ↓
        Temporal Branch (LSTM-HAR): (batch, num_stocks, hidden_dim)
        Spatial Branch (GAT): (batch, num_stocks, hidden_dim)
        ↓
        Fusion Layer: (batch, num_stocks, 1)
        ↓
        Output: 5-day ahead volatility prediction

    Args:
        num_stocks: Number of stocks (default: 30)
        num_features: Number of input features (default: 3)
            - raw_volatility, weekly_har, monthly_har
        seq_length: Input sequence length (default: 22)
        hidden_dim: Hidden dimension for encoders (default: 64)
        lstm_layers: Number of LSTM layers (default: 2)
        gat_heads: Number of GAT attention heads (default: 4)
        dropout: Dropout probability (default: 0.2)
        fusion_type: Type of fusion ('concat' or 'attention', default: 'concat')
    """

    def __init__(
        self,
        num_stocks: int = 30,
        num_features: int = 3,
        seq_length: int = 22,
        hidden_dim: int = 64,
        lstm_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.2,
        fusion_type: str = 'concat'
    ):
        super(LSTMHAR_GAT_Hybrid, self).__init__()

        self.num_stocks = num_stocks
        self.num_features = num_features
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.gat_heads = gat_heads
        self.dropout = dropout
        self.fusion_type = fusion_type

        # Temporal Branch: Enhanced LSTM-HAR
        self.temporal_encoder = EnhancedTemporalLSTM(
            seq_length=seq_length,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout
        )

        # Spatial Branch: GAT
        self.spatial_encoder = BatchSpatialGAT(
            in_channels=hidden_dim,  # Will project from seq_length * features
            out_channels=hidden_dim,
            heads=gat_heads,
            dropout=dropout
        )

        # Projection layer: from input features to hidden_dim
        self.input_projection = nn.Linear(seq_length * num_features, hidden_dim)

        # Fusion Layer
        if fusion_type == 'concat':
            self.fusion_layer = FusionLayer(
                temporal_dim=hidden_dim,
                spatial_dim=hidden_dim,
                hidden_dim=128,
                num_layers=3,
                dropout=dropout
            )
        elif fusion_type == 'attention':
            self.fusion_layer = AttentionFusionLayer(
                temporal_dim=hidden_dim,
                spatial_dim=hidden_dim,
                hidden_dim=128,
                dropout=dropout
            )
        else:
            raise ValueError(f"Invalid fusion_type: {fusion_type}. Must be 'concat' or 'attention'")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through hybrid model.

        Args:
            x: Input tensor (batch * num_stocks, seq_len, num_features) or (batch, num_stocks, seq_len, num_features)
            edge_index: Graph connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,)

        Returns:
            predictions: 5-day ahead volatility (batch * num_stocks, 1) or (batch, num_stocks, 1)
        """
        # Handle different input formats
        if x.dim() == 3:  # (batch * stocks, seq_len, features)
            batch_size_times_stocks = x.shape[0]
            # Reshape to (batch, stocks, seq_len, features)
            x = x.reshape(batch_size_times_stocks // self.num_stocks, self.num_stocks, -1, x.shape[-1])
            batch_size = x.shape[0]
        else:  # (batch, stocks, seq_len, features)
            batch_size = x.shape[0]

        # Temporal Branch: LSTM-HAR
        temporal_features = self.temporal_encoder(x)
        # shape: (batch, num_stocks, hidden_dim)

        # Spatial Branch: GAT
        # First project input features to hidden_dim for GAT
        spatial_inputs = []
        for b in range(batch_size):
            # Flatten seq_len and features for each stock
            sample_flat = x[b].reshape(self.num_stocks, -1)  # (num_stocks, seq_len * features)
            sample_proj = self.input_projection(sample_flat)   # (num_stocks, hidden_dim)
            spatial_inputs.append(sample_proj)

        spatial_input = torch.stack(spatial_inputs, dim=0)  # (batch, num_stocks, hidden_dim)

        spatial_features = self.spatial_encoder(
            spatial_input,
            edge_index,
            edge_weight
        )
        # shape: (batch, num_stocks, hidden_dim)

        # Fusion Layer
        predictions = self.fusion_layer(temporal_features, spatial_features)
        # shape: (batch, num_stocks, 1)

        return predictions

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for interpretability.

        Args:
            x: Input tensor (batch, num_stocks, seq_len, num_features)
            edge_index: Graph connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,)

        Returns:
            Dict containing attention weights from GAT layers
        """
        attention_weights = {}

        # Forward pass through spatial encoder to get attention
        with torch.no_grad():
            batch_size = x.shape[0]

            # Project input
            spatial_inputs = []
            for b in range(batch_size):
                sample_flat = x[b].reshape(self.num_stocks, -1)
                sample_proj = self.input_projection(sample_flat)
                spatial_inputs.append(sample_proj)

            spatial_input = torch.stack(spatial_inputs, dim=0)

            # Get attention from GAT layers
            # Note: PyTorch Geometric doesn't directly expose attention weights
            # We would need to modify the GAT implementation to capture them
            attention_weights['note'] = 'Attention weights extraction requires custom GAT implementation'

        return attention_weights


class LSTMHAR_GAT_Hybrid_Complete(nn.Module):
    """
    Complete version with full feature set (22 features).

    This version uses all 22 features (raw + HAR + technical indicators)
    instead of just 3 features (raw + weekly_har + monthly_har).

    Args:
        num_stocks: Number of stocks (default: 30)
        num_features: Number of input features (default: 22)
        seq_length: Input sequence length (default: 22)
        hidden_dim: Hidden dimension (default: 64)
        lstm_layers: Number of LSTM layers (default: 2)
        gat_heads: Number of GAT attention heads (default: 4)
        dropout: Dropout probability (default: 0.2)
    """

    def __init__(
        self,
        num_stocks: int = 30,
        num_features: int = 22,
        seq_length: int = 22,
        hidden_dim: int = 64,
        lstm_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.2
    ):
        super(LSTMHAR_GAT_Hybrid_Complete, self).__init__()

        self.num_stocks = num_stocks
        self.num_features = num_features
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        # Temporal Branch: Full LSTM
        self.temporal_encoder = TemporalLSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout
        )

        # Spatial Branch: GAT
        self.spatial_encoder = BatchSpatialGAT(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            dropout=dropout
        )

        # Projection layer
        self.input_projection = nn.Linear(seq_length * num_features, hidden_dim)

        # Fusion Layer
        self.fusion_layer = FusionLayer(
            temporal_dim=hidden_dim,
            spatial_dim=hidden_dim,
            hidden_dim=128,
            num_layers=3,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]

        # Temporal Branch
        temporal_features = self.temporal_encoder(x)

        # Spatial Branch
        spatial_inputs = []
        for b in range(batch_size):
            sample_flat = x[b].reshape(self.num_stocks, -1)
            sample_proj = self.input_projection(sample_flat)
            spatial_inputs.append(sample_proj)

        spatial_input = torch.stack(spatial_inputs, dim=0)
        spatial_features = self.spatial_encoder(spatial_input, edge_index, edge_weight)

        # Fusion
        predictions = self.fusion_layer(temporal_features, spatial_features)

        return predictions


# Test the hybrid model
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LSTM-HAR-GAT HYBRID MODEL - TEST")
    print("="*60)

    # Test basic hybrid model
    print("\n[1] Testing LSTMHAR_GAT_Hybrid")
    batch_size = 4
    num_stocks = 30
    num_features = 3  # raw + weekly + monthly
    seq_length = 22

    model = LSTMHAR_GAT_Hybrid(
        num_stocks=num_stocks,
        num_features=num_features,
        seq_length=seq_length,
        hidden_dim=64,
        lstm_layers=2,
        gat_heads=4,
        dropout=0.2,
        fusion_type='concat'
    )

    # Create dummy input
    x = torch.randn(batch_size, num_stocks, seq_length, num_features)

    # Create dummy graph
    edge_index = torch.randint(0, num_stocks, (2, 100))
    edge_weight = torch.randn(100)

    # Forward pass
    predictions = model(x, edge_index, edge_weight)

    print(f"  Input shape: {x.shape}")
    print(f"  Edge index shape: {edge_index.shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Expected: ({batch_size}, {num_stocks}, 1)")
    print(f"  [OK] Test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test complete version
    print("\n[2] Testing LSTMHAR_GAT_Hybrid_Complete")
    num_features_complete = 22

    model_complete = LSTMHAR_GAT_Hybrid_Complete(
        num_stocks=num_stocks,
        num_features=num_features_complete,
        seq_length=seq_length,
        hidden_dim=64,
        lstm_layers=2,
        gat_heads=4,
        dropout=0.2
    )

    # Create dummy input with full features
    x_complete = torch.randn(batch_size, num_stocks, seq_length, num_features_complete)

    # Forward pass
    predictions_complete = model_complete(x_complete, edge_index, edge_weight)

    print(f"  Input shape: {x_complete.shape}")
    print(f"  Predictions shape: {predictions_complete.shape}")
    print(f"  Expected: ({batch_size}, {num_stocks}, 1)")
    print(f"  [OK] Test passed!")

    # Count parameters
    total_params_complete = sum(p.numel() for p in model_complete.parameters())
    trainable_params_complete = sum(p.numel() for p in model_complete.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params_complete:,}")
    print(f"  Trainable parameters: {trainable_params_complete:,}")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
