"""
Spatial Encoder: Graph Attention Network for Cross-Stock Relationships

Implements spatial relationship modeling using GCN + GAT layers.
Captures volatility spillovers and dependencies between stocks.

Architecture:
    Input: Node features (num_stocks, in_channels)
    GCN Layers: 2 layers for structural aggregation
    GAT Layers: 2 layers with multi-head attention
    Output: Updated node embeddings (num_stocks, out_channels)

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

try:
    from torch_geometric.nn import GATConv, GCNConv
except ImportError:
    raise ImportError(
        "PyTorch Geometric required for spatial encoder. "
        "Install with: pip install torch-geometric"
    )


class SpatialGAT(nn.Module):
    """
    Spatial Graph Attention Network encoder for cross-stock relationships.

    Combines GCN (structural aggregation) with GAT (attention mechanism)
    to model volatility spillovers and dependencies between stocks.

    Args:
        in_channels: Input feature dimension (default: 64)
        out_channels: Output feature dimension (default: 64)
        heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.2)
        concat: Whether to concatenate attention heads (default: True)
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
        concat: bool = True
    ):
        super(SpatialGAT, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.concat = concat

        # GCN Layer 1: Structural aggregation
        self.gcn1 = GCNConv(in_channels, out_channels)

        # GAT Layer 1: Multi-head attention
        # Output dim per head: out_channels // heads if concat=True
        gat1_out_channels = out_channels // heads if concat else out_channels
        self.gat1 = GATConv(
            in_channels=out_channels,
            out_channels=gat1_out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            edge_dim=1  # Use edge weights
        )

        # GCN Layer 2
        self.gcn2 = GCNConv(out_channels, out_channels)

        # GAT Layer 2
        self.gat2 = GATConv(
            in_channels=out_channels,
            out_channels=gat1_out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            edge_dim=1
        )

        self.output_dim = out_channels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through spatial encoder.

        Args:
            x: Node features (num_stocks, in_channels)
            edge_index: Graph connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,) - optional

        Returns:
            output: Updated node embeddings (num_stocks, out_channels)
        """
        # GCN 1
        x = self.gcn1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT 1
        x = self.gat1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GCN 2
        x = self.gcn2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT 2
        x = self.gat2(x, edge_index, edge_weight)
        x = F.relu(x)

        return x

    def get_output_dim(self) -> int:
        """Return output dimension of the encoder."""
        return self.output_dim


class BatchSpatialGAT(nn.Module):
    """
    Batch processing wrapper for Spatial GAT.

    Processes multiple graph samples in a batch by applying
    the same graph structure to each sample.

    Args:
        in_channels: Input feature dimension (default: 64)
        out_channels: Output feature dimension (default: 64)
        heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.2)
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.2
    ):
        super(BatchSpatialGAT, self).__init__()

        self.spatial_encoder = SpatialGAT(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for batched inputs.

        Args:
            x: Node features (batch, num_stocks, in_channels)
            edge_index: Graph connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,)

        Returns:
            output: Updated node embeddings (batch, num_stocks, out_channels)
        """
        batch_size, num_stocks, in_channels = x.shape

        # Process each batch sample
        outputs = []
        for b in range(batch_size):
            # Extract sample: (num_stocks, in_channels)
            sample_features = x[b, :, :]

            # Apply spatial encoder
            sample_output = self.spatial_encoder(
                sample_features,
                edge_index,
                edge_weight
            )

            outputs.append(sample_output)

        # Stack outputs: (batch_size, num_stocks, out_channels)
        outputs = torch.stack(outputs, dim=0)

        return outputs

    def get_output_dim(self) -> int:
        """Return output dimension of the encoder."""
        return self.spatial_encoder.get_output_dim()


# Test the encoder
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SPATIAL ENCODER - TEST")
    print("="*60)

    # Test basic SpatialGAT
    print("\n[1] Testing SpatialGAT")
    num_stocks = 30
    in_channels = 64
    out_channels = 64
    heads = 4

    encoder = SpatialGAT(
        in_channels=in_channels,
        out_channels=out_channels,
        heads=heads,
        dropout=0.2
    )

    # Create dummy input
    x = torch.randn(num_stocks, in_channels)

    # Create dummy graph (random edges)
    edge_index = torch.randint(0, num_stocks, (2, 100))
    edge_weight = torch.randn(100)

    # Forward pass
    output = encoder(x, edge_index, edge_weight)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({num_stocks}, {out_channels})")
    print(f"  [OK] Test passed!")

    # Test BatchSpatialGAT
    print("\n[2] Testing BatchSpatialGAT")
    batch_size = 4

    batch_encoder = BatchSpatialGAT(
        in_channels=in_channels,
        out_channels=out_channels,
        heads=heads,
        dropout=0.2
    )

    # Create dummy batch input
    x_batch = torch.randn(batch_size, num_stocks, in_channels)

    # Forward pass
    output_batch = batch_encoder(x_batch, edge_index, edge_weight)

    print(f"  Input shape: {x_batch.shape}")
    print(f"  Output shape: {output_batch.shape}")
    print(f"  Expected: ({batch_size}, {num_stocks}, {out_channels})")
    print(f"  [OK] Test passed!")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
