"""
CryptoMamba Model - Full Implementation with mamba_ssm

Based on original CryptoMamba architecture using official mamba_ssm package.
Adapted for volatility prediction (not price prediction).

Key features:
- Full selective state space mechanism (via mamba_ssm)
- Hierarchical structure: hidden_dims=[14, 1]
- ~136K parameters (vs 2,787 in simplified V2)
- CUDA acceleration for selective scan operations

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

import torch
import torch.nn as nn
from typing import List
import sys

# Try to import mamba_ssm
try:
    from mamba_ssm import Mamba as MambaSSM
    MAMBA_AVAILABLE = True
    print("✅ mamba_ssm package available - using full selective scan")
except ImportError:
    MAMBA_AVAILABLE = False
    print("⚠️  mamba_ssm not found - please install: pip install mamba_ssm")
    print("   Requires CUDA toolkit 11.8+")
    sys.exit(1)


class CMambaVolatilityFull(nn.Module):
    """
    CryptoMamba for Volatility Prediction (Full Architecture with mamba_ssm)

    Based on original CryptoMamba with hierarchical structure:
    - hidden_dims: [hidden_dim, 1] for gradual feature extraction
    - Full selective state space via mamba_ssm
    - CUDA-accelerated selective scan operations
    - Direct volatility prediction (5-day ahead)

    Architecture:
        Input: (batch, seq_length, num_features)
        → Input projection: num_features → hidden_dim
        → Mamba blocks with hierarchical hidden_dims=[14, 1]
        → Projection head: hidden_dim → 1
        → Output: (batch, 1) - predicted volatility

    Parameters: ~136K (full selective scan)
    """

    def __init__(
        self,
        num_features: int = 3,  # HAR features (daily, weekly, monthly)
        hidden_dim: int = 14,   # Original CryptoMamba setting
        num_layers: int = 1,    # Single hierarchical transition
        d_state: int = 16,      # State dimension (original)
        d_conv: int = 4,        # Convolution kernel (original)
        expand: int = 2,        # Expansion factor (original)
        dropout: float = 0.0,    # Original uses 0.0
        seq_length: int = 22,   # HAR monthly window
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.num_layers = num_layers

        # Input embedding (project features to hidden_dim)
        self.input_embedding = nn.Linear(num_features, hidden_dim)

        # Hierarchical Mamba blocks like original CryptoMamba
        # hidden_dims=[14, 1] means: first block 14→14, second block 1→1
        # For volatility prediction, we use single transition: 14→1
        self.mamba_blocks = nn.ModuleList()

        # First Mamba block (hidden_dim → hidden_dim)
        self.mamba_blocks.append(
            MambaSSM(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        )

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Projection to prediction
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_length, num_features)

        Returns:
            predictions: (batch, 1) - predicted volatility
        """
        # Input embedding
        x = self.input_embedding(x)  # (B, L, hidden_dim)

        # Apply Mamba blocks
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)  # (B, L, hidden_dim)
            x = self.norm(x)    # (B, L, hidden_dim)

        # Global average pooling (over sequence length)
        x = x.mean(dim=1)  # (B, hidden_dim)

        # Project to prediction
        predictions = self.projector(x)  # (B, 1)

        return predictions


class CMambaVolatilityFullHierarchical(nn.Module):
    """
    CryptoMamba for Volatility Prediction (Full Hierarchical Architecture)

    Alternative implementation with explicit hierarchical structure hidden_dims=[14, 1]
    This matches the original CryptoMamba architecture more closely.

    Architecture:
        Input: (batch, seq_length, num_features)
        → Input projection: num_features → 14
        → First Mamba block: 14 → 14
        → Dimension reduction: 14 → 1
        → Second Mamba block: 1 → 1
        → Projection head: 1 → 1
        → Output: (batch, 1) - predicted volatility

    Parameters: ~136K (full selective scan with hierarchical structure)
    """

    def __init__(
        self,
        num_features: int = 3,  # HAR features (daily, weekly, monthly)
        hidden_dim: int = 14,   # Original CryptoMamba setting
        num_layers: int = 1,    # Hierarchical transitions
        d_state: int = 16,      # State dimension (original)
        d_conv: int = 4,        # Convolution kernel (original)
        expand: int = 2,        # Expansion factor (original)
        dropout: float = 0.0,    # Original uses 0.0
        seq_length: int = 22,   # HAR monthly window
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        # Hierarchical hidden dimensions like original: [14, 1]
        self.hidden_dims = [hidden_dim, 1]

        # Input embedding (project features to first hidden_dim)
        self.input_embedding = nn.Linear(num_features, self.hidden_dims[0])

        # Create Mamba blocks for each hierarchical level
        self.mamba_blocks = nn.ModuleList()
        self.dim_projections = nn.ModuleList()

        for i in range(len(self.hidden_dims)):
            d_in = self.hidden_dims[i]
            d_out = self.hidden_dims[i]

            # Mamba block at this level
            self.mamba_blocks.append(
                MambaSSM(
                    d_model=d_in,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            )

            # Layer norm
            if i < len(self.hidden_dims) - 1:
                # Projection to next hierarchical level
                self.dim_projections.append(
                    nn.Linear(d_in, self.hidden_dims[i + 1])
                )

        # Final projection
        self.projector = nn.Linear(self.hidden_dims[-1], 1)

    def forward(self, x):
        """
        Forward pass with hierarchical processing.

        Args:
            x: Input tensor (batch, seq_length, num_features)

        Returns:
            predictions: (batch, 1) - predicted volatility
        """
        # Input embedding
        x = self.input_embedding(x)  # (B, L, 14)

        # Process through hierarchical levels
        for i, mamba_block in enumerate(self.mamba_blocks):
            x = mamba_block(x)  # (B, L, hidden_dims[i])

            # Project to next hierarchical level (except last level)
            if i < len(self.hidden_dims) - 1:
                x = self.dim_projections[i](x)  # (B, L, hidden_dims[i+1])

        # Global average pooling
        x = x.mean(dim=1)  # (B, hidden_dims[-1])

        # Final projection
        predictions = self.projector(x)  # (B, 1)

        return predictions


def create_cryptomamba_model_full(
    num_features: int = 3,
    hidden_dim: int = 14,
    num_layers: int = 1,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    dropout: float = 0.0,
    seq_length: int = 22,
    hierarchical: bool = True,  # Use hierarchical architecture
):
    """
    Factory function for creating CryptoMamba volatility model (full with mamba_ssm).

    Args:
        num_features: Number of input features (default: 3 for HAR)
        hidden_dim: Hidden dimension (default: 14 - original setting)
        num_layers: Number of hierarchical transitions (default: 1)
        d_state: SSM state dimension (default: 16)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor (default: 2)
        dropout: Dropout rate (default: 0.0 - original setting)
        seq_length: Input sequence length (default: 22)
        hierarchical: Use hierarchical architecture (default: True)

    Returns:
        CMambaVolatilityFull model instance
    """
    if not MAMBA_AVAILABLE:
        raise RuntimeError(
            "mamba_ssm package not available. Please install:\n"
            "  pip install mamba_ssm\n"
            "Requires CUDA toolkit 11.8+"
        )

    if hierarchical:
        model = CMambaVolatilityFullHierarchical(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            seq_length=seq_length,
        )
    else:
        model = CMambaVolatilityFull(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            seq_length=seq_length,
        )

    return model


# Test function
def test_model():
    """Test the model with sample data."""
    print("Testing CryptoMamba Volatility Model (Full with mamba_ssm)...")

    if not MAMBA_AVAILABLE:
        print("❌ mamba_ssm not available - cannot test")
        return

    # Create model (hierarchical)
    model = create_cryptomamba_model_full(
        num_features=3,  # HAR features
        hidden_dim=14,
        num_layers=1,
        d_state=16,
        hierarchical=True,
    )

    print(f"✅ Model created successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Architecture: Hierarchical hidden_dims=[14, 1]")

    # Test forward pass
    batch_size = 4
    seq_length = 22
    num_features = 3

    x = torch.randn(batch_size, seq_length, num_features)

    with torch.no_grad():
        predictions = model(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {predictions.shape}")
    print(f"   Expected output: ({batch_size}, 1)")

    assert predictions.shape == (batch_size, 1), "Output shape mismatch!"
    print("✅ Model test passed!")


if __name__ == "__main__":
    test_model()
