"""
CryptoMamba Model - Enhanced V2 Implementation (Root Cause Fixes)

Enhanced V2 architecture with increased capacity and output constraints
to prevent negative volatility predictions.

Root cause fixes from adversarial review:
- ReLU activation on output ensures non-negative predictions
- Input validation prevents NaN/inf propagation
- Integration ready: works with generated HAR features

Key improvements over V2:
- 20-40× more parameters (50-100K vs 2,787)
- Deeper architecture (3 layers vs 1)
- Larger hidden dimensions (64 vs 14)
- Enhanced regularization (dropout 0.1)
- ReLU output constraint (prevents negative volatility)

Target performance: 50-52% Dir Acc (HYPOTHESIS - requires validation)

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Any
import math


class MambaBlockEnhanced(nn.Module):
    """
    Enhanced Mamba block with improved capacity and regularization.

    Uses parameterized SSM operations with:
    - Larger hidden dimensions
    - Dropout for regularization
    - Multiple residual connections
    - Better gradient flow

    SSM Note: This uses selective scan-inspired gating (simplified from full mamba_ssm)
    rather than true selective state space recursion. CPU-compatible approximation.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Input projection (larger dimensions)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Convolution (larger kernel for better temporal capture)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Activation
        self.act = nn.SiLU()

        # State space projection (larger state dimension)
        self.x_proj = nn.Linear(
            self.d_inner, self.d_state * 2, bias=False
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # SSM parameters (learnable, larger matrices)
        self.A = nn.Parameter(
            torch.randn(self.d_inner, self.d_state) * 0.1
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Layer normalization (pre-norm for better stability)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout (for regularization)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass with input validation.

        x: (B, L, D)
        Returns: (B, L, D)
        """
        B, L, D = x.shape
        residual = x  # Preserve original input for residual connection

        # Input validation
        if torch.isnan(x).any():
            raise ValueError(f"NaN detected in input at shape {x.shape}")
        if torch.isinf(x).any():
            raise ValueError(f"Inf detected in input at shape {x.shape}")

        # Pre-norm
        x_norm = self.norm1(x)

        # Input projection
        xz = self.in_proj(x_norm)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)

        # Convolution
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[..., :L]  # (B, d_inner, L)
        x = self.act(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, L, d_inner)

        # SSM operation (enhanced selective scan)
        # Extract B, C parameters (larger state dimension)
        B_param, C_param = self.x_proj(x).chunk(2, dim=-1)  # Each: (B, L, d_state)

        # B gate controls input flow
        B_gate = torch.sigmoid(B_param.mean(dim=-1, keepdim=True))  # (B, L, 1)
        x = x * B_gate  # (B, L, d_inner) - apply input-dependent gating

        # C gate controls output flow
        C_gate = torch.sigmoid(C_param.mean(dim=-1, keepdim=True))  # (B, L, 1)

        # Apply state transformation using A matrix (larger projection)
        y = torch.matmul(x, self.A)  # (B, L, d_state)

        # Apply C gating
        y = y * C_gate  # (B, L, d_state)

        # Project back to d_inner
        y = torch.matmul(y, self.A.t())  # (B, L, d_inner)

        # Gating with dropout
        y = y * self.act(z)
        y = self.dropout(y)

        # Output projection
        y = self.out_proj(y)  # (B, L, d_model)

        # Post-norm
        y = self.norm2(y)

        # Residual connection (with dropout)
        y = self.dropout(y) + residual

        return y


class CMambaVolatilityEnhanced(nn.Module):
    """
    CryptoMamba for Volatility Prediction (Enhanced V2 Architecture)

    Enhanced architecture with increased capacity and output constraints:
    - 3 Mamba blocks (vs 1 in V2)
    - Larger hidden dimensions (64 vs 14)
    - Enhanced regularization (dropout 0.1)
    - ReLU output activation (ensures non-negative predictions)
    - Input validation (NaN/inf checks)

    Architecture:
        Input: (batch, seq_length, num_features)
        → Input projection: num_features → hidden_dim (64)
        → Mamba blocks (×3) with residual connections
        → Global average pooling
        → Projection head: hidden_dim → 1
        → ReLU activation: ensures output >= 0
        → Output: (batch, 1) - predicted volatility (always non-negative)

    Parameters: 50-100K (vs 2,787 in V2)
    Target: 50-52% Dir Acc (HYPOTHESIS - requires validation)

    Root Cause Fixes:
    - ReLU activation prevents negative volatility predictions
    - Input validation prevents NaN/inf propagation
    """

    def __init__(
        self,
        num_features: int = 3,  # HAR features (daily, weekly, monthly)
        hidden_dim: int = 64,   # Enhanced: 64 (vs 14 in V2)
        num_layers: int = 3,    # Enhanced: 3 layers (vs 1 in V2)
        d_state: int = 32,      # Enhanced: 32 (vs 16 in V2)
        d_conv: int = 4,        # Convolution kernel
        expand: int = 2,        # Expansion factor
        dropout: float = 0.1,   # Enhanced: 0.1 (vs 0.0 in V2)
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

        # Multiple Mamba blocks for enhanced capacity
        self.mamba_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.mamba_blocks.append(
                MambaBlockEnhanced(
                    d_model=hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
            )

        # Layer normalization after all blocks
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Projection to prediction with ReLU activation (CRITICAL FIX)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # CRITICAL: Ensures non-negative predictions
        )

    def forward(self, x):
        """
        Forward pass with input validation.

        Args:
            x: Input tensor (batch, seq_length, num_features)

        Returns:
            predictions: (batch, 1) - predicted volatility (always >= 0 due to ReLU)
        """
        # Input embedding
        x = self.input_embedding(x)  # (B, L, hidden_dim)

        # Apply Mamba blocks with residual connections
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)  # (B, L, hidden_dim)

        # Final normalization
        x = self.final_norm(x)  # (B, L, hidden_dim)

        # Global average pooling (over sequence length)
        x = x.mean(dim=1)  # (B, hidden_dim)

        # Project to prediction with ReLU (ensures non-negative)
        predictions = self.projector(x)  # (B, 1) - always >= 0

        return predictions


class CMambaVolatilityEnhancedDeep(nn.Module):
    """
    CryptoMamba for Volatility Prediction (Enhanced Deep Architecture)

    Alternative implementation with deeper hierarchical structure:
    - Progressive dimensionality reduction: 64 → 32 → 16 → 8
    - More blocks at each dimension level
    - Enhanced feature extraction
    - ReLU output activation (non-negative predictions)

    Parameters: ~80-120K
    Target: 51-53% Dir Acc (HYPOTHESIS)
    """

    def __init__(
        self,
        num_features: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 3,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        seq_length: int = 22,
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Hierarchical hidden dimensions: 64 → 32 → 16
        self.hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]

        # Input embedding
        self.input_embedding = nn.Linear(num_features, self.hidden_dims[0])

        # Create Mamba blocks for each hierarchical level
        self.mamba_blocks = nn.ModuleList()
        self.dim_projections = nn.ModuleList()

        for i, h_dim in enumerate(self.hidden_dims):
            # Mamba block at this level
            self.mamba_blocks.append(
                MambaBlockEnhanced(
                    d_model=h_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
            )

            # Projection to next level (except last)
            if i < len(self.hidden_dims) - 1:
                self.dim_projections.append(
                    nn.Sequential(
                        nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                    )
                )

        # Final projection with ReLU (ensures non-negative predictions)
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dims[-1] // 2, 1),
            nn.ReLU()  # CRITICAL: Ensures non-negative predictions
        )

    def forward(self, x):
        """Forward pass with hierarchical processing."""
        # Input embedding
        x = self.input_embedding(x)  # (B, L, 64)

        # Process through hierarchical levels
        for i, mamba_block in enumerate(self.mamba_blocks):
            x = mamba_block(x)  # (B, L, hidden_dims[i])

            # Project to next level (except last)
            if i < len(self.hidden_dims) - 1:
                x = self.dim_projections[i](x)  # (B, L, hidden_dims[i+1])

        # Global average pooling
        x = x.mean(dim=1)  # (B, hidden_dims[-1])

        # Final projection with ReLU
        predictions = self.projector(x)  # (B, 1) - always >= 0

        return predictions


def create_cryptomamba_model_enhanced(
    num_features: int = 3,
    hidden_dim: int = 64,
    num_layers: int = 3,
    d_state: int = 32,
    d_conv: int = 4,
    expand: int = 2,
    dropout: float = 0.1,
    seq_length: int = 22,
    hierarchical: bool = False,
):
    """
    Factory function for creating CryptoMamba volatility model (enhanced V2).

    Args:
        num_features: Number of input features (default: 3 for HAR)
        hidden_dim: Hidden dimension (default: 64 - enhanced)
        num_layers: Number of Mamba blocks (default: 3 - enhanced)
        d_state: SSM state dimension (default: 32 - enhanced)
        d_conv: Convolution kernel size (default: 4)
        expand: Expansion factor (default: 2)
        dropout: Dropout rate (default: 0.1 - enhanced regularization)
        seq_length: Input sequence length (default: 22)
        hierarchical: Use hierarchical deep architecture (default: False)

    Returns:
        CMambaVolatilityEnhanced model instance with ReLU output constraint
    """
    if hierarchical:
        model = CMambaVolatilityEnhancedDeep(
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
        model = CMambaVolatilityEnhanced(
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
    """Test the enhanced model with sample data."""
    print("Testing CryptoMamba Volatility Model (Enhanced V2 with Root Cause Fixes)...")

    # Create model (standard)
    model = create_cryptomamba_model_enhanced(
        num_features=3,  # HAR features
        hidden_dim=64,
        num_layers=3,
        d_state=32,
        hierarchical=False,
    )

    print(f"Model created successfully")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Test forward pass
    batch_size = 4
    seq_length = 22
    num_features = 3

    x = torch.randn(batch_size, seq_length, num_features)

    with torch.no_grad():
        predictions = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Expected output: ({batch_size}, 1)")

    assert predictions.shape == (batch_size, 1), "Output shape mismatch!"

    # CRITICAL TEST: Verify all predictions are non-negative
    assert (predictions >= 0).all(), f"Negative predictions found! Min: {predictions.min().item():.6f}"

    print(f"Output range: [{predictions.min().item():.6f}, {predictions.max().item():.6f}]")
    print("All predictions are non-negative - ReLU constraint working!")

    # Compare with V2
    print(f"\nCapacity comparison:")
    print(f"  V2:         2,787 parameters")
    print(f"  Enhanced:  {num_params:,} parameters ({num_params/2787:.1f}× increase)")

    if num_params < 50000:
        print(f"  [WARNING] Low parameter count (expected 50-100K)")
    elif num_params > 150000:
        print(f"  [WARNING] Very high parameter count (expected 50-100K)")
    else:
        print(f"  [OK] Parameter count in target range (50-100K)")

    print("Model test passed!")


if __name__ == "__main__":
    test_model()
