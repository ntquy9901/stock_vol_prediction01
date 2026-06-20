"""
Enhanced LSTM-HAR Model with Raw + HAR Features

This module contains the LSTM architecture that uses both
raw Parkinson volatility AND HAR features as input.

Architecture (IMPROVED 2026-06-19):
    Input: (batch, seq_len, 3) - [raw, weekly, monthly]
    LSTM: 3 layers, 128 hidden units (INCREASED CAPACITY)
    Dropout: 0.1 (REDUCED REGULARIZATION)
    Output: (batch, 1) - Volatility prediction

Performance Fix: PRIORITY 1 & 2
- Expected Dir Acc improvement: +8-15% (from 48% to >55%)
- Expected RMSE improvement: +10% (from 0.00055 to <0.00050)

Author: Stock Volatility Prediction Team
Date: 2026-06-19
Version: 2.0 (Performance Optimized)
"""

import torch
import torch.nn as nn


class EnhancedHARVolatilityLSTM(nn.Module):
    """
    Enhanced LSTM Model with Raw Volatility + HAR Features.

    Architecture (IMPROVED 2026-06-19):
        Input: (batch_size, seq_length, 3) - [raw, weekly, monthly]
        LSTM: 3 layers, hidden_size=128 (INCREASED CAPACITY)
        Dropout: 0.1 (REDUCED REGULARIZATION)
        Output: (batch_size, 1) - Volatility prediction

    Args:
        hidden_size: Number of LSTM hidden units (default: 128, was: 64)
        num_layers: Number of LSTM layers (default: 3, was: 2)
        dropout: Dropout rate (default: 0.1, was: 0.2)
    """

    def __init__(self, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super(EnhancedHARVolatilityLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=3,  # 3 features: raw, weekly, monthly (NO daily - redundant)
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output layer (no activation - allow negative predictions during training)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_length, 3)

        Returns:
            output: Volatility predictions (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Extract last timestep
        last_output = lstm_out[:, -1, :]

        # Output layer
        output = self.fc(last_output)

        return output


if __name__ == "__main__":
    # Test the model
    print("Testing Enhanced HAR LSTM Model (IMPROVED)...")

    model = EnhancedHARVolatilityLSTM(hidden_size=128, num_layers=3, dropout=0.1)

    # Create dummy input
    batch_size = 32
    seq_length = 22
    n_features = 3  # 3 features: raw, weekly, monthly

    dummy_input = torch.randn(batch_size, seq_length, n_features)

    print(f"\nInput shape: {dummy_input.shape}")

    # Forward pass
    output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Check output statistics
    print(f"\nOutput statistics:")
    print(f"  Mean: {output.mean().item():.6f}")
    print(f"  Std: {output.std().item():.6f}")
    print(f"  Min: {output.min().item():.6f}")
    print(f"  Max: {output.max().item():.6f}")

    print("\n[OK] Model test successful!")
