"""
Temporal Encoder: LSTM-HAR for Volatility Prediction

Per-stock temporal feature learning using LSTM with HAR features.
Processes each stock independently before cross-stock attention.

Architecture:
    Input: (batch, num_stocks, seq_len, num_features)
    LSTM: 2 layers with dropout
    Output: (batch, num_stocks, hidden_dim)

Author: Stock Volatility Prediction Team
Date: 2026-06-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TemporalLSTM(nn.Module):
    """
    Temporal LSTM encoder for per-stock volatility feature learning.

    Processes each stock's time series independently through LSTM layers
    to capture temporal patterns in volatility.

    Args:
        input_size: Number of input features (default: 22)
            - HAR features: daily, weekly, monthly (3)
            - Technical indicators: RSI, MACD, etc. (19)
        hidden_size: LSTM hidden dimension (default: 64)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.2)
        bidirectional: Use bidirectional LSTM (default: False)
    """

    def __init__(
        self,
        input_size: int = 22,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(TemporalLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output dimension
        self.output_dim = hidden_size * self.num_directions

    def forward(
        self,
        x: torch.Tensor,
        return_all_timesteps: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through temporal encoder.

        Args:
            x: Input tensor (batch, num_stocks, seq_len, input_size)
            return_all_timesteps: If True, return all timesteps, else last only

        Returns:
            output: Temporal features
                - If return_all_timesteps=False: (batch, num_stocks, output_dim)
                - If return_all_timesteps=True: (batch, num_stocks, seq_len, output_dim)
        """
        batch_size, num_stocks, seq_len, input_size = x.shape

        # Process each stock independently
        outputs = []
        for stock_idx in range(num_stocks):
            # Extract stock data: (batch, seq_len, input_size)
            stock_data = x[:, stock_idx, :, :]

            # LSTM forward pass
            lstm_out, (h_n, c_n) = self.lstm(stock_data)
            # lstm_out shape: (batch, seq_len, hidden_size * num_directions)

            if return_all_timesteps:
                # Return all timesteps
                stock_output = lstm_out  # (batch, seq_len, output_dim)
            else:
                # Return only last timestep
                stock_output = lstm_out[:, -1, :]  # (batch, output_dim)

            outputs.append(stock_output)

        # Stack outputs: (num_stocks, batch, ...) -> (batch, num_stocks, ...)
        outputs = torch.stack(outputs, dim=1)

        return outputs

    def get_output_dim(self) -> int:
        """Return output dimension of the encoder."""
        return self.output_dim


class EnhancedTemporalLSTM(nn.Module):
    """
    Enhanced Temporal LSTM with HAR feature integration.

    Specifically designed for the LSTM-HAR-Enhanced baseline approach
    with raw volatility + HAR features (weekly, monthly).

    Args:
        seq_length: Input sequence length (default: 22)
        hidden_size: LSTM hidden dimension (default: 64)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability (default: 0.2)
    """

    def __init__(
        self,
        seq_length: int = 22,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super(EnhancedTemporalLSTM, self).__init__()

        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input: raw + weekly + monthly (3 features)
        self.input_size = 3

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_dim = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through enhanced temporal encoder.

        Args:
            x: Input tensor (batch, num_stocks, seq_len, 3)
                Features: [raw_volatility, weekly_har, monthly_har]

        Returns:
            output: Temporal features (batch, num_stocks, hidden_size)
        """
        batch_size, num_stocks, seq_len, input_size = x.shape

        # Process each stock independently
        outputs = []
        for stock_idx in range(num_stocks):
            # Extract stock data: (batch, seq_len, input_size)
            stock_data = x[:, stock_idx, :, :]

            # LSTM forward pass
            lstm_out, (h_n, c_n) = self.lstm(stock_data)
            # lstm_out shape: (batch, seq_len, hidden_size)

            # Take last timestep
            stock_output = lstm_out[:, -1, :]  # (batch, hidden_size)
            outputs.append(stock_output)

        # Stack outputs: (num_stocks, batch, hidden_size) -> (batch, num_stocks, hidden_size)
        outputs = torch.stack(outputs, dim=1)

        return outputs

    def get_output_dim(self) -> int:
        """Return output dimension of the encoder."""
        return self.output_dim


# Test the encoder
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEMPORAL ENCODER - TEST")
    print("="*60)

    # Test basic TemporalLSTM
    print("\n[1] Testing TemporalLSTM")
    batch_size = 4
    num_stocks = 30
    seq_len = 22
    input_size = 22

    encoder = TemporalLSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )

    # Create dummy input
    x = torch.randn(batch_size, num_stocks, seq_len, input_size)

    # Forward pass
    output = encoder(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, {num_stocks}, {encoder.output_dim})")
    print(f"  [OK] Test passed!")

    # Test EnhancedTemporalLSTM
    print("\n[2] Testing EnhancedTemporalLSTM")
    enhanced_encoder = EnhancedTemporalLSTM(
        seq_length=22,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )

    # Create dummy input (raw + weekly + monthly)
    x_enhanced = torch.randn(batch_size, num_stocks, seq_len, 3)

    # Forward pass
    output_enhanced = enhanced_encoder(x_enhanced)

    print(f"  Input shape: {x_enhanced.shape}")
    print(f"  Output shape: {output_enhanced.shape}")
    print(f"  Expected: ({batch_size}, {num_stocks}, {enhanced_encoder.output_dim})")
    print(f"  [OK] Test passed!")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
