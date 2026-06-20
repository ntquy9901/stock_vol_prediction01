"""
Simple 1-Layer LSTM Model for Volatility Prediction

This module contains the Simple LSTM architecture designed for
Parkinson volatility prediction using the pool approach.

Architecture:
    Input: (batch, seq_len, 1) - Parkinson volatility
    LSTM: 1 layer, 32 hidden units
    Output: (batch, 1) - Volatility prediction

Author: Stock Volatility Prediction Team
Date: 2026-06-17
"""

import torch
import torch.nn as nn


class SimpleVolatilityLSTM(nn.Module):
    """
    Simple 1-Layer LSTM for Volatility Prediction.

    Architecture:
        Input: (batch, seq_len, 1) - Parkinson volatility
        LSTM: 1 layer, hidden_size=32
        Output: (batch, 1) - Volatility prediction

    Args:
        hidden_size: Number of LSTM hidden units
    """

    def __init__(self, hidden_size: int = 128):
        super(SimpleVolatilityLSTM, self).__init__()

        self.hidden_size = hidden_size

        # Single LSTM layer
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Output layer (no activation - allow negative predictions during training)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_length, 1)

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
