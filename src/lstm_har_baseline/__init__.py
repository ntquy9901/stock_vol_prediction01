"""
LSTM-HAR Baseline Package

LSTM model with Heterogeneous Autoregressive (HAR) features for volatility prediction.

Features:
    - HAR features: daily, weekly, monthly volatility averages
    - LSTM architecture: 2 layers, 64 hidden units
    - Dropout regularization: 0.2
    - Input: (batch, seq_len, 3) - 3 HAR features
    - Output: (batch, 1) - Volatility prediction

Author: Stock Volatility Prediction Team
Date: 2026-06-18
"""

from .model import HARVolatilityLSTM
from .dataset import HARVolatilityDataset
from .features import create_har_features, create_har_dataset_with_targets

__all__ = [
    'HARVolatilityLSTM',
    'HARVolatilityDataset',
    'create_har_features',
    'create_har_dataset_with_targets'
]