"""
Enhanced LSTM-HAR Baseline Package

This package contains the enhanced LSTM-HAR model that uses both
raw Parkinson volatility and HAR features (weekly, monthly) for
5-day ahead volatility forecasting.

Components:
    - dataset_enhanced: Enhanced dataset with raw + HAR features
    - model_enhanced: Enhanced LSTM model architecture
    - train_enhanced: Training script for enhanced model

Features: [raw parkinson, har weekly, har monthly] (3 features, no redundancy)

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

from .dataset_enhanced import EnhancedHARDataset
from .model_enhanced import EnhancedHARVolatilityLSTM

__all__ = ['EnhancedHARDataset', 'EnhancedHARVolatilityLSTM']
