"""
TimesNet Baseline for Volatility Prediction

Implements TimesNet (ICLR 2023) architecture adapted for cryptocurrency volatility forecasting.
Uses HAR features (daily, weekly, monthly) with temporal embeddings for multi-scale analysis.
"""

from .config import TimesNetConfig

__all__ = ['TimesNetConfig']
