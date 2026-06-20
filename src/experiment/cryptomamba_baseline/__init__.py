"""
CryptoMamba Baseline for Volatility Prediction

Phase 1 Proof-of-Concept: Adapt CryptoMamba architecture for VN30 stock volatility forecasting
Based on: https://github.com/MShahabSepehri/CryptoMamba (IEEE ICBC 2025)

Key adaptations:
- Target: Price → Parkinson volatility
- Sequence length: 14 → 22 days (HAR monthly)
- Features: OHLCV + HAR features
- Evaluation: Dir Acc, RMSE, QLIKE (project metrics)

Author: Stock Volatility Prediction Team
Date: 2026-06-19
"""

from .model import CryptoMambaVolatility
from .dataset import CryptoMambaDataset

__all__ = ['CryptoMambaVolatility', 'CryptoMambaDataset']
