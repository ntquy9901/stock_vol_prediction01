"""
LSTM-GAT Hybrid for Volatility Prediction

Next-generation architecture combining:
- LSTM encoder: Temporal feature learning for each stock
- Graph Attention Network: Cross-stock relationship modeling
- Dynamic graph construction: Correlation + volatility spillover
- Target: RMSE < 0.15, Dir Acc > 75% (vs LSTM-HAR: 0.18, 67.90%)
"""

from .config import LSTMGATConfig

__all__ = ['LSTMGATConfig']
