"""
Stock Volatility Prediction System

This package implements a multi-horizon volatility forecasting system for VN30 stocks
using HAR (Heterogeneous Autoregressive) methodology with Parkinson volatility estimator.

Modules:
    data_processing: Volatility calculation and data validation
    feature_engineering: HAR feature creation and target generation
    models: Model implementations (HAR-R, LSTM, GNN)
    training: Training pipeline and evaluation metrics
    api: FastAPI prediction service
"""
