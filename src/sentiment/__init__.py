"""
VN30 Sentiment Analysis Module

This module provides sentiment analysis capabilities for VN30 stocks,
integrating with the existing volatility prediction pipeline.

Components:
- data_collection: News and social media data collection
- models: Sentiment analysis models (FinBERT)
- processing: Data processing and feature engineering
- utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "QUY"
__date__ = "2026-06-27"

from .models.finbert_sentiment import FinBERTSentiment
from .processing.har_sentiment_features import HARSentimentFeatures

__all__ = [
    "FinBERTSentiment",
    "HARSentimentFeatures",
]
