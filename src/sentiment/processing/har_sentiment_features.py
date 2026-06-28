"""
HAR-like Sentiment Feature Engineering

Implementation of Heterogeneous Autoregressive (HAR) features
adapted for sentiment analysis, aligning with existing volatility
prediction methodology.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class HARSentimentFeatures:
    """
    Create HAR-like sentiment features for volatility prediction.

    This class implements sentiment feature engineering following the
    HAR methodology used in volatility prediction, enabling seamless
    integration with existing models.
    """

    def __init__(self, daily_windows: List[int] = None):
        """
        Initialize HAR sentiment feature engineer.

        Args:
            daily_windows: List of day windows for HAR features
                         Default: [1, 5, 22] for Daily, Weekly, Monthly
        """
        if daily_windows is None:
            self.daily_windows = [1, 5, 22]  # Daily, Weekly, Monthly
        else:
            self.daily_windows = daily_windows

        logger.info(f"HAR Sentiment Features initialized with windows: {self.daily_windows}")

    def create_har_features(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all HAR-like sentiment features.

        Args:
            sentiment_df: DataFrame with columns ['date', 'ticker', 'sentiment_score']

        Returns:
            DataFrame with HAR sentiment features added
        """
        if sentiment_df.empty:
            logger.warning("Empty DataFrame provided")
            return sentiment_df

        # Make copy to avoid modifying original
        features_df = sentiment_df.copy()

        # Ensure date is datetime and sorted
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df = features_df.sort_values('date')

        # Group by ticker if multiple tickers present
        if 'ticker' in features_df.columns:
            features_df = features_df.groupby('ticker', group_keys=False).apply(
                lambda x: self._create_features_for_group(x)
            ).reset_index(drop=True)

        return features_df

    def _create_features_for_group(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for single group (ticker)"""
        # HAR sentiment windows
        for window in self.daily_windows:
            df[f'sent_{window}d'] = (
                df['sentiment_score']
                .rolling(window=window, min_periods=1)
                .mean()
            )

        # Moving averages
        df['sent_ma5'] = (
            df['sentiment_score']
            .rolling(window=5, min_periods=1)
            .mean()
        )

        df['sent_ma22'] = (
            df['sentiment_score']
            .rolling(window=22, min_periods=1)
            .mean()
        )

        # Sentiment volatility (standard deviation)
        df['sentiment_volatility'] = (
            df['sentiment_score']
            .rolling(window=22, min_periods=5)
            .std()
        )

        # Sentiment momentum (rate of change)
        df['sentiment_momentum'] = (
            df['sentiment_score']
            .diff(periods=1)
        )

        # Trend indicators
        df['positive_trend'] = (
            df['sent_ma5'] > df['sent_ma22']
        )

        df['negative_trend'] = (
            df['sent_ma5'] < df['sent_ma22']
        )

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names created by this class"""
        features = []

        # HAR windows
        for window in self.daily_windows:
            features.append(f'sent_{window}d')

        # Moving averages
        features.extend(['sent_ma5', 'sent_ma22'])

        # Volatility and momentum
        features.extend(['sentiment_volatility', 'sentiment_momentum'])

        # Trend indicators
        features.extend(['positive_trend', 'negative_trend'])

        return features

    def validate_features(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that all features are present and valid.

        Returns:
            Dict mapping feature names to validity status
        """
        validation_results = {}
        feature_names = self.get_feature_names()

        for feature in feature_names:
            if feature not in df.columns:
                validation_results[feature] = False
                logger.warning(f"Missing feature: {feature}")
            else:
                # Check for null values
                null_count = df[feature].isnull().sum()
                validation_results[feature] = (null_count == 0)

                if null_count > 0:
                    logger.warning(f"Feature {feature} has {null_count} null values")

        return validation_results


# Example usage
if __name__ == "__main__":
    # Create sample sentiment data
    dates = pd.date_range('2026-06-01', '2026-06-30')
    sample_data = {
        'date': dates,
        'ticker': 'ACB',
        'sentiment_score': np.random.uniform(-0.8, 0.8, len(dates))
    }

    df = pd.DataFrame(sample_data)

    print("Original data:")
    print(df.head())

    # Create HAR features
    har_features = HARSentimentFeatures()
    enhanced_df = har_features.create_har_features(df)

    print("\nEnhanced data with HAR features:")
    print(enhanced_df[['date', 'sentiment_score', 'sent_1d', 'sent_5d', 'sent_22d']].head(10))

    # Validate features
    validation = har_features.validate_features(enhanced_df)
    print("\nFeature validation:", validation)
