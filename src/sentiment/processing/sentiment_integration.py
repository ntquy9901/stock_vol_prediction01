"""
Sentiment Feature Integration for LSTM-GNN

Creates 3 sentiment features aligned with HAR dataset:
1. sentiment_score_3d: Weighted average sentiment over 3-day window
2. sentiment_confidence: Mean confidence score
3. news_count_norm: Number of articles (normalized)

Output format aligned with data/processed/{ticker}_processed.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class SentimentIntegration:
    """
    Integrate sentiment features into LSTM-GNN training pipeline.

    Usage:
        integrator = SentimentIntegration()
        integrator.create_all_sentiment_features()
    """

    def __init__(self, sentiment_dir: str = "data/processed/vn30_sentiment/daily",
                 output_dir: str = "data/processed/sentiment_features",
                 har_dir: str = "data/processed"):
        """
        Initialize sentiment integrator.

        Args:
            sentiment_dir: Directory containing vn30_sentiment_combined.csv
            output_dir: Output directory for sentiment features
            har_dir: Directory containing HAR volatility data (for date alignment)
        """
        self.sentiment_dir = Path(sentiment_dir)
        self.output_dir = Path(output_dir)
        self.har_dir = Path(har_dir)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[SentimentIntegration] Initialized:")
        logger.info(f"  - Sentiment dir: {self.sentiment_dir}")
        logger.info(f"  - Output dir: {self.output_dir}")
        logger.info(f"  - HAR dir: {self.har_dir}")

    def create_all_sentiment_features(self) -> Dict[str, pd.DataFrame]:
        """
        Create sentiment features for all VN30 stocks.

        Returns:
            Dict mapping ticker -> sentiment_features DataFrame
        """
        # Load combined sentiment data
        sentiment_file = self.sentiment_dir / "vn30_sentiment_combined.csv"

        if not sentiment_file.exists():
            logger.error(f"Sentiment file not found: {sentiment_file}")
            return {}

        logger.info(f"[SentimentIntegration] Loading sentiment data from {sentiment_file}")
        df = pd.read_csv(sentiment_file)
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"[SentimentIntegration] Loaded {len(df)} sentiment records")

        # Get unique tickers
        tickers = sorted(df['ticker'].unique())
        logger.info(f"[SentimentIntegration] Processing {len(tickers)} tickers")

        all_features = {}

        for ticker in tickers:
            logger.info(f"[SentimentIntegration] Processing {ticker}...")

            # Filter for this ticker
            ticker_df = df[df['ticker'] == ticker].copy()

            # Create features
            features_df = self._create_stock_sentiment_features(ticker_df, ticker)

            if features_df is not None:
                all_features[ticker] = features_df

                # Save to file
                output_file = self.output_dir / f"{ticker}_sentiment_features.csv"
                features_df.to_csv(output_file, index=False)
                logger.info(f"  → Saved {len(features_df)} rows to {output_file}")

        logger.info(f"[SentimentIntegration] Complete: {len(all_features)} stocks processed")

        return all_features

    def _create_stock_sentiment_features(self, ticker_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create sentiment features for a single stock.

        Features created:
        - sentiment_score_3d: Weighted average sentiment (3-day window)
        - sentiment_confidence: Mean confidence score
        - news_count_norm: Number of articles / max_count

        Args:
            ticker_df: DataFrame with sentiment data for this ticker
            ticker: Stock symbol

        Returns:
            DataFrame with columns: [date, sentiment_score_3d, sentiment_confidence, news_count_norm]
        """
        if ticker_df.empty:
            logger.warning(f"  No sentiment data for {ticker}")
            return None

        # Group by date and aggregate
        daily_features = ticker_df.groupby('date').agg({
            'sentiment_score': 'mean',  # Average sentiment across all articles
            'article_id': 'count'       # Number of articles
        }).reset_index()

        # Calculate confidence from sentiment scores
        if 'positive_score' in ticker_df.columns and 'negative_score' in ticker_df.columns:
            # Confidence = 1 - neutral_score (high confidence when model is decisive)
            daily_confidence = ticker_df.groupby('date').apply(
                lambda x: (1 - x['neutral_score']).mean()
            ).reset_index()
            daily_confidence.columns = ['date', 'confidence_mean']
            daily_features = daily_features.merge(daily_confidence, on='date', how='left')
        else:
            daily_features['confidence_mean'] = 0.8  # Default confidence

        daily_features.columns = ['date', 'sentiment_score_mean', 'news_count', 'confidence_mean']

        # Sort by date
        daily_features = daily_features.sort_values('date').reset_index(drop=True)

        # Create 3-day rolling average sentiment
        daily_features['sentiment_score_3d'] = daily_features['sentiment_score_mean'].rolling(
            window=3, min_periods=1
        ).mean()

        # Normalize news count (0-1 range)
        max_count = daily_features['news_count'].max()
        if max_count > 0:
            daily_features['news_count_norm'] = daily_features['news_count'] / max_count
        else:
            daily_features['news_count_norm'] = 0.0

        # Confidence already calculated above, ensure it exists
        if 'confidence_mean' not in daily_features.columns:
            daily_features['confidence_mean'] = 0.8

        # Select final columns
        final_features = daily_features[[
            'date',
            'sentiment_score_3d',
            'confidence_mean',
            'news_count_norm'
        ]].copy()

        final_features.columns = ['date', 'sentiment_score_3d', 'sentiment_confidence', 'news_count_norm']

        # Fill NaN values
        final_features = final_features.fillna({
            'sentiment_score_3d': 0.0,
            'sentiment_confidence': 0.5,
            'news_count_norm': 0.0
        })

        return final_features

    def align_with_har_data(self, sentiment_features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align sentiment features with HAR dataset dates.

        For each stock:
        1. Load HAR data (dates: 2006-2026)
        2. Align sentiment dates with HAR dates
        3. Fill missing sentiment with 0 (no news = neutral)

        Args:
            sentiment_features: Dict of ticker -> sentiment features DataFrame

        Returns:
            Dict of ticker -> aligned features
        """
        aligned_features = {}

        for ticker, sent_df in sentiment_features.items():
            har_file = self.har_dir / f"{ticker}_processed.csv"

            if not har_file.exists():
                logger.warning(f"  HAR file not found: {har_file}")
                continue

            # Load HAR dates
            har_df = pd.read_csv(har_file)
            har_df['date'] = pd.to_datetime(har_df['date'])

            # Create date range from HAR data
            date_range = har_df['date'].unique()

            # Align sentiment to HAR dates
            sent_df['date'] = pd.to_datetime(sent_df['date'])

            # Create DataFrame with all HAR dates
            aligned_df = pd.DataFrame({'date': date_range})

            # Merge with sentiment features (left join to keep all HAR dates)
            aligned_df = aligned_df.merge(sent_df, on='date', how='left')

            # Fill missing values (no news = neutral sentiment)
            aligned_df = aligned_df.fillna({
                'sentiment_score_3d': 0.0,
                'sentiment_confidence': 0.0,
                'news_count_norm': 0.0
            })

            aligned_features[ticker] = aligned_df

            logger.info(f"  → Aligned {ticker}: {len(aligned_df)} dates (HAR range)")

        return aligned_features


def main():
    """Main function to create and save all sentiment features."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("=" * 60)
    logger.info("Starting Sentiment Integration for LSTM-GNN Phase 1")
    logger.info("=" * 60)

    # Create integrator
    integrator = SentimentIntegration()

    # Create sentiment features
    logger.info("\n[Step 1] Creating sentiment features...")
    all_features = integrator.create_all_sentiment_features()

    # Align with HAR data
    logger.info("\n[Step 2] Aligning with HAR dataset dates...")
    aligned_features = integrator.align_with_har_data(all_features)

    # Save aligned features
    logger.info("\n[Step 3] Saving aligned features...")
    for ticker, features_df in aligned_features.items():
        output_file = integrator.output_dir / f"{ticker}_sentiment_aligned.csv"
        features_df.to_csv(output_file, index=False)
        logger.info(f"  → Saved {ticker}: {len(features_df)} rows")

    logger.info("\n" + "=" * 60)
    logger.info("Sentiment Integration Complete!")
    logger.info(f"Output directory: {integrator.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
