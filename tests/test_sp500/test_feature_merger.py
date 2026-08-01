"""Tests for feature merger."""
import os
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Bootstrap path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)


@pytest.fixture
def sample_processed_data():
    """Sample HAR processed data."""
    dates = pd.date_range("2011-01-03", periods=100, freq="B")
    return pd.DataFrame({
        "date": dates,
        "parkinson_volatility": np.random.uniform(0.001, 0.05, 100),
    })


@pytest.fixture
def sample_market_data():
    """Sample market data."""
    dates = pd.date_range("2011-01-03", periods=100, freq="B")
    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Close": np.random.uniform(15, 30, 100),
    })


@pytest.fixture
def sample_sentiment_data():
    """Sample sentiment data."""
    dates = pd.date_range("2011-01-03", periods=50, freq="B")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "sentiment_score": np.random.uniform(-1, 1, 50),
        "sentiment_confidence": np.random.uniform(0.5, 1, 50),
        "news_count": np.random.randint(1, 10, 50),
    })


class TestFeatureMerger:
    """Test feature merger functions."""

    def test_merge_har_only(self, sample_processed_data, tmp_path):
        """Merge with HAR-only feature set should produce 3 features + target."""
        from src.common.feature_merger import merge_features

        # Write sample data
        processed_dir = str(tmp_path / "processed")
        os.makedirs(processed_dir)
        sample_processed_data.to_csv(os.path.join(processed_dir, "AAPL_processed.csv"), index=False)

        output_dir = str(tmp_path / "enhanced")
        result = merge_features(
            ticker="AAPL",
            processed_dir=processed_dir,
            market_data_dir=str(tmp_path / "market"),
            sentiment_dir=str(tmp_path / "sentiment"),
            output_dir=output_dir,
            feature_set="har",
        )

        assert os.path.isfile(result)
        df = pd.read_csv(result, parse_dates=["Date"], index_col="Date")
        assert "target_5d" in df.columns
        assert "har_daily_vol" in df.columns

    def test_merge_full_features(self, sample_processed_data, sample_market_data, sample_sentiment_data, tmp_path):
        """Merge with full feature set should produce 9 features + target."""
        from src.common.feature_merger import merge_features

        # Write sample data
        processed_dir = str(tmp_path / "processed")
        market_dir = str(tmp_path / "market_data/sp500")
        sentiment_dir = str(tmp_path / "sentiment")
        output_dir = str(tmp_path / "enhanced")

        os.makedirs(processed_dir)
        os.makedirs(market_dir)
        os.makedirs(sentiment_dir)
        os.makedirs(output_dir)

        sample_processed_data.to_csv(os.path.join(processed_dir, "AAPL_processed.csv"), index=False)
        sample_market_data.to_csv(os.path.join(market_dir, "vix.csv"), index=False)
        sample_market_data.to_csv(os.path.join(market_dir, "treasury_10y.csv"), index=False)
        sample_market_data.to_csv(os.path.join(market_dir, "sp500_index.csv"), index=False)
        sample_sentiment_data.to_csv(os.path.join(sentiment_dir, "AAPL_sentiment.csv"), index=False)

        result = merge_features(
            ticker="AAPL",
            processed_dir=processed_dir,
            market_data_dir=market_dir,
            sentiment_dir=sentiment_dir,
            output_dir=output_dir,
            feature_set="full",
        )

        assert os.path.isfile(result)
        df = pd.read_csv(result, parse_dates=["Date"], index_col="Date")
        assert "target_5d" in df.columns
        assert "vix" in df.columns
        assert "sentiment_score" in df.columns
