"""
Feature Merger for Enhanced Volatility Prediction

Merges HAR features, market indicators, and sentiment scores
into a single enhanced feature set for model training.

Author: Stock Volatility Prediction Team
Date: 2026-08-01
"""

import os
import sys
import pandas as pd

# Bootstrap path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


def merge_features(
    ticker: str,
    processed_dir: str = None,
    market_data_dir: str = None,
    sentiment_dir: str = None,
    output_dir: str = None,
    feature_set: str = "full",
    horizon: int = 5,
):
    """
    Merge HAR, market, and sentiment features for a single ticker.

    Args:
        ticker: Stock ticker symbol
        processed_dir: Directory with HAR processed CSVs
        market_data_dir: Directory with market indicator CSVs
        sentiment_dir: Directory with sentiment CSVs
        output_dir: Directory to save enhanced CSV
        feature_set: "har" (3), "har_market" (6), or "full" (9)
        horizon: forecast horizon in trading days ahead (default 5). Target column
            is named f"target_{horizon}d".

    Returns:
        Path to enhanced CSV file
    """
    if processed_dir is None:
        processed_dir = os.path.join(project_root, "data", "processed_sp500")
    if market_data_dir is None:
        market_data_dir = os.path.join(project_root, "data", "raw", "market_data", "sp500")
    if sentiment_dir is None:
        sentiment_dir = os.path.join(project_root, "data", "sentiment", "sp500")
    if output_dir is None:
        output_dir = os.path.join(project_root, "data", "processed_sp500_enhanced")

    os.makedirs(output_dir, exist_ok=True)

    # Load HAR features
    har_file = os.path.join(processed_dir, f"{ticker}_processed.csv")
    if not os.path.isfile(har_file):
        raise FileNotFoundError(f"HAR file not found: {har_file}")

    df = pd.read_csv(har_file)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.set_index("Date")

    # Generate HAR features if not present
    if "har_daily_vol" not in df.columns:
        from src.common.har_features import generate_har_features
        df = generate_har_features(df.reset_index(), volatility_col="parkinson_volatility")
        df["Date"] = pd.to_datetime(df["date"])
        df = df.set_index("Date")

    # Add target (horizon-day ahead)
    target_col = f"target_{horizon}d"
    df[target_col] = df["parkinson_volatility"].shift(-horizon)

    if feature_set in ["har_market", "full"]:
        # Load market data
        from src.common.market_data_loader import load_market_data
        market_df = load_market_data(market="sp500", data_dir=market_data_dir)

        # Merge
        df = df.join(market_df, how="left")
        df[["vix", "treasury_10y", "sp500_index"]] = df[["vix", "treasury_10y", "sp500_index"]].ffill()

    if feature_set == "full":
        # Load sentiment
        sentiment_file = os.path.join(sentiment_dir, f"{ticker}_sentiment.csv")
        if os.path.isfile(sentiment_file):
            sent_df = pd.read_csv(sentiment_file, parse_dates=["date"])
            sent_df = sent_df.set_index("date")
            sent_df = sent_df.rename(columns={
                "sentiment_score": "sentiment_score",
                "sentiment_confidence": "sentiment_confidence",
                "news_count": "news_count",
            })

            # Merge
            df = df.join(sent_df, how="left")
            df[["sentiment_score", "sentiment_confidence", "news_count"]] = df[
                ["sentiment_score", "sentiment_confidence", "news_count"]
            ].ffill().fillna(0)
        else:
            print(f"[WARN] Sentiment file not found for {ticker}, skipping sentiment features")

    # Drop rows with NaN target
    df = df.dropna(subset=[target_col])

    # Save
    output_file = os.path.join(output_dir, f"{ticker}_enhanced.csv")
    df.to_csv(output_file)

    n_features = len([c for c in df.columns if c != target_col])
    print(f"[OK] {ticker}: {len(df)} rows, {n_features} features -> {output_file}")

    return output_file


def merge_all_tickers(
    tickers: list = None,
    feature_set: str = "full",
    **kwargs
):
    """
    Merge features for multiple tickers.

    Args:
        tickers: List of ticker symbols
        feature_set: "har", "har_market", or "full"
        **kwargs: Passed to merge_features
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL"]

    results = []
    for ticker in tickers:
        try:
            path = merge_features(ticker, feature_set=feature_set, **kwargs)
            results.append({"ticker": ticker, "file": path})
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")

    print(f"\n[SUCCESS] Merged features for {len(results)}/{len(tickers)} tickers")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge features for enhanced volatility prediction")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL"])
    parser.add_argument("--feature_set", default="full", choices=["har", "har_market", "full"])
    parser.add_argument("--forecast_horizon", type=int, default=5, choices=[1, 5, 10],
                         help="Forecast horizon in trading days ahead")
    args = parser.parse_args()

    merge_all_tickers(tickers=args.tickers, feature_set=args.feature_set, horizon=args.forecast_horizon)
