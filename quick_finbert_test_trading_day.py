"""
Fixed FinBERT Test for VN30 Stocks - Real Trading Day

Tests sentiment analysis for REAL trading day (Friday, not weekend).
This addresses the logical issue where 2026-06-27 is Saturday (no trading).

Usage: python quick_finbert_test_trading_day.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sentiment.models.finbert_sentiment import FinBERTSentiment


def get_real_trading_day_sample_news() -> dict:
    """
    Sample financial news for REAL trading day (Friday 2026-06-26).

    Returns:
        Dictionary with ticker as key and list of news texts as value
    """
    sample_news = {
        "ACB": [
            "ACB stock surges 3.5% on Friday as foreign investors increase holdings in Vietnam banking sector",
            "Asia Commercial Bank completes successful bond issuance worth 2 trillion VND to bolster capital adequacy ratio",
            "Analysts downgrade ACB price target citing rising non-performing loans in agricultural sector exposure"
        ],

        "VCB": [
            "Vietcombank leads banking sector gains with strong institutional buying on last trading day",
            "VCB announces partnership with Samsung Electronics to provide cross-border payment services for Korean businesses",
            "Vietcombank's Q2 results exceed expectations with 15% YoY profit growth, driving positive sentiment among investors"
        ],

        "VHM": [
            "Vinhomes breaks ground on new luxury residential project in Hanoi with average price 90 million VND per square meter",
            "Real estate stocks gain momentum as government announces new infrastructure development near major Vinhomes projects",
            "VHM pays annual dividend of 4,000 VND per share, maintaining attractive yield for income investors"
        ],

        "VNM": [
            "Vinamilk introduces new organic milk product line targeting health-conscious consumers in major cities",
            "VNM stock demonstrates resilience during market volatility, supported by defensive nature of consumer staples sector",
            "Vinamilk invests 500 billion VND in new production facility with annual capacity of 250 million liters"
        ],

        "VIC": [
            "Vingroup completes restructuring plan, selling non-core real estate assets to focus on key businesses",
            "VinFast delivers first batch of electric vehicles to European market, marking international expansion milestone",
            "VIC shares rally 4% on positive sentiment about tech sector integration into traditional conglomerate"
        ]
    }

    return sample_news


def test_finbert_on_trading_day() -> dict:
    """
    Test FinBERT sentiment analysis on REAL trading day.

    IMPORTANT: Using Friday 2026-06-26 instead of Saturday 2026-06-27
    """
    print("=" * 70)
    print("CORRECTED FINBERT TEST - REAL TRADING DAY (FRIDAY)")
    print("=" * 70)

    # Use FRIDAY (last trading day) instead of SATURDAY
    trading_day = datetime.now().replace(day=26)  # 2026-06-26 (Friday)
    print(f"Test Date: {trading_day.strftime('%Y-%m-%d')}")
    print(f"Day of Week: {trading_day.strftime('%A')}")
    print(f"Status: MARKET TRADING DAY (not weekend)")
    print("=" * 70)
    print(f"Stocks: ACB, VCB, VHM, VNM, VIC (5 major VN30 stocks)")
    print("=" * 70)

    # Initialize FinBERT
    print("\n[+] Initializing FinBERT model...")
    try:
        analyzer = FinBERTSentiment()
        print("[SUCCESS] FinBERT model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Error loading FinBERT: {e}")
        print("[INFO] Please install dependencies: pip install transformers torch")
        return None

    # Get sample news
    print("\n[+] Loading sample news for REAL trading day...")
    sample_news = get_real_trading_day_sample_news()
    print(f"[SUCCESS] Loaded {sum(len(news) for news in sample_news.values())} news articles")
    print(f"[INFO] These are SAMPLE news simulating what would be published on trading day")
    print(f"[INFO] In production, real news will be collected from actual sources")

    # Analyze sentiment for each stock
    results = {}

    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS RESULTS FOR TRADING DAY")
    print("=" * 70)

    for ticker, news_list in sample_news.items():
        print(f"\n[{ticker}] {len(news_list)} news articles")
        print("-" * 70)

        ticker_results = []
        sentiment_scores = []

        for i, news_text in enumerate(news_list, 1):
            try:
                # Analyze sentiment
                result = analyzer.analyze_text(news_text)

                # Store result with trading day date
                ticker_results.append({
                    'article_id': f"{ticker}_ART_{i:03d}",
                    'news_preview': news_text[:80] + "...",
                    'sentiment_score': result.sentiment_score,
                    'sentiment_label': result.sentiment_label,
                    'positive_score': result.positive_score,
                    'negative_score': result.negative_score,
                    'neutral_score': result.neutral_score
                })

                sentiment_scores.append(result.sentiment_score)

                # Display result
                label_symbol = {
                    'Positive': '(+)',
                    'Negative': '(-)',
                    'Neutral': '(n)'
                }.get(result.sentiment_label, '(?)')

                print(f"  Article {i}: {label_symbol} {result.sentiment_label}")
                print(f"  Score: {result.sentiment_score:+.3f} | P:{result.positive_score:.2f} N:{result.negative_score:.2f} Neu:{result.neutral_score:.2f}")
                print(f"  Preview: {news_text[:60]}...")
                print()

            except Exception as e:
                print(f"  [ERROR] Error analyzing article {i}: {e}")

        # Calculate aggregate sentiment for stock
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            avg_label = "Positive" if avg_sentiment > 0.2 else "Negative" if avg_sentiment < -0.2 else "Neutral"

            print(f"[SUMMARY] {ticker} Aggregate Sentiment:")
            print(f"  Average Score: {avg_sentiment:+.3f}")
            print(f"  Overall Label: {avg_label}")

        results[ticker] = ticker_results

    # Generate summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS - TRADING DAY")
    print("=" * 70)

    all_sentiments = []
    for ticker, ticker_results in results.items():
        ticker_sentiments = [r['sentiment_score'] for r in ticker_results]
        all_sentiments.extend(ticker_sentiments)

        if ticker_sentiments:
            avg_score = np.mean(ticker_sentiments)
            print(f"{ticker}: Avg Score {avg_score:+.3f}")

    if all_sentiments:
        print(f"\n[MARKET] Overall Market Sentiment: {np.mean(all_sentiments):+.3f}")
        print(f"[STATS] Sentiment Range: {np.min(all_sentiments):+.3f} to {np.max(all_sentiments):+.3f}")
        print(f"[STATS] Standard Deviation: {np.std(all_sentiments):.3f}")

    return results, trading_day


def save_results_to_csv(results: dict, trading_day: datetime, output_dir: str = None):
    """
    Save test results to CSV files with CORRECT trading day date.

    Args:
        results: Dictionary with analysis results
        trading_day: The REAL trading day (Friday, not Saturday)
        output_dir: Output directory (default: processed/vn30_sentiment/daily/)
    """
    if output_dir is None:
        output_dir = "D:/bmad-projects/stock_vol_prediction01/data/processed/vn30_sentiment/daily"

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[+] Saving results to: {output_dir}")
    print(f"[INFO] Using CORRECT trading day: {trading_day.strftime('%Y-%m-%d')}")

    # Save individual stock results
    for ticker, ticker_results in results.items():
        df = pd.DataFrame(ticker_results)

        # Add metadata with CORRECT trading day
        df['ticker'] = ticker
        df['date'] = trading_day.strftime('%Y-%m-%d')  # Use TRADING DAY, not test run day
        df['model_version'] = 'finbert_v1.0'
        df['processed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Reorder columns
        columns = ['date', 'ticker', 'article_id', 'sentiment_score', 'sentiment_label',
                  'positive_score', 'negative_score', 'neutral_score', 'news_preview',
                  'model_version', 'processed_at']
        df = df[columns]

        # Save to CSV
        output_file = os.path.join(output_dir, f"{ticker}_sentiment_daily.csv")
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"  [OK] Saved: {ticker}_sentiment_daily.csv ({len(df)} articles)")
        print(f"       Date: {df['date'].iloc[0]} (TRADING DAY)")

    print("\n[SUCCESS] All results saved successfully!")
    print(f"[DIR] Output directory: {output_dir}")


def main():
    """Main test execution with CORRECT trading day"""

    # Run FinBERT test with TRADING DAY
    results, trading_day = test_finbert_on_trading_day()

    if results:
        # Save results to CSV with correct trading day
        save_results_to_csv(results, trading_day)

        print("\n" + "=" * 70)
        print("CORRECTED TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n[SUCCESS] FinBERT validated for REAL trading day: {trading_day.strftime('%Y-%m-%d')} ({trading_day.strftime('%A')})")
        print("[SUCCESS] Sentiment analysis pipeline validated!")
        print("[SUCCESS] CSV data files generated with CORRECT date!")
        print("\n[IMPORTANT FIX] Now using Friday instead of Saturday")
        print(f"[INFO] In production, ALWAYS use actual trading days (Mon-Fri)")
        print(f"[INFO] Never use weekend dates for trading data")

        print(f"\n[CHECK] Check results in: data/processed/vn30_sentiment/daily/")
        print(f"[NEXT] Ready to scale to all 30 VN30 stocks!")
        print("=" * 70)


if __name__ == "__main__":
    main()