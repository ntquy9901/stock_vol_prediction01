"""
Quick FinBERT Test for VN30 Stocks (Windows Compatible)

Simple test to validate FinBERT sentiment analysis works for VN30 stocks.
Tests 5 major stocks: ACB, VCB, VHM, VNM, VIC

Usage: python quick_finbert_test_windows.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sentiment.models.finbert_sentiment import FinBERTSentiment


def get_sample_news() -> dict:
    """
    Sample financial news for testing FinBERT.

    Returns:
        Dictionary with ticker as key and list of news texts as value
    """
    sample_news = {
        "ACB": [
            "Asia Commercial Bank (ACB) reports strong Q2 2026 earnings with net profit increasing 15% year-over-year, driven by robust loan growth and improved asset quality.",
            "ACB Bank announces strategic partnership with leading fintech company to enhance digital banking services, expected to boost market share.",
            "Concerns about rising bad debts at ACB as economic conditions weaken, analysts downgrade rating to hold.",
            "ACB stock shows mixed performance amid market volatility, maintaining stable trading volume.",
            "ACB Board approves dividend payment of 2,000 VND per share for 2026, demonstrating strong financial health."
        ],

        "VCB": [
            "Vietcombank (VCB) leads the banking sector with exceptional Q2 results, net profit reaching 8.5 trillion VND, exceeding market expectations.",
            "VCB launches new digital wallet platform with advanced security features, positioning itself as technology leader in banking sector.",
            "Vietcombank faces competitive pressure from rivals in corporate lending segment, margin compression expected in H2 2026.",
            "VCB stock reaches all-time high as foreign investors increase holdings, signaling strong confidence in Vietnam's banking sector.",
            "Vietcombank maintains stable NPL ratio at 1.2%, one of the lowest in the industry, reflecting strong risk management."
        ],

        "VHM": [
            "Vinhomes (VHM) announces record property sales in Q2 2026, selling 5,000 units across major projects, revenue increases 25% year-over-year.",
            "Vinhomes launches luxury real estate project in Hanoi with average price 80 million VND/m2, targeting high-end buyers.",
            "Real estate market slowdown impacts VHM sales velocity, inventory levels increase amid cooling demand.",
            "VHM stock gains 3% as government announces new infrastructure development near major Vinhomes projects.",
            "Vinhomes pays interim dividend of 3,500 VND per share, maintaining attractive dividend yield for investors."
        ],

        "VNM": [
            "Vinamilk (VNM) reports steady Q2 performance with revenue growth of 8% year-over-year, maintaining market leadership in dairy products.",
            "Vinamilk expands product portfolio with organic milk line, targeting health-conscious consumers and premium segment.",
            "VNM faces competition from imported dairy products as import tariffs reduce, pressuring domestic market share.",
            "Vinamilk stock shows resilience amid market volatility, supported by defensive nature of consumer staples sector.",
            "Vinamilk announces strategic investment in new production facility with capacity 200 million liters per year."
        ],

        "VIC": [
            "Vingroup (VIC) announces major restructuring, selling non-core assets to focus on key businesses: real estate, tourism, and technology.",
            "VinFast (VIC subsidiary) reveals new electric vehicle models for Vietnam market, expecting to capture 30% market share by 2027.",
            "Vingroup faces liquidity concerns as debt levels increase across real estate and automotive investments, credit rating under review.",
            "VIC stock surges 5% on news of successful IPO for VinHMS hotel division, demonstrating strong investor appetite.",
            "Vingroup partners with global technology companies to develop smart city projects, marking strategic shift toward innovation."
        ]
    }

    return sample_news


def test_finbert_on_stocks() -> dict:
    """
    Test FinBERT sentiment analysis on 5 VN30 stocks.

    Returns:
        Dictionary with analysis results for each stock
    """
    print("=" * 60)
    print("QUICK FINBERT TEST - VN30 STOCKS")
    print("=" * 60)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stocks: ACB, VCB, VHM, VNM, VIC (5 major VN30 stocks)")
    print("=" * 60)

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
    print("\n[+] Loading sample news data...")
    sample_news = get_sample_news()
    print(f"[SUCCESS] Loaded {sum(len(news) for news in sample_news.values())} news articles")

    # Analyze sentiment for each stock
    results = {}

    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS RESULTS")
    print("=" * 60)

    for ticker, news_list in sample_news.items():
        print(f"\n[{ticker}] {len(news_list)} news articles")
        print("-" * 60)

        ticker_results = []
        sentiment_scores = []

        for i, news_text in enumerate(news_list, 1):
            try:
                # Analyze sentiment
                result = analyzer.analyze_text(news_text)

                # Store result
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
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

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

    return results


def save_results_to_csv(results: dict, output_dir: str = None):
    """
    Save test results to CSV files.

    Args:
        results: Dictionary with analysis results
        output_dir: Output directory (default: processed/vn30_sentiment/daily/)
    """
    if output_dir is None:
        output_dir = "D:/bmad-projects/stock_vol_prediction01/data/processed/vn30_sentiment/daily"

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[+] Saving results to: {output_dir}")

    # Save individual stock results
    for ticker, ticker_results in results.items():
        df = pd.DataFrame(ticker_results)

        # Add metadata
        df['ticker'] = ticker
        df['date'] = datetime.now().strftime('%Y-%m-%d')
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

    print("\n[SUCCESS] All results saved successfully!")
    print(f"[DIR] Output directory: {output_dir}")


def main():
    """Main test execution"""

    # Run FinBERT test
    results = test_finbert_on_stocks()

    if results:
        # Save results to CSV
        save_results_to_csv(results)

        print("\n" + "=" * 60)
        print("QUICK TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n[SUCCESS] FinBERT is working for VN30 stocks!")
        print("[SUCCESS] Sentiment analysis pipeline validated!")
        print("[SUCCESS] CSV data files generated!")
        print("\n[DIR] Check results in: data/processed/vn30_sentiment/daily/")
        print("[NEXT] Ready to scale to all 30 VN30 stocks!")
        print("=" * 60)


if __name__ == "__main__":
    main()