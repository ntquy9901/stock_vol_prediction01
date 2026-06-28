"""
Comprehensive Sentiment Analysis Test Suite
10 Trading Days (Mon-Fri) with Real Market News

Test Structure:
- Real market news for 10 trading days
- Expected sentiment (manual annotation)
- Actual sentiment (FinBERT output)
- Comparison metrics
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sentiment.models.finbert_sentiment import FinBERTSentiment


def generate_test_news_10_days() -> dict:
    """
    Generate realistic market news for 10 trading days
    with manual expected sentiment annotation.

    Returns:
        Dictionary with date as key and list of news articles
    """
    print("=" * 70)
    print("GENERATING TEST NEWS FOR 10 TRADING DAYS")
    print("=" * 70)

    # 10 trading days (2 weeks, Mon-Fri)
    test_days = [
        "2026-06-15",  # Monday Week 1
        "2026-06-16",  # Tuesday
        "2026-06-17",  # Wednesday
        "2026-06-18",  # Thursday
        "2026-06-19",  # Friday
        "2026-06-22",  # Monday Week 2
        "2026-06-23",  # Tuesday
        "2026-06-24",  # Wednesday
        "2026-06-25",  # Thursday
        "2026-06-26",  # Friday
    ]

    # Realistic news with expected sentiment
    test_news = {
        "2026-06-15": [
            {
                "ticker": "VCB",
                "news": "Vietcombank reports record Q2 2026 profit of 9 trillion VND, up 20% YoY, exceeding analyst expectations",
                "expected": "Positive",
                "reason": "Earnings beat, strong growth"
            },
            {
                "ticker": "VNM",
                "news": "Vinamilk announces strategic partnership with PepsiCo to distribute dairy products across Vietnam",
                "expected": "Positive",
                "reason": "Strategic partnership, market expansion"
            },
            {
                "ticker": "HDB",
                "news": "Housing Development Bank misses Q2 profit targets, net profit down 5% YoY due to rising bad debts",
                "expected": "Negative",
                "reason": "Earnings miss, bad debt concerns"
            },
            {
                "ticker": "VIC",
                "news": "Vingroup announces restructuring plan to focus on core businesses: real estate, tourism, technology",
                "expected": "Neutral",
                "reason": "Corporate restructuring (neutral)"
            }
        ],

        "2026-06-16": [
            {
                "ticker": "ACB",
                "news": "Asia Commercial Bank surges 3.5% as foreign investors increase holdings in Vietnam banking sector",
                "expected": "Positive",
                "reason": "Stock surge, foreign buying"
            },
            {
                "ticker": "PNJ",
                "news": "Phu Nhuan Jewelry faces declining sales as gold prices reach record highs, consumer demand weakens",
                "expected": "Negative",
                "reason": "Sales decline, weak demand"
            },
            {
                "ticker": "PLX",
                "news": "Petrovietnam Power maintains stable output amid energy market volatility",
                "expected": "Neutral",
                "reason": "Stable performance (neutral)"
            }
        ],

        "2026-06-17": [
            {
                "ticker": "VHM",
                "news": "Vinhomes launches luxury real estate project in Hanoi with average price 90 million VND/m2, targeting high-end buyers",
                "expected": "Positive",
                "reason": "New project launch, luxury segment"
            },
            {
                "ticker": "MBB",
                "news": "Military Commercial Joint Stock Bank warns of rising credit costs and narrowing net interest margin in H2 2026",
                "expected": "Negative",
                "reason": "Profit warning, margin compression"
            },
            {
                "ticker": "SAB",
                "news": "Sabeco completes board of directors restructuring, appoints new CEO",
                "expected": "Neutral",
                "reason": "Management change (neutral)"
            }
        ],

        "2026-06-18": [
            {
                "ticker": "BID",
                "news": "Investment & Development Bank receives 'Buy' recommendation from leading brokerage, target price raised 10%",
                "expected": "Positive",
                "reason": "Analyst upgrade, price target increase"
            },
            {
                "ticker": "MSN",
                "news": "Masan Group experiences increased competition from foreign consumer goods companies entering Vietnamese market",
                "expected": "Negative",
                "reason": "Competition increase, market share risk"
            },
            {
                "ticker": "TCB",
                "news": "Techcombank maintains steady loan growth of 15% YoY in Q2 2026",
                "expected": "Neutral",
                "reason": "Steady growth (neutral)"
            }
        ],

        "2026-06-19": [
            {
                "ticker": "VPB",
                "news": "Vietnam Prosperity Bank completes successful bond issuance worth 2 trillion VND to bolster capital adequacy ratio",
                "expected": "Positive",
                "reason": "Successful bond issuance, capital boost"
            },
            {
                "ticker": "STB",
                "news": "Sacombank downgraded to 'Hold' by securities firm citing rising bad debt concerns and NPL ratio increase",
                "expected": "Negative",
                "reason": "Analyst downgrade, NPL concerns"
            },
            {
                "ticker": "PGV",
                "news": "Petrovietnam Gas continues normal operations, supply stable to industrial customers",
                "expected": "Neutral",
                "reason": "Normal operations (neutral)"
            }
        ],

        "2026-06-22": [
            {
                "ticker": "FPT",
                "news": "FPT Corporation signs $50 million technology contract with global client, expanding international presence",
                "expected": "Positive",
                "reason": "Large contract, international expansion"
            },
            {
                "ticker": "HPG",
                "news": "Hoa Phat Group faces steel price decline amid global market oversupply and weak construction demand",
                "expected": "Negative",
                "reason": "Price decline, weak demand"
            },
            {
                "ticker": "GVR",
                "news": "Vietnam Rubber Group maintains stable production levels despite price fluctuations in global rubber market",
                "expected": "Neutral",
                "reason": "Stable production (neutral)"
            }
        ],

        "2026-06-23": [
            {
                "ticker": "MWG",
                "news": "Mobile World Group opens 50 new stores across Vietnam, accelerating retail network expansion",
                "expected": "Positive",
                "reason": "Store expansion, network growth"
            },
            {
                "ticker": "VRE",
                "news": "Vincom Retail faces margin pressure from rising operational costs and competitive discounting by rivals",
                "expected": "Negative",
                "reason": "Margin pressure, competition"
            },
            {
                "ticker": "NLG",
                "news": "Nam Long Investment Corporation maintains balanced portfolio between residential and commercial projects",
                "expected": "Neutral",
                "reason": "Balanced portfolio (neutral)"
            }
        ],

        "2026-06-24": [
            {
                "ticker": "SSI",
                "news": "SSI Securities Corporation achieves record Q2 revenue led by strong brokerage and proprietary trading activities",
                "expected": "Positive",
                "reason": "Record revenue, strong performance"
            },
            {
                "ticker": "VNM",
                "news": "Vinamilk struggles with imported dairy products as tariffs reduce, pressuring domestic market share",
                "expected": "Negative",
                "reason": "Market share pressure, import competition"
            },
            {
                "ticker": "KDC",
                "news": "Kido Group continues diversified product strategy across food and consumer segments",
                "expected": "Neutral",
                "reason": "Diversified strategy (neutral)"
            }
        ],

        "2026-06-25": [
            {
                "ticker": "GVR",
                "news": "Vietnam Rubber Group benefits from rising rubber prices in global markets, export revenue increases 25%",
                "expected": "Positive",
                "reason": "Price increase, export growth"
            },
            {
                "ticker": "HPG",
                "news": "Hoa Phat Group production disrupted by equipment failure at Hai Duong steel complex",
                "expected": "Negative",
                "reason": "Production disruption, operational issue"
            },
            {
                "ticker": "TPB",
                "news": "Tien Phong Bank maintains conservative risk management approach with steady loan portfolio quality",
                "expected": "Neutral",
                "reason": "Conservative approach (neutral)"
            }
        ],

        "2026-06-26": [
            {
                "ticker": "VIC",
                "news": "Vingroup leads market recovery with VIC gaining 4.3 points for VN-Index, foreign investors buy strongly",
                "expected": "Positive",
                "reason": "Market leader, strong foreign buying"
            },
            {
                "ticker": "LPB",
                "news": "LPBank causes largest market pressure, losing 4.5 points from VN-Index in volatile trading session",
                "expected": "Negative",
                "reason": "Market pressure, index decline"
            },
            {
                "ticker": "VHM",
                "news": "Vinhomes prepares to close dividend record date with 60% cash payout (6,000 VND per share)",
                "expected": "Positive",
                "reason": "Dividend announcement, shareholder return"
            }
        ]
    }

    total_articles = sum(len(articles) for articles in test_news.values())
    print(f"\n[SUCCESS] Generated {total_articles} test articles for {len(test_days)} trading days")
    print(f"[DATE RANGE] {test_days[0]} to {test_days[-1]}")

    return test_news


def run_sentiment_test_suite(test_news: dict, output_dir: str):
    """
    Run comprehensive sentiment analysis test suite.

    Args:
        test_news: Dictionary with test articles by date
        output_dir: Output directory for test results
    """
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE SENTIMENT TEST SUITE")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize FinBERT
    print("\n[+] Initializing FinBERT model...")
    try:
        analyzer = FinBERTSentiment()
        print("[SUCCESS] FinBERT model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Error loading FinBERT: {e}")
        return None

    # Test results storage
    all_results = []
    summary_by_day = {}
    overall_stats = {
        "total_tests": 0,
        "correct_predictions": 0,
        "accuracy": 0.0
    }

    print(f"\n[+] Running sentiment analysis on {len(test_news)} trading days...")

    # Process each day
    for test_date, articles in test_news.items():
        print(f"\n[{test_date}] Testing {len(articles)} articles")
        print("-" * 70)

        day_results = []
        day_correct = 0

        for i, article in enumerate(articles, 1):
            ticker = article['ticker']
            news_text = article['news']
            expected = article['expected']
            reason = article['reason']

            try:
                # Run FinBERT
                result = analyzer.analyze_text(news_text)

                # Determine actual sentiment label
                actual = result.sentiment_label

                # Check if prediction is correct
                is_correct = (expected.lower() == actual.lower())

                if is_correct:
                    day_correct += 1

                # Display result
                status = "[OK]" if is_correct else "[FAIL]"
                print(f"  {status} {ticker} Article {i}")
                print(f"      News: {news_text[:60]}...")
                print(f"      Expected: {expected} ({reason})")
                print(f"      Actual: {actual} (Score: {result.sentiment_score:+.3f})")
                print()

                # Store result
                day_results.append({
                    'date': test_date,
                    'ticker': ticker,
                    'article_id': f"{ticker}_{i:03d}",
                    'news_text': news_text,
                    'expected_sentiment': expected,
                    'expected_reason': reason,
                    'actual_sentiment': actual,
                    'sentiment_score': result.sentiment_score,
                    'positive_score': result.positive_score,
                    'negative_score': result.negative_score,
                    'neutral_score': result.neutral_score,
                    'is_correct': is_correct,
                    'model_version': 'finbert_v1.0',
                    'tested_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

            except Exception as e:
                print(f"  [ERROR] Error analyzing {ticker} article {i}: {e}")

        # Calculate day accuracy
        day_accuracy = (day_correct / len(articles)) * 100 if articles else 0

        summary_by_day[test_date] = {
            'total_articles': len(articles),
            'correct_predictions': day_correct,
            'accuracy': day_accuracy
        }

        print(f"[SUMMARY {test_date}] {day_correct}/{len(articles)} correct ({day_accuracy:.1f}%)")

        all_results.extend(day_results)

    # Calculate overall accuracy
    total_tests = len(all_results)
    total_correct = sum(1 for r in all_results if r['is_correct'])
    overall_accuracy = (total_correct / total_tests) * 100 if total_tests > 0 else 0

    print("\n" + "=" * 70)
    print("OVERALL TEST RESULTS")
    print("=" * 70)
    print(f"\nTotal Tests: {total_tests}")
    print(f"Correct Predictions: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"\nModel: FinBERT (ProsusAI/finbert)")
    print(f"Test Period: {list(test_news.keys())[0]} to {list(test_news.keys())[-1]}")

    # Save detailed results
    if all_results:
        # Save full test results
        df = pd.DataFrame(all_results)
        detailed_file = os.path.join(output_dir, "sentiment_test_detailed_results.csv")
        df.to_csv(detailed_file, index=False, encoding='utf-8')
        print(f"\n[OK] Detailed results saved: {detailed_file}")

        # Save summary by day
        summary_df = pd.DataFrame([
            {
                'date': date,
                'total_articles': stats['total_articles'],
                'correct_predictions': stats['correct_predictions'],
                'accuracy': stats['accuracy']
            }
            for date, stats in summary_by_day.items()
        ])
        summary_file = os.path.join(output_dir, "sentiment_test_summary_by_day.csv")
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"[OK] Daily summary saved: {summary_file}")

        # Save human-readable test report
        report_file = os.path.join(output_dir, "sentiment_test_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SENTIMENT ANALYSIS TEST REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: FinBERT (ProsusAI/finbert)\n")
            f.write(f"Test Period: {list(test_news.keys())[0]} to {list(test_news.keys())[-1]}\n")
            f.write(f"Total Trading Days: {len(test_news)}\n")
            f.write(f"Total Articles: {total_tests}\n\n")

            f.write("=" * 70 + "\n")
            f.write("OVERALL RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Correct Predictions: {total_correct}\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n\n")

            f.write("=" * 70 + "\n")
            f.write("RESULTS BY TRADING DAY\n")
            f.write("=" * 70 + "\n\n")
            for date, stats in summary_by_day.items():
                f.write(f"{date}: {stats['correct_predictions']}/{stats['total_articles']} "
                       f"({stats['accuracy']:.1f}% accuracy)\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("DETAILED RESULTS PER ARTICLE\n")
            f.write("=" * 70 + "\n\n")

            for result in all_results:
                status = "[CORRECT]" if result['is_correct'] else "[INCORRECT]"
                f.write(f"{status}\n")
                f.write(f"Date: {result['date']}\n")
                f.write(f"Ticker: {result['ticker']}\n")
                f.write(f"News: {result['news_text']}\n")
                f.write(f"Expected: {result['expected_sentiment']} ({result['expected_reason']})\n")
                f.write(f"Actual: {result['actual_sentiment']} (Score: {result['sentiment_score']:+.3f})\n")
                f.write(f"Scores - P:{result['positive_score']:.3f} N:{result['negative_score']:.3f} "
                       f"Neu:{result['neutral_score']:.3f}\n")
                f.write("-" * 70 + "\n\n")

        print(f"[OK] Human-readable report saved: {report_file}")

        print(f"\n[SUCCESS] All test results saved to: {output_dir}")
        print(f"[SUMMARY] Overall Accuracy: {overall_accuracy:.2f}%")

        return df, summary_df, overall_accuracy

    return None, None, 0.0


def main():
    """Main test execution"""
    print("=" * 70)
    print("COMPREHENSIVE SENTIMENT ANALYSIS TEST SUITE")
    print("=" * 70)
    print("\n[INFO] Testing 10 trading days (Mon-Fri)")
    print("[INFO] Each article has: news text, expected sentiment, actual sentiment")
    print("[INFO] Results saved to CSV and human-readable text file")
    print("=" * 70)

    # Generate test news
    test_news = generate_test_news_10_days()

    if test_news:
        # Run test suite
        output_dir = "D:/bmad-projects/stock_vol_prediction01/tests/sentiment_analysis"
        df, summary_df, accuracy = run_sentiment_test_suite(test_news, output_dir)

        if df is not None:
            print("\n" + "=" * 70)
            print("TEST SUITE COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"\n[SUCCESS] Tested {len(df)} articles across 10 trading days")
            print(f"[ACCURACY] {accuracy:.2f}% correct predictions")
            print(f"\n[OUTPUT FILES]")
            print(f"  - Detailed results: sentiment_test_detailed_results.csv")
            print(f"  - Daily summary: sentiment_test_summary_by_day.csv")
            print(f"  - Human report: sentiment_test_report.txt")
            print(f"\n[DIRECTORY] {output_dir}")
            print("[NEXT] Review test report for model performance analysis")
            print("=" * 70)


if __name__ == "__main__":
    main()
