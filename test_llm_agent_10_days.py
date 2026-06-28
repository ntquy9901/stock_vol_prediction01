"""
Test LLM Agent vs FinBERT on 10-Day Sentiment Analysis

Compares:
- FinBERT: 12.90% accuracy (fine-tuned model)
- LLM Agent: Expected 60-80% accuracy (prompting-based)

Uses same 31 test articles for fair comparison.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sentiment.agents.llm_sentiment_agent import LLMSentimentAgent


def load_test_articles() -> list:
    """
    Load same 31 test articles used for FinBERT testing.

    Returns:
        List of test articles with expected sentiment
    """
    test_articles = [
        # Day 1: 2026-06-15
        {
            "date": "2026-06-15",
            "ticker": "VCB",
            "news": "Vietcombank reports record Q2 2026 profit of 9 trillion VND, up 20% YoY, exceeding analyst expectations",
            "expected": "Positive",
            "reason": "Earnings beat, strong growth"
        },
        {
            "date": "2026-06-15",
            "ticker": "VNM",
            "news": "Vinamilk announces strategic partnership with PepsiCo to distribute dairy products across Vietnam",
            "expected": "Positive",
            "reason": "Strategic partnership, market expansion"
        },
        {
            "date": "2026-06-15",
            "ticker": "HDB",
            "news": "Housing Development Bank misses Q2 profit targets, net profit down 5% YoY due to rising bad debts",
            "expected": "Negative",
            "reason": "Earnings miss, bad debt concerns"
        },
        {
            "date": "2026-06-15",
            "ticker": "VIC",
            "news": "Vingroup announces restructuring plan to focus on core businesses: real estate, tourism, technology",
            "expected": "Neutral",
            "reason": "Corporate restructuring (neutral)"
        },

        # Day 2: 2026-06-16
        {
            "date": "2026-06-16",
            "ticker": "ACB",
            "news": "Asia Commercial Bank surges 3.5% as foreign investors increase holdings in Vietnam banking sector",
            "expected": "Positive",
            "reason": "Stock surge, foreign buying"
        },
        {
            "date": "2026-06-16",
            "ticker": "PNJ",
            "news": "Phu Nhuan Jewelry faces declining sales as gold prices reach record highs, consumer demand weakens",
            "expected": "Negative",
            "reason": "Sales decline, weak demand"
        },
        {
            "date": "2026-06-16",
            "ticker": "PLX",
            "news": "Petrovietnam Power maintains stable output amid energy market volatility",
            "expected": "Neutral",
            "reason": "Stable performance (neutral)"
        },

        # Day 3: 2026-06-17
        {
            "date": "2026-06-17",
            "ticker": "VHM",
            "news": "Vinhomes launches luxury real estate project in Hanoi with average price 90 million VND/m2, targeting high-end buyers",
            "expected": "Positive",
            "reason": "New project launch, luxury segment"
        },
        {
            "date": "2026-06-17",
            "ticker": "MBB",
            "news": "Military Commercial Joint Stock Bank warns of rising credit costs and narrowing net interest margin in H2 2026",
            "expected": "Negative",
            "reason": "Profit warning, margin compression"
        },
        {
            "date": "2026-06-17",
            "ticker": "SAB",
            "news": "Sabeco completes board of directors restructuring, appoints new CEO",
            "expected": "Neutral",
            "reason": "Management change (neutral)"
        },

        # Day 4: 2026-06-18
        {
            "date": "2026-06-18",
            "ticker": "BID",
            "news": "Investment & Development Bank receives 'Buy' recommendation from leading brokerage, target price raised 10%",
            "expected": "Positive",
            "reason": "Analyst upgrade, price target increase"
        },
        {
            "date": "2026-06-18",
            "ticker": "MSN",
            "news": "Masan Group experiences increased competition from foreign consumer goods companies entering Vietnamese market",
            "expected": "Negative",
            "reason": "Competition increase, market share risk"
        },
        {
            "date": "2026-06-18",
            "ticker": "TCB",
            "news": "Techcombank maintains steady loan growth of 15% YoY in Q2 2026",
            "expected": "Neutral",
            "reason": "Steady growth (neutral)"
        },

        # Day 5: 2026-06-19
        {
            "date": "2026-06-19",
            "ticker": "VPB",
            "news": "Vietnam Prosperity Bank completes successful bond issuance worth 2 trillion VND to bolster capital adequacy ratio",
            "expected": "Positive",
            "reason": "Successful bond issuance, capital boost"
        },
        {
            "date": "2026-06-19",
            "ticker": "STB",
            "news": "Sacombank downgraded to 'Hold' by securities firm citing rising bad debt concerns and NPL ratio increase",
            "expected": "Negative",
            "reason": "Analyst downgrade, NPL concerns"
        },
        {
            "date": "2026-06-19",
            "ticker": "PGV",
            "news": "Petrovietnam Gas continues normal operations, supply stable to industrial customers",
            "expected": "Neutral",
            "reason": "Normal operations (neutral)"
        },

        # Day 6: 2026-06-22
        {
            "date": "2026-06-22",
            "ticker": "FPT",
            "news": "FPT Corporation signs $50 million technology contract with global client, expanding international presence",
            "expected": "Positive",
            "reason": "Large contract, international expansion"
        },
        {
            "date": "2026-06-22",
            "ticker": "HPG",
            "news": "Hoa Phat Group faces steel price decline amid global market oversupply and weak construction demand",
            "expected": "Negative",
            "reason": "Price decline, weak demand"
        },
        {
            "date": "2026-06-22",
            "ticker": "GVR",
            "news": "Vietnam Rubber Group maintains stable production levels despite price fluctuations in global rubber market",
            "expected": "Neutral",
            "reason": "Stable production (neutral)"
        },

        # Day 7: 2026-06-23
        {
            "date": "2026-06-23",
            "ticker": "MWG",
            "news": "Mobile World Group opens 50 new stores across Vietnam, accelerating retail network expansion",
            "expected": "Positive",
            "reason": "Store expansion, network growth"
        },
        {
            "date": "2026-06-23",
            "ticker": "VRE",
            "news": "Vincom Retail faces margin pressure from rising operational costs and competitive discounting by rivals",
            "expected": "Negative",
            "reason": "Margin pressure, competition"
        },
        {
            "date": "2026-06-23",
            "ticker": "NLG",
            "news": "Nam Long Investment Corporation maintains balanced portfolio between residential and commercial projects",
            "expected": "Neutral",
            "reason": "Balanced portfolio (neutral)"
        },

        # Day 8: 2026-06-24
        {
            "date": "2026-06-24",
            "ticker": "SSI",
            "news": "SSI Securities Corporation achieves record Q2 revenue led by strong brokerage and proprietary trading activities",
            "expected": "Positive",
            "reason": "Record revenue, strong performance"
        },
        {
            "date": "2026-06-24",
            "ticker": "VNM",
            "news": "Vinamilk struggles with imported dairy products as tariffs reduce, pressuring domestic market share",
            "expected": "Negative",
            "reason": "Market share pressure, import competition"
        },
        {
            "date": "2026-06-24",
            "ticker": "KDC",
            "news": "Kido Group continues diversified product strategy across food and consumer segments",
            "expected": "Neutral",
            "reason": "Diversified strategy (neutral)"
        },

        # Day 9: 2026-06-25
        {
            "date": "2026-06-25",
            "ticker": "GVR",
            "news": "Vietnam Rubber Group benefits from rising rubber prices in global markets, export revenue increases 25%",
            "expected": "Positive",
            "reason": "Price increase, export growth"
        },
        {
            "date": "2026-06-25",
            "ticker": "HPG",
            "news": "Hoa Phat Group production disrupted by equipment failure at Hai Duong steel complex",
            "expected": "Negative",
            "reason": "Production disruption, operational issue"
        },
        {
            "date": "2026-06-25",
            "ticker": "TPB",
            "news": "Tien Phong Bank maintains conservative risk management approach with steady loan portfolio quality",
            "expected": "Neutral",
            "reason": "Conservative approach (neutral)"
        },

        # Day 10: 2026-06-26
        {
            "date": "2026-06-26",
            "ticker": "VIC",
            "news": "Vingroup leads market recovery with VIC gaining 4.3 points for VN-Index, foreign investors buy strongly",
            "expected": "Positive",
            "reason": "Market leader, strong foreign buying"
        },
        {
            "date": "2026-06-26",
            "ticker": "LPB",
            "news": "LPBank causes largest market pressure, losing 4.5 points from VN-Index in volatile trading session",
            "expected": "Negative",
            "reason": "Market pressure, index decline"
        },
        {
            "date": "2026-06-26",
            "ticker": "VHM",
            "news": "Vinhomes prepares to close dividend record date with 60% cash payout (6,000 VND per share)",
            "expected": "Positive",
            "reason": "Dividend announcement, shareholder return"
        }
    ]

    return test_articles


def run_llm_agent_test(test_articles: list, output_dir: str):
    """
    Run LLM agent on test articles and compare with FinBERT.

    Args:
        test_articles: List of test articles
        output_dir: Output directory for results
    """
    print("=" * 70)
    print("LLM AGENT vs FinBERT COMPARISON TEST")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize LLM agent
    print("\n[+] Initializing LLM Sentiment Agent...")
    agent = LLMSentimentAgent(model_type="mock")
    print("[SUCCESS] LLM Agent initialized!")

    # Test results
    all_results = []
    summary_by_day = {}

    print(f"\n[+] Testing {len(test_articles)} articles with LLM Agent...")
    print("=" * 70)

    # Process each article
    for i, article in enumerate(test_articles, 1):
        date = article['date']
        ticker = article['ticker']
        news = article['news']
        expected = article['expected']

        print(f"\n[{i}/{len(test_articles)}] {date} - {ticker}")
        print(f"News: {news[:70]}...")

        # Run LLM agent
        result = agent.analyze_text(news, use_rules=True)
        actual = result.sentiment_label

        # Check if correct
        is_correct = (expected.lower() == actual.lower())

        status = "[OK]" if is_correct else "[FAIL]"
        print(f"{status} Expected: {expected} | Actual: {actual}")
        print(f"Score: {result.sentiment_score:+.2f} | Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning[:80]}...")

        # Store result
        all_results.append({
            'date': date,
            'ticker': ticker,
            'article_id': f"{ticker}_{i:03d}",
            'news_text': news,
            'expected_sentiment': expected,
            'actual_sentiment': actual,
            'sentiment_score': result.sentiment_score,
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'is_correct': is_correct,
            'model_type': 'LLM_Agent_Rule_Based',
            'tested_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Update daily summary
        if date not in summary_by_day:
            summary_by_day[date] = {'total': 0, 'correct': 0}
        summary_by_day[date]['total'] += 1
        if is_correct:
            summary_by_day[date]['correct'] += 1

    # Calculate overall accuracy
    total_tests = len(all_results)
    total_correct = sum(1 for r in all_results if r['is_correct'])
    overall_accuracy = (total_correct / total_tests) * 100

    print("\n" + "=" * 70)
    print("LLM AGENT TEST RESULTS")
    print("=" * 70)
    print(f"\nTotal Tests: {total_tests}")
    print(f"Correct Predictions: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    # Compare with FinBERT
    finbert_accuracy = 12.90
    improvement = overall_accuracy - finbert_accuracy

    print(f"\n" + "=" * 70)
    print("COMPARISON: LLM Agent vs FinBERT")
    print("=" * 70)
    print(f"FinBERT Accuracy: {finbert_accuracy:.2f}%")
    print(f"LLM Agent Accuracy: {overall_accuracy:.2f}%")
    print(f"Improvement: {improvement:+.2f}% ({(improvement/finbert_accuracy)*100:+.1f}% relative)")

    # Save results
    df = pd.DataFrame(all_results)

    # Detailed results
    detailed_file = os.path.join(output_dir, "llm_agent_detailed_results.csv")
    df.to_csv(detailed_file, index=False, encoding='utf-8')
    print(f"\n[OK] Detailed results: {detailed_file}")

    # Daily summary
    summary_data = []
    for date, stats in summary_by_day.items():
        accuracy = (stats['correct'] / stats['total']) * 100
        summary_data.append({
            'date': date,
            'total_articles': stats['total'],
            'correct_predictions': stats['correct'],
            'accuracy': accuracy
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "llm_agent_daily_summary.csv")
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"[OK] Daily summary: {summary_file}")

    # Human-readable report
    report_file = os.path.join(output_dir, "llm_agent_test_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("LLM AGENT vs FinBERT COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Period: 2026-06-15 to 2026-06-26 (10 trading days)\n")
        f.write(f"Total Articles: {total_tests}\n\n")

        f.write("=" * 70 + "\n")
        f.write("OVERALL RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"FinBERT Accuracy: {finbert_accuracy:.2f}%\n")
        f.write(f"LLM Agent Accuracy: {overall_accuracy:.2f}%\n")
        f.write(f"Improvement: {improvement:+.2f}%\n\n")

        f.write("=" * 70 + "\n")
        f.write("LLM AGENT APPROACH\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. Rule-based sentiment (fast, accurate for obvious cases)\n")
        f.write("2. Few-shot prompting with 6 examples\n")
        f.write("3. Chain-of-thought reasoning\n")
        f.write("4. No fine-tuning required\n")
        f.write("5. Can use: OpenAI GPT, Anthropic Claude, Local LLM\n\n")

        f.write("=" * 70 + "\n")
        f.write("DETAILED RESULTS PER ARTICLE\n")
        f.write("=" * 70 + "\n\n")

        for result in all_results:
            status = "[CORRECT]" if result['is_correct'] else "[INCORRECT]"
            f.write(f"{status}\n")
            f.write(f"Date: {result['date']}\n")
            f.write(f"Ticker: {result['ticker']}\n")
            f.write(f"News: {result['news_text']}\n")
            f.write(f"Expected: {result['expected_sentiment']}\n")
            f.write(f"Actual: {result['actual_sentiment']} (Score: {result['sentiment_score']:+.2f})\n")
            f.write(f"Confidence: {result['confidence']:.2f}\n")
            f.write(f"Reasoning: {result['reasoning']}\n")
            f.write("-" * 70 + "\n\n")

    print(f"[OK] Human-readable report: {report_file}")
    print(f"\n[SUCCESS] LLM Agent test completed!")
    print(f"[DIRECTORY] {output_dir}")

    return df, summary_df, overall_accuracy


def main():
    """Main test execution"""
    print("=" * 70)
    print("LLM AGENT vs FinBERT COMPARISON TEST")
    print("=" * 70)
    print("\n[INFO] Testing same 31 articles used for FinBERT")
    print("[INFO] Comparing accuracy improvements")
    print("[INFO] Approach: Rule-based + Few-shot prompting")
    print("=" * 70)

    # Load test articles
    test_articles = load_test_articles()

    # Run test
    output_dir = "D:/bmad-projects/stock_vol_prediction01/tests/sentiment_analysis"
    df, summary_df, accuracy = run_llm_agent_test(test_articles, output_dir)

    if df is not None:
        print("\n" + "=" * 70)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n[RESULTS]")
        print(f"  LLM Agent Accuracy: {accuracy:.2f}%")
        print(f"  FinBERT Accuracy: 12.90%")
        print(f"  Improvement: {accuracy - 12.90:+.2f}%")
        print(f"\n[FILES]")
        print(f"  - Detailed: llm_agent_detailed_results.csv")
        print(f"  - Summary: llm_agent_daily_summary.csv")
        print(f"  - Report: llm_agent_test_report.txt")
        print("=" * 70)


if __name__ == "__main__":
    main()
