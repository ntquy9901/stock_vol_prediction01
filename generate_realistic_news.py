"""
Realistic Financial News Generator

Generates realistic financial news patterns for VN30 stocks
based on actual market scenarios WITHOUT web scraping.

This avoids legal issues while providing educational value.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# VN30 stocks with sectors
VN30_STOCKS = {
    "ACB": {"name": "Asia Commercial Bank", "sector": "Banking"},
    "VCB": {"name": "Vietcombank", "sector": "Banking"},
    "BID": {"name": "Investment & Development Bank", "sector": "Banking"},
    "HDB": {"name": "Housing Development Bank", "sector": "Banking"},
    "MBB": {"name": "Military Commercial Joint Stock Bank", "sector": "Banking"},
    "TCB": {"name": "Techcombank", "sector": "Banking"},
    "VHM": {"name": "Vinhomes", "sector": "Real Estate"},
    "VIC": {"name": "Vingroup", "sector": "Conglomerate"},
    "VNM": {"name": "Vinamilk", "sector": "Consumer Goods"},
    "PGV": {"name": "Petrovietnam Gas", "sector": "Energy"},
    "PLX": {"name": "Petrovietnam Power", "sector": "Energy"},
    "SAB": {"name": "Sabeco", "sector": "Consumer Goods"},
    "PNJ": {"name": "Phu Nhuan Jewelry", "sector": "Retail"},
    "MSN": {"name": "Masan Group", "sector": "Consumer Goods"},
    "STB": {"name": "Sacombank", "sector": "Banking"},
    "VPB": {"name": "Vietnam Prosperity Bank", "sector": "Banking"},
}

# Realistic market scenarios
SCENARIOS = {
    "earnings_beat": "reports better-than-expected quarterly earnings, profit growth",
    "earnings_miss": "misses earnings expectations, profit decline or warning",
    "dividend": "announces dividend payment or increase in dividend payout",
    "partnership": "announces strategic partnership or joint venture",
    "expansion": "announces business expansion or new product launch",
    "concern": "faces regulatory concerns or increased competition",
    "upgrade": "receives upgrade or positive analyst recommendation",
    "downgrade": "receives downgrade or negative analyst coverage",
    "strong_day": "strong performance driven by positive sector momentum",
    "weak_day": "weak performance due to sector headwinds"
}

def generate_realistic_news(ticker: str, date: str) -> list:
    """Generate realistic news for given ticker and date"""

    stock_info = VN30_STOCKS[ticker]
    scenarios = list(SCENARIOS.keys())

    news_templates = {
        "earnings_beat": [
            f"{stock_info['name']} ({ticker}) reports Q2 2026 net profit of {np.random.randint(2, 5)} trillion VND, up {np.random.randint(10, 25)}% year-over-year, beating analyst expectations",
            f"{ticker} announces robust Q2 results with net interest income growing {np.random.randint(15, 30)}% YoY, driven by strong loan demand",
        ],
        "earnings_miss": [
            f"{stock_info['name']} misses Q2 profit targets, net profit {np.random.randint(1, 3)} trillion VND, down {np.random.randint(5, 15)}% from previous year",
            f"{ticker} warns of rising credit costs and narrowing net interest margin in H1 2026, concerning investors",
        ],
        "dividend": [
            f"{stock_info['name']} Board approves cash dividend of {np.random.randint(2000, 5000)} VND per share for 2026, payable in September",
            f"{ticker} announces interim dividend of {np.random.randint(1000, 3000)} VND per share with record payout ratio",
        ],
        "partnership": [
            f"{stock_info['name']} signs strategic partnership with leading {stock_info['sector']} firm to enhance market position",
            f"{ticker} enters collaboration with international partner to expand {stock_info['sector']} services into neighboring countries",
        ],
        "expansion": [
            f"{stock_info['name']} opens new branch network in {np.random.randint(3, 10)} provinces across Vietnam, strengthening retail presence",
            f"{ticker} invests {np.random.randint(500, 2000)} billion VND in new technology platform to digitize customer services",
        ],
        "concern": [
            f"{stock_info['name']} faces regulatory scrutiny over compliance requirements, may incur additional compliance costs",
            f"{ticker} experiences increased competition from foreign banks entering Vietnamese {stock_info['sector']} market",
        ],
        "upgrade": [
            f"{stock_info['name']} receives 'Buy' recommendation from leading brokerage, target price raised {np.random.randint(5, 15)}%",
            f"{ticker} upgraded to 'Outperform' by investment bank citing strong fundamentals and growth prospects",
        ],
        "downgrade": [
            f"{stock_info['name']} downgraded to 'Hold' by securities firm citing rising bad debt concerns",
            f"{ticker} faces downward pressure from foreign investors selling amid {stock_info['sector']} sector weakness",
        ],
        "strong_day": [
            f"{ticker} stock gains {np.random.uniform(1.5, 4.0):.1f}% as {stock_info['sector']} sector leads market rally",
            f"{stock_info['name']} surges to {np.random.randint(2000, 5000)} new accounts in Q2, exceeding customer acquisition targets",
        ],
        "weak_day": [
            f"{ticker} stock declines {np.random.uniform(1.0, 3.0):.1f} on profit taking after strong Q1 performance",
            f"{stock_info['name']} faces margin compression as operating costs rise faster than revenue in H1 2026",
        ]
    }

    # Select random scenarios for variety
    num_articles = np.random.randint(3, 6)
    selected_scenarios = np.random.choice(scenarios, num_articles, replace=False)

    articles = []
    for scenario in selected_scenarios:
        article_text = np.random.choice(news_templates[scenario])
        articles.append(article_text)

    return articles


def create_realistic_dataset(start_date: str, end_date: str, output_path: str):
    """
    Create realistic financial news dataset for trading days only.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_path: Where to save the CSV files
    """

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Create output directory
    import os
    os.makedirs(output_path, exist_ok=True)

    print(f"Generating realistic news dataset: {start_date} to {end_date}")
    print(f"Output directory: {output_path}")
    print("=" * 70)

    # Generate for each trading day only (Mon-Fri)
    current_date = start

    while current_date <= end:
        day_of_week = current_date.strftime("%A")

        # Skip weekends
        if day_of_week in ["Saturday", "Sunday"]:
            current_date += timedelta(days=1)
            continue

        print(f"\nGenerating news for: {current_date.strftime('%Y-%m-%d')} ({day_of_week})")

        # Generate news for all 30 VN30 stocks
        daily_news = []

        for ticker in VN30_STOCKS.keys():
            articles = generate_realistic_news(ticker, current_date.strftime("%Y-%m-%d"))

            for i, article_text in enumerate(articles):
                daily_news.append({
                    'date': current_date.strftime("%Y-%m-%d"),
                    'ticker': ticker,
                    'article_id': f"{ticker}_ART_{i:03d}",
                    'news_text': article_text,
                    'news_source': 'Realistic_Sample',
                    'is_real': False  # Mark as generated sample
                })

        if daily_news:
            # Save daily news
            df = pd.DataFrame(daily_news)
            output_file = os.path.join(output_path, f"vn30_news_{current_date.strftime('%Y%m%d')}.csv")
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"  [OK] Generated {len(daily_news)} articles for {len(df['ticker'].unique())} stocks")

        current_date += timedelta(days=1)

    print(f"\n[SUCCESS] Realistic news dataset created: {start_date} to {end_date}")
    print(f"[INFO] Total files created in: {output_path}")


if __name__ == "__main__":
    # Generate 2 weeks of realistic data (trading days only)
    create_realistic_dataset(
        start_date="2026-06-15",  # Monday
        end_date="2026-06-30",     # 2 weeks
        output_path="D:/bmad-projects/stock_vol_prediction01/data/raw/vn30_sentiment/news"
    )