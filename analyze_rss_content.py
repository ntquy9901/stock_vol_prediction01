import feedparser
import sys
import io
import re

# Fix UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load main RSS feed
url = 'https://cafef.vn/home.rss'
feed = feedparser.parse(url)

print(f"Total entries: {len(feed.entries)}")
print(f"\nAnalyzing content...")

# Analyze entries for stock-related keywords
stock_keywords = ['VCB', 'VNM', 'VIC', 'VN30', 'VN-Index', 'chung khoan', 'co phieu', 'tu san', 'ngan hang']

stock_related = []
general_news = []

for entry in feed.entries:
    title_lower = entry.title.lower()

    # Check if title contains stock keywords
    is_stock_related = any(keyword.lower() in title_lower for keyword in stock_keywords)

    if is_stock_related:
        stock_related.append(entry)
    else:
        general_news.append(entry)

print(f"\nStock-related articles: {len(stock_related)}")
print(f"General news articles: {len(general_news)}")

if len(stock_related) > 0:
    print(f"\n--- Stock-Related Articles ---")
    for i, entry in enumerate(stock_related[:5], 1):
        print(f"\n{i}. {entry.title}")
        print(f"   Date: {entry.published}")
        print(f"   Link: {entry.link}")

print(f"\n--- General News Samples ---")
for i, entry in enumerate(general_news[:3], 1):
    print(f"\n{i}. {entry.title}")
    print(f"   Date: {entry.published}")

# Check if RSS feed is sufficient
print(f"\n--- RSS Feed Viability Assessment ---")
print(f"Total entries: {len(feed.entries)}")
print(f"Stock-related ratio: {len(stock_related) / len(feed.entries) * 100:.1f}%")

if len(stock_related) >= 10:
    print("✓ RSS feed contains sufficient stock-related news")
else:
    print("✗ RSS feed has too few stock-related articles")
