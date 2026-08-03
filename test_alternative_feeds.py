import feedparser
import sys
import io

# Fix UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Test alternative RSS feeds
feeds = [
    ('CafeBiz (Business News)', 'https://cafebiz.vn/index.rss'),
]

stock_keywords = ['VCB', 'VNM', 'VIC', 'VN30', 'VN-Index', 'chung khoan', 'co phieu', 'tu san', 'ngan hang', 'kinh doanh', 'tai chinh']

for name, url in feeds:
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print(f"{'='*60}")

    try:
        feed = feedparser.parse(url)
        print(f"Total entries: {len(feed.entries)}")

        if hasattr(feed.feed, 'title'):
            print(f"Feed title: {feed.feed.title}")

        # Analyze for stock-related content
        stock_related = []
        for entry in feed.entries:
            title_lower = entry.title.lower()
            is_stock_related = any(keyword.lower() in title_lower for keyword in stock_keywords)

            if is_stock_related:
                stock_related.append(entry)

        print(f"\nStock-related articles: {len(stock_related)}/{len(feed.entries)} ({len(stock_related)/len(feed.entries)*100:.1f}%)")

        if len(stock_related) > 0:
            print("\n--- Sample Stock Articles ---")
            for i, entry in enumerate(stock_related[:5], 1):
                print(f"\n{i}. {entry.title}")
                if hasattr(entry, 'published'):
                    print(f"   Date: {entry.published}")
                print(f"   Link: {entry.link}")

        # Viability check
        if len(stock_related) >= 10:
            print("\n✓ VIABLE: Good stock news coverage")
        elif len(stock_related) >= 5:
            print("\n~ MARGINAL: Some stock news, may need filtering")
        else:
            print("\n✗ NOT VIABLE: Insufficient stock coverage")

    except Exception as e:
        print(f"Error: {e}")
