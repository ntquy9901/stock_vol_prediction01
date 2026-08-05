import feedparser
import sys
import io

# Fix UTF-8 encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Test Cafef RSS feeds
rss_urls = [
    'https://cafef.vn/sitemaps/category.rss',
    'https://cafef.vn/home.rss',
    'https://m.cafef.vn/sitemaps/category.rss'
]

for url in rss_urls:
    print(f"\n{'='*60}")
    print(f"Testing: {url}")
    print(f"{'='*60}")

    try:
        feed = feedparser.parse(url)
        print("[OK] Feed loaded successfully")
        print(f"Total entries: {len(feed.entries)}")

        if hasattr(feed.feed, 'title'):
            print(f"Feed title: {feed.feed.title}")

        if len(feed.entries) > 0:
            print("\nFirst 3 entries:")
            for i, entry in enumerate(feed.entries[:3], 1):
                print(f"\nEntry {i}:")
                print(f"  Title: {entry.title[:80]}...")
                print(f"  Link: {entry.link}")
                if hasattr(entry, 'published'):
                    print(f"  Published: {entry.published}")
                if hasattr(entry, 'description'):
                    print(f"  Description: {entry.description[:100]}...")
        else:
            print("[WARNING] No entries found")

    except Exception as e:
        print(f"[ERROR] {e}")
