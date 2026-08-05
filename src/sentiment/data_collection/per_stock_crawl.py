"""
Per-Stock News Crawl Orchestrator

Runs news crawlers for each VN30 stock separately, organizing data into
individual folders per ticker.

Usage:
    python per_stock_crawl.py --ticker VCB --start 2020-01-01 --end 2023-12-31
    python per_stock_crawl.py --all --start 2020-01-01 --end 2026-06-30
"""

import argparse
import logging

from src.sentiment.data_collection.crawlers import CafefCrawler, get_vn30_tickers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def crawl_single_ticker(ticker: str, start_date: str, end_date: str,
                        base_dir: str = "data/sentiment/raw"):
    """
    Crawl news for a single ticker

    Args:
        ticker: Stock symbol (e.g., 'VCB')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        base_dir: Base directory for saving data
    """
    logger.info("=" * 60)
    logger.info(f"Crawling {ticker} news from {start_date} to {end_date}")
    logger.info("=" * 60)

    crawler = CafefCrawler(base_dir=base_dir)

    try:
        articles = crawler.fetch_news_for_ticker(ticker, start_date, end_date)

        if articles:
            crawler.save_articles(ticker, articles)
            logger.info(f"✅ Successfully crawled {len(articles)} articles for {ticker}")
            logger.info(f"   Location: {base_dir}/{ticker}/")
            return len(articles)
        else:
            logger.warning(f"⚠️  No articles found for {ticker}")
            return 0

    except Exception as e:
        logger.error(f"❌ Error crawling {ticker}: {e}")
        return 0


def crawl_all_tickers(start_date: str, end_date: str,
                       base_dir: str = "data/sentiment/raw",
                       tickers: list = None):
    """
    Crawl news for all VN30 tickers

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        base_dir: Base directory for saving data
        tickers: List of tickers (default: all VN30)
    """
    if tickers is None:
        tickers = get_vn30_tickers()

    logger.info("=" * 60)
    logger.info(f"CRAWLING ALL {len(tickers)} VN30 TICKERS")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info("=" * 60)

    total_articles = 0
    successful_tickers = 0

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"\n[{i}/{len(tickers)}] Processing {ticker}...")

        count = crawl_single_ticker(ticker, start_date, end_date, base_dir)

        if count > 0:
            total_articles += count
            successful_tickers += 1

        # Rate limiting between tickers
        import time
        time.sleep(3)

    logger.info("\n" + "=" * 60)
    logger.info("CRAWL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Successful tickers: {successful_tickers}/{len(tickers)}")
    logger.info(f"Total articles: {total_articles}")
    logger.info(f"Data location: {base_dir}/")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Crawl news for VN30 stocks (per-stock folders)'
    )

    parser.add_argument(
        '--ticker',
        type=str,
        help='Single ticker to crawl (e.g., VCB)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Crawl all VN30 tickers'
    )

    parser.add_argument(
        '--start',
        type=str,
        default='2020-01-01',
        help='Start date (YYYY-MM-DD), default: 2020-01-01'
    )

    parser.add_argument(
        '--end',
        type=str,
        default='2026-06-30',
        help='End date (YYYY-MM-DD), default: 2026-06-30'
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default='data/sentiment/raw',
        help='Base directory for saving data, default: data/sentiment/raw'
    )

    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        help='Specific tickers to crawl (space-separated)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.ticker and not args.all and not args.tickers:
        parser.error("Must specify --ticker, --all, or --tickers")

    # Determine which tickers to crawl
    if args.ticker:
        crawl_single_ticker(args.ticker, args.start, args.end, args.base_dir)
    elif args.tickers:
        crawl_all_tickers(args.start, args.end, args.base_dir, args.tickers)
    else:  # args.all
        crawl_all_tickers(args.start, args.end, args.base_dir)


if __name__ == "__main__":
    main()
