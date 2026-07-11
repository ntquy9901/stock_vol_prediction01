"""
Base News Crawler for Vietnamese Financial News

Provides common functionality for all news source crawlers.
"""

import requests
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class NewsArticle:
    """Represents a single news article"""

    def __init__(self, title: str, url: str, date: str, content: str,
                 ticker: str, source: str, metadata: Optional[Dict] = None):
        self.title = title
        self.url = url
        self.date = date  # YYYY-MM-DD format
        self.content = content
        self.ticker = ticker  # VN30 ticker symbol
        self.source = source  # cafef, vietstock, etc.
        self.metadata = metadata or {}

    def to_dict(self):
        """Convert to dictionary for saving"""
        return {
            'title': self.title,
            'url': self.url,
            'date': self.date,
            'content': self.content,
            'ticker': self.ticker,
            'source': self.source,
            **self.metadata
        }


class BaseNewsCrawler(ABC):
    """Base class for news crawlers"""

    def __init__(self, base_dir: str = "data/sentiment/raw",
                 rate_limit: float = 2.0):
        """
        Initialize crawler

        Args:
            base_dir: Base directory for saving raw data
            rate_limit: Seconds to wait between requests (default: 2s)
        """
        self.base_dir = Path(base_dir)
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        logger.info(f"[{self.__class__.__name__}] Initialized with rate_limit={rate_limit}s")

    @abstractmethod
    def fetch_news_for_ticker(self, ticker: str, start_date: str, end_date: str) -> List[NewsArticle]:
        """
        Fetch news for a specific ticker and date range

        Args:
            ticker: Stock symbol (e.g., 'VCB')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of NewsArticle objects
        """
        pass

    def save_articles(self, ticker: str, articles: List[NewsArticle]):
        """
        Save articles to per-stock CSV file

        File structure: data/sentiment/raw/{TICKER}/{YYYY-MM}.csv

        Args:
            ticker: Stock symbol
            articles: List of articles to save
        """
        if not articles:
            logger.warning(f"[{self.__class__.__name__}] No articles to save for {ticker}")
            return

        # Group articles by month (separate files per month)
        articles_by_month = {}
        for article in articles:
            # Extract year-month from date
            try:
                year_month = article.date[:7]  # YYYY-MM
                if year_month not in articles_by_month:
                    articles_by_month[year_month] = []
                articles_by_month[year_month].append(article)
            except Exception as e:
                logger.warning(f"Invalid date format for article: {article.date}")

        # Save each month separately
        ticker_dir = self.base_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        for year_month, monthly_articles in articles_by_month.items():
            output_file = ticker_dir / f"{year_month}.csv"

            # Check if file exists, append if needed
            existing_articles = []
            if output_file.exists():
                try:
                    df_existing = pd.read_csv(output_file)
                    existing_urls = set(df_existing['url'].tolist())
                    # Filter out duplicates
                    monthly_articles = [
                        a for a in monthly_articles
                        if a.url not in existing_urls
                    ]
                    if len(monthly_articles) < len(articles_by_month[year_month]):
                        logger.info(f"  Skipping {len(monthly_articles)} duplicate articles")
                except Exception as e:
                    logger.warning(f"Error reading existing file {output_file}: {e}")

            # Convert to DataFrame and save
            if monthly_articles:
                df = pd.DataFrame([a.to_dict() for a in monthly_articles])

                if output_file.exists():
                    # Append to existing file
                    df_existing = pd.read_csv(output_file)
                    df = pd.concat([df_existing, df], ignore_index=True)

                df.to_csv(output_file, index=False)
                logger.info(f"  Saved {len(monthly_articles)} articles to {output_file}")

    def fetch_with_retry(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """
        Fetch URL with retry logic

        Args:
            url: URL to fetch
            max_retries: Maximum number of retry attempts

        Returns:
            Response object or None if all retries fail
        """
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except Exception as e:
                logger.warning(f"  Attempt {attempt + 1}/{max_retries} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"  All {max_retries} attempts failed for {url}")

        return None

    def rate_limit_sleep(self):
        """Sleep for rate limiting"""
        time.sleep(self.rate_limit)


def get_vn30_tickers() -> List[str]:
    """Get list of VN30 ticker symbols"""
    return [
        "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
        "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SSI", "STB", "TCB", "TPB",
        "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
    ]
