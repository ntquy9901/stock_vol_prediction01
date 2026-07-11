"""
Cafef.vn News Crawler for VN30 Stocks

Cafef is the largest Vietnamese financial news website with extensive coverage
of VN30 stocks. This crawler fetches news articles and filters by ticker symbols.

Author: Stock Volatility Prediction Team
Date: 2026-06-28
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Optional
import re
import logging
from datetime import datetime, timedelta
from urllib.parse import urljoin

from .base_crawler import BaseNewsCrawler, NewsArticle, get_vn30_tickers

logger = logging.getLogger(__name__)


class CafefCrawler(BaseNewsCrawler):
    """
    Cafef.vn news crawler specialized for VN30 stocks

    Features:
    - Search by ticker symbol (e.g., "VCB")
    - Filter by date range
    - Extract article content
    - Handle pagination
    """

    def __init__(self, base_dir: str = "data/sentiment/raw", rate_limit: float = 2.0):
        super().__init__(base_dir, rate_limit)
        self.base_url = "https://cafef.vn"
        self.search_url = "https://cafef.vn/tim-kiem"

        logger.info(f"[CafefCrawler] Initialized")

    def fetch_news_for_ticker(self, ticker: str, start_date: str, end_date: str) -> List[NewsArticle]:
        """
        Fetch news for a specific ticker and date range from Cafef

        Uses Google site search to find articles mentioning the ticker on cafef.vn

        Args:
            ticker: Stock symbol (e.g., 'VCB')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of NewsArticle objects
        """
        logger.info(f"[CafefCrawler] Fetching {ticker} news from {start_date} to {end_date} (via Google site search)")

        articles = []
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

        # Use Google site search to find Cafef articles
        # Example: "VCB site:cafef.vn"
        google_search_url = f"https://www.google.com/search?q={ticker}+site%3Acafef.vn&tbm=nws"

        page = 1
        max_pages = 5  # Limit to avoid too many requests

        while page <= max_pages:
            try:
                # Construct Google search URL with pagination
                google_url = f"{google_search_url}&start={ (page-1) * 10 }"

                logger.info(f"  Searching Google page {page}...")
                response = self.fetch_with_retry(google_url, max_retries=3)

                if response is None:
                    logger.warning(f"  Failed to fetch Google search page {page}")
                    break

                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract Cafef article links from Google search results
                article_links = self._extract_links_from_google_search(soup, ticker)

                if not article_links:
                    # No more articles found
                    logger.info(f"  No more articles found for {ticker} on Google page {page}")
                    break

                logger.info(f"  Found {len(article_links)} article links on page {page}")

                # Fetch each article
                for link in article_links:
                    try:
                        article = self._fetch_article(link, ticker)
                        if article and self._is_in_date_range(article.date, start_date, end_date):
                            articles.append(article)
                            logger.debug(f"  Fetched: {article.title[:50]}...")
                    except Exception as e:
                        logger.warning(f"  Failed to fetch article {link}: {e}")

                    # Rate limiting between articles
                    self.rate_limit_sleep()

                page += 1

                # Check if we have enough articles (limit to avoid too many requests)
                if len(articles) >= 100:
                    logger.info(f"  Reached limit of 100 articles for {ticker}")
                    break

            except Exception as e:
                logger.error(f"  Error processing Google search page {page}: {e}")
                break

        logger.info(f"[CafefCrawler] Fetched {len(articles)} articles for {ticker}")
        return articles

    def _extract_article_links(self, soup: BeautifulSoup, ticker: str) -> List[str]:
        """
        Extract article links from search results page (DEPRECATED - now using Google search)

        Args:
            soup: BeautifulSoup object of search results page
            ticker: Ticker symbol (for validation)

        Returns:
            List of article URLs
        """
        links = []

        # Cafef search results structure
        # Articles are in div with class 'item-news'
        news_items = soup.find_all('div', class_='item-news')

        for item in news_items:
            try:
                link_tag = item.find('a', href=True)
                if link_tag and link_tag.get('href'):
                    article_url = link_tag['href']

                    # Ensure absolute URL
                    if not article_url.startswith('http'):
                        article_url = urljoin(self.base_url, article_url)

                    # Validate URL contains article path (not category/search)
                    if '/tin-tuc/' in article_url or '/doi-moi/' in article_url:
                        links.append(article_url)

            except Exception as e:
                logger.debug(f"  Error extracting link: {e}")

        return links

    def _extract_links_from_google_search(self, soup: BeautifulSoup, ticker: str) -> List[str]:
        """
        Extract Cafef article links from Google search results

        Args:
            soup: BeautifulSoup object of Google search results page
            ticker: Ticker symbol (for validation)

        Returns:
            List of article URLs from cafef.vn
        """
        links = []

        try:
            # Google search results are in divs with class 'g'
            search_results = soup.find_all('div', class_='g')

            for result in search_results:
                try:
                    # Find the main link in the search result
                    link_tag = result.find('a', href=True)
                    if link_tag and link_tag.get('href'):
                        article_url = link_tag['href']

                    # Only include links from cafef.vn
                    if 'cafef.vn' in article_url and article_url.startswith('http'):
                        # Filter out non-article pages (categories, data pages, etc.)
                        if any(pattern in article_url for pattern in ['/chn', '/tin-tuc/', '/doi-moi/']):
                            # Remove any Google redirect parameters
                            if '&url=' in article_url:
                                article_url = article_url.split('&url=')[1].split('&')[0]

                            links.append(article_url)
                            logger.debug(f"    Found: {article_url}")

                except Exception as e:
                    logger.debug(f"    Error extracting link from search result: {e}")

        except Exception as e:
            logger.debug(f"  Error parsing Google search results: {e}")

        return links

    def _fetch_article(self, article_url: str, ticker: str) -> Optional[NewsArticle]:
        """
        Fetch full article content

        Args:
            article_url: Article URL
            ticker: Ticker symbol

        Returns:
            NewsArticle object or None
        """
        response = self.fetch_with_retry(article_url)

        if response is None:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title = self._extract_title(soup)

        # Extract date
        date = self._extract_date(soup)

        # Extract content
        content = self._extract_content(soup)

        # Extract metadata
        metadata = {
            'author': self._extract_author(soup),
            'tags': self._extract_tags(soup),
            'related_stocks': self._extract_related_stocks(soup)
        }

        # Validate article mentions the ticker
        if not self._mentions_ticker(title, content, ticker, metadata):
            logger.debug(f"  Article doesn't mention {ticker}, skipping")
            return None

        article = NewsArticle(
            title=title,
            url=article_url,
            date=date,
            content=content,
            ticker=ticker,
            source='cafef',
            metadata=metadata
        )

        return article

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title"""
        try:
            # Try multiple title selectors
            title_selectors = [
                'h1.title',
                'h1.article-title',
                'h2.title',
                '.title'
            ]

            for selector in title_selectors:
                title_tag = soup.select_one(selector)
                if title_tag:
                    return title_tag.get_text().strip()

            # Fallback to meta title
            meta_title = soup.find('meta', property='og:title')
            if meta_title:
                return meta_title.get('content', '').strip()

        except Exception as e:
            logger.debug(f"  Error extracting title: {e}")

        return "Unknown Title"

    def _extract_date(self, soup: BeautifulSoup) -> str:
        """Extract article date in YYYY-MM-DD format"""
        try:
            # Try multiple date selectors
            date_selectors = [
                'time.datetime',
                'span.date',
                '.date'
            ]

            for selector in date_selectors:
                date_tag = soup.select_one(selector)
                if date_tag:
                    date_str = date_tag.get_text().strip()
                    # Parse Vietnamese date format
                    # Common formats: "15/06/2026", "15-06-2026", "2026-06-15"
                    date_obj = self._parse_vietnamese_date(date_str)
                    if date_obj:
                        return date_obj.strftime('%Y-%m-%d')

        except Exception as e:
            logger.debug(f"  Error extracting date: {e}")

        # Fallback to current date
        return datetime.now().strftime('%Y-%m-%d')

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract article content"""
        try:
            # Try multiple content selectors
            content_selectors = [
                'div.article-content',
                'div.detail-content',
                'div.content',
                'article'
            ]

            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    # Get text, remove extra whitespace
                    content = content_div.get_text(separator=' ', strip=True)
                    return content

        except Exception as e:
            logger.debug(f"  Error extracting content: {e}")

        return ""

    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extract article author"""
        try:
            author_tag = soup.find('span', class_='author')
            if author_tag:
                return author_tag.get_text().strip()
        except:
            pass
        return ""

    def _extract_tags(self, soup: BeautifulSoup) -> List[str]:
        """Extract article tags"""
        try:
            tag_links = soup.find_all('a', class_='tag')
            return [link.get_text().strip() for link in tag_links if link]
        except:
            return []

    def _extract_related_stocks(self, soup: BeautifulSoup) -> List[str]:
        """Extract related stock symbols from article"""
        try:
            # Find all VN30 ticker mentions in content
            content = soup.get_text()
            tickers = get_vn30_tickers()
            mentioned = [t for t in tickers if t in content]
            return mentioned
        except:
            return []

    def _has_next_page(self, soup: BeautifulSoup) -> bool:
        """Check if there's a next page"""
        try:
            next_button = soup.find('a', class_='next')
            return next_button is not None and next_button.get('href')
        except:
            return False

    def _is_in_date_range(self, article_date: str, start_date: str, end_date: str) -> bool:
        """Check if article date is within range"""
        try:
            article_date_obj = datetime.strptime(article_date, '%Y-%m-%d')
            start_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_obj = datetime.strptime(end_date, '%Y-%m-%d')
            return start_obj <= article_date_obj <= end_obj
        except:
            return False

    def _mentions_ticker(self, title: str, content: str, ticker: str, metadata: dict) -> bool:
        """Validate article mentions the ticker"""
        # Check title
        if ticker in title:
            return True

        # Check content
        if ticker in content:
            return True

        # Check related stocks
        related = metadata.get('related_stocks', [])
        if ticker in related:
            return True

        return False

    def _parse_vietnamese_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse Vietnamese date string

        Common formats:
        - "15/06/2026" (DD/MM/YYYY)
        - "15-06-2026" (DD-MM-YYYY)
        - "2026-06-15" (YYYY-MM-DD)
        """
        formats_to_try = [
            '%d/%m/%Y',    # 15/06/2026
            '%d-%m-%Y',    # 15-06-2026
            '%Y-%m-%d',    # 2026-06-15
            '%d/%m/%y',    # 15/06/26
        ]

        for fmt in formats_to_try:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue

        return None


def main():
    """Test Cafef crawler"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    crawler = CafefCrawler()

    # Test on VCB for recent date range
    ticker = "VCB"
    start_date = "2026-06-01"
    end_date = "2026-06-30"

    logger.info(f"Testing Cafef crawler for {ticker} ({start_date} to {end_date})")

    articles = crawler.fetch_news_for_ticker(ticker, start_date, end_date)

    logger.info(f"Fetched {len(articles)} articles")

    if articles:
        crawler.save_articles(ticker, articles)
        logger.info(f"Saved articles to data/sentiment/raw/{ticker}/")

        # Show sample
        for article in articles[:3]:
            print(f"\nSample Article:")
            print(f"  Title: {article.title}")
            print(f"  Date: {article.date}")
            print(f"  URL: {article.url}")


if __name__ == "__main__":
    main()
