"""
TradingView Webscraper
Author: Tom Hogan | Alpha Loop Capital, LLC

Scrapes advanced trading indicators, scripts, and modules from TradingView.
Pulls institutional-grade strategies and technical indicators.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingViewScript:
    """TradingView Pine Script"""
    title: str
    author: str
    description: str
    script_code: str
    likes: int
    category: str
    url: str
    indicators: List[str]
    timeframes: List[str]
    tags: List[str]


class TradingViewScraper:
    """
    Scrapes TradingView for:
    - Top-rated Pine Scripts
    - Popular indicators
    - Community strategies
    - Technical analysis modules
    """

    BASE_URL = "https://www.tradingview.com"

    def __init__(self, rate_limit: float = 2.0):
        """
        Args:
            rate_limit: Seconds to wait between requests (respect TradingView)
        """
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

    def search_scripts(
        self,
        query: str = "",
        category: str = "indicators",
        sort_by: str = "popularity",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search TradingView scripts

        Args:
            query: Search query (e.g., "VWAP", "RSI divergence")
            category: "indicators", "strategies", "libraries"
            sort_by: "popularity", "recent", "rating"
            limit: Max number of results

        Returns:
            List of script metadata
        """
        logger.info(f"Searching TradingView scripts: query='{query}', category={category}")

        scripts = []
        page = 1

        while len(scripts) < limit:
            url = f"{self.BASE_URL}/script/{category}/?page={page}"
            if query:
                url += f"&q={query}"
            if sort_by:
                url += f"&sort={sort_by}"

            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Parse script listings (structure may vary)
                script_elements = soup.find_all('div', class_='tv-widget-idea')

                if not script_elements:
                    logger.warning("No script elements found - TradingView may have changed structure")
                    break

                for elem in script_elements:
                    if len(scripts) >= limit:
                        break

                    script_data = self._parse_script_element(elem)
                    if script_data:
                        scripts.append(script_data)

                if not script_elements:
                    break

                page += 1
                time.sleep(self.rate_limit)

            except Exception as e:
                logger.error(f"Error fetching page {page}: {e}")
                break

        logger.info(f"Found {len(scripts)} scripts")
        return scripts[:limit]

    def _parse_script_element(self, elem) -> Optional[Dict[str, Any]]:
        """Parse a script element from search results"""
        try:
            title_elem = elem.find('a', class_='tv-widget-idea__title')
            title = title_elem.text.strip() if title_elem else "Unknown"
            url = self.BASE_URL + title_elem['href'] if title_elem and 'href' in title_elem.attrs else ""

            author_elem = elem.find('span', class_='tv-card-user-info__name')
            author = author_elem.text.strip() if author_elem else "Unknown"

            likes_elem = elem.find('span', class_='tv-social-row__likes')
            likes = int(re.sub(r'\D', '', likes_elem.text)) if likes_elem else 0

            description_elem = elem.find('p', class_='tv-widget-idea__description')
            description = description_elem.text.strip() if description_elem else ""

            return {
                'title': title,
                'author': author,
                'url': url,
                'likes': likes,
                'description': description,
            }
        except Exception as e:
            logger.debug(f"Error parsing script element: {e}")
            return None

    def get_script_details(self, script_url: str) -> Optional[TradingViewScript]:
        """
        Fetch full details and code for a specific script

        Args:
            script_url: Full URL to TradingView script

        Returns:
            TradingViewScript object with code
        """
        try:
            response = self.session.get(script_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract script code (usually in a code block or script tag)
            code_elem = soup.find('div', class_='tv-pine-script')
            script_code = code_elem.text.strip() if code_elem else ""

            # Extract metadata
            title_elem = soup.find('h1', class_='tv-chart-view__title')
            title = title_elem.text.strip() if title_elem else "Unknown"

            author_elem = soup.find('a', class_='tv-user-link')
            author = author_elem.text.strip() if author_elem else "Unknown"

            description_elem = soup.find('div', class_='tv-chart-view__description')
            description = description_elem.text.strip() if description_elem else ""

            # Extract tags
            tag_elements = soup.find_all('a', class_='tv-widget-idea__tag')
            tags = [tag.text.strip() for tag in tag_elements]

            time.sleep(self.rate_limit)

            return TradingViewScript(
                title=title,
                author=author,
                description=description,
                script_code=script_code,
                likes=0,
                category="indicator",
                url=script_url,
                indicators=[],
                timeframes=[],
                tags=tags
            )

        except Exception as e:
            logger.error(f"Error fetching script details from {script_url}: {e}")
            return None

    def get_top_indicators(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top-rated indicators from TradingView"""
        return self.search_scripts(
            query="",
            category="indicators",
            sort_by="popularity",
            limit=limit
        )

    def get_top_strategies(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top-rated trading strategies from TradingView"""
        return self.search_scripts(
            query="",
            category="strategies",
            sort_by="popularity",
            limit=limit
        )

    def search_by_keyword(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for specific indicators/strategies by keyword

        Examples:
            - "VWAP"
            - "Order Flow"
            - "Volume Profile"
            - "Smart Money"
            - "Market Structure"
        """
        return self.search_scripts(
            query=keyword,
            category="indicators",
            sort_by="popularity",
            limit=limit
        )

    def get_institutional_indicators(self) -> List[Dict[str, Any]]:
        """
        Fetch institutional-grade indicators:
        - Volume Profile
        - Order Flow
        - Market Profile
        - VWAP variants
        - Liquidity levels
        """
        keywords = [
            "Volume Profile",
            "Order Flow",
            "Market Profile",
            "Anchored VWAP",
            "Smart Money",
            "Liquidity",
            "Market Structure",
            "Footprint Chart",
            "Delta Volume",
            "Cumulative Volume Delta"
        ]

        all_indicators = []
        for keyword in keywords:
            logger.info(f"Searching for: {keyword}")
            results = self.search_by_keyword(keyword, limit=5)
            all_indicators.extend(results)
            time.sleep(self.rate_limit)

        return all_indicators

    def save_scripts_to_file(self, scripts: List[Dict[str, Any]], filename: str):
        """Save scraped scripts to JSON file"""
        output_path = Path("data/tradingview") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scripts, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(scripts)} scripts to {output_path}")
        return str(output_path)


def main():
    """Example usage"""
    logging.basicConfig(level=logging.INFO)

    scraper = TradingViewScraper(rate_limit=2.0)

    print("="*60)
    print("TRADINGVIEW SCRAPER")
    print("="*60)

    # Fetch top indicators
    print("\n[1] Fetching top indicators...")
    top_indicators = scraper.get_top_indicators(limit=10)
    print(f"Found {len(top_indicators)} indicators")

    # Fetch top strategies
    print("\n[2] Fetching top strategies...")
    top_strategies = scraper.get_top_strategies(limit=10)
    print(f"Found {len(top_strategies)} strategies")

    # Fetch institutional indicators
    print("\n[3] Fetching institutional indicators...")
    institutional = scraper.get_institutional_indicators()
    print(f"Found {len(institutional)} institutional indicators")

    # Save results
    all_scripts = {
        'top_indicators': top_indicators,
        'top_strategies': top_strategies,
        'institutional': institutional,
        'fetched_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    output_file = scraper.save_scripts_to_file([all_scripts], 'scraped_scripts.json')

    print(f"\n[OK] Results saved to: {output_file}")
    print("\nTop 5 Indicators:")
    for i, script in enumerate(top_indicators[:5], 1):
        print(f"{i}. {script.get('title', 'Unknown')} by {script.get('author', 'Unknown')}")
        print(f"   Likes: {script.get('likes', 0)} | URL: {script.get('url', 'N/A')}")


if __name__ == "__main__":
    main()
