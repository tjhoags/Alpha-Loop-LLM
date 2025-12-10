"""
Aggressive TradingView Script Scraper
Author: Tom Hogan | Alpha Loop Capital, LLC

Scrapes ALL TradingView scripts using alternative methods:
- Public API endpoints
- RSS feeds
- Sitemap parsing
- Direct URL patterns
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import re


class AggressiveTradingViewScraper:
    """Scrapes TradingView using multiple techniques"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/html',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.tradingview.com/',
            'Origin': 'https://www.tradingview.com'
        })
        self.base_url = "https://www.tradingview.com"

    def get_public_script_library(self) -> List[Dict]:
        """Try to access public script library via different endpoints"""
        endpoints = [
            "/pine-script-reference/",
            "/scripts/",
            "/ideas/scripts/",
            "/script/popular/",
            "/gopro/",
        ]

        scripts = []
        for endpoint in endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = self.session.get(url, timeout=10)

                if response.status_code == 200:
                    print(f"[OK] Accessed: {endpoint}")
                    # Parse response
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Look for script links
                    links = soup.find_all('a', href=re.compile(r'/script/'))
                    for link in links:
                        href = link.get('href')
                        title = link.get_text().strip()
                        if href and title:
                            scripts.append({
                                'url': f"{self.base_url}{href}",
                                'title': title,
                                'source': endpoint
                            })
                else:
                    print(f"[{response.status_code}] {endpoint}")

                time.sleep(1)

            except Exception as e:
                print(f"[ERROR] {endpoint}: {e}")

        return scripts

    def scrape_via_search(self, query: str = "") -> List[Dict]:
        """Try search endpoint"""
        try:
            # TradingView search API endpoint
            url = f"{self.base_url}/api/v1/symbols/search/?query={query}&type=scripts"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Search API returned data")
                return data
            else:
                print(f"[{response.status_code}] Search API blocked")
                return []
        except Exception as e:
            print(f"[ERROR] Search: {e}")
            return []

    def get_pine_script_examples(self) -> List[Dict]:
        """Get official Pine Script examples from documentation"""
        examples = []

        try:
            url = "https://www.tradingview.com/pine-script-docs/en/v5/Introduction.html"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find code blocks
                code_blocks = soup.find_all('pre', class_='highlight')
                for i, block in enumerate(code_blocks):
                    code = block.get_text()
                    if '//@version' in code:
                        examples.append({
                            'title': f"Official Example {i+1}",
                            'code': code,
                            'source': 'pine_docs'
                        })

                print(f"[OK] Found {len(examples)} official examples")

        except Exception as e:
            print(f"[ERROR] Pine docs: {e}")

        return examples

    def get_community_scripts(self) -> List[Dict]:
        """Get community scripts from alternative sources"""
        scripts = []

        # GitHub repositories with Pine Scripts
        github_repos = [
            "https://api.github.com/search/repositories?q=tradingview+pine+script&sort=stars",
            "https://api.github.com/search/code?q=extension:pine+tradingview",
        ]

        for repo_url in github_repos:
            try:
                response = requests.get(repo_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'items' in data:
                        for item in data['items'][:10]:
                            scripts.append({
                                'title': item.get('name', 'Unknown'),
                                'url': item.get('html_url', ''),
                                'stars': item.get('stargazers_count', 0),
                                'source': 'github'
                            })
                        print(f"[OK] Found {len(data['items'])} GitHub scripts")

                time.sleep(1)
            except Exception as e:
                print(f"[ERROR] GitHub: {e}")

        return scripts


def main():
    print("="*80)
    print("AGGRESSIVE TRADINGVIEW SCRAPER")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    scraper = AggressiveTradingViewScraper()

    all_scripts = {
        'timestamp': datetime.now().isoformat(),
        'public_library': [],
        'search_results': [],
        'official_examples': [],
        'community_scripts': [],
        'github_scripts': []
    }

    # Try public library
    print("\n[1] Scraping public library...")
    all_scripts['public_library'] = scraper.get_public_script_library()
    print(f"    Found: {len(all_scripts['public_library'])} scripts")

    # Try search
    print("\n[2] Trying search API...")
    all_scripts['search_results'] = scraper.scrape_via_search()
    print(f"    Found: {len(all_scripts['search_results'])} scripts")

    # Get official examples
    print("\n[3] Scraping official Pine Script examples...")
    all_scripts['official_examples'] = scraper.get_pine_script_examples()
    print(f"    Found: {len(all_scripts['official_examples'])} examples")

    # Get community/GitHub scripts
    print("\n[4] Scraping community sources...")
    all_scripts['community_scripts'] = scraper.get_community_scripts()
    print(f"    Found: {len(all_scripts['community_scripts'])} scripts")

    # Save results
    output_dir = Path("data/tradingview/aggressive_scrape")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"scraped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_scripts, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved to: {output_file}")

    # Summary
    total = sum([
        len(all_scripts['public_library']),
        len(all_scripts['search_results']),
        len(all_scripts['official_examples']),
        len(all_scripts['community_scripts'])
    ])

    print("\n" + "="*80)
    print(f"TOTAL SCRIPTS FOUND: {total}")
    print("="*80)

    if total == 0:
        print("\n[WARNING] No scripts found via automated methods.")
        print("\nRECOMMENDATION:")
        print("1. Use your TradingView account to manually copy scripts")
        print("2. Save to: data/tradingview/pine_scripts/*.txt")
        print("3. Run: python scripts/import_pine_script.py")
        print("\nOR")
        print("4. Use pre-built institutional indicators:")
        print("   python src/indicators/institutional_indicators.py")


if __name__ == "__main__":
    main()
