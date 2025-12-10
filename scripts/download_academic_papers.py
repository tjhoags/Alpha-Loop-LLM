"""
Academic Whitepaper Downloader
Author: Tom Hogan | Alpha Loop Capital, LLC

Downloads institutional research from:
- AQR Capital Management
- Citadel
- Renaissance Technologies
- Two Sigma
- DE Shaw
- Academic journals (SSRN, arXiv)
"""

import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import time
import re


class AcademicPaperDownloader:
    """Downloads quant finance whitepapers"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_aqr_papers(self) -> List[Dict]:
        """AQR Capital Management research"""
        papers = []

        try:
            url = "https://www.aqr.com/Insights/Research"
            response = self.session.get(url, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find research papers
                articles = soup.find_all('a', href=re.compile(r'/Insights/Research/'))

                for article in articles[:50]:  # Top 50
                    title = article.get_text().strip()
                    link = article.get('href')

                    if title and link:
                        papers.append({
                            'title': title,
                            'url': f"https://www.aqr.com{link}" if link.startswith('/') else link,
                            'source': 'AQR',
                            'type': 'whitepaper'
                        })

                print(f"[OK] AQR: {len(papers)} papers")

        except Exception as e:
            print(f"[ERROR] AQR: {e}")

        return papers

    def get_ssrn_papers(self, query: str = "quantitative trading") -> List[Dict]:
        """SSRN (Social Science Research Network)"""
        papers = []

        try:
            # SSRN search
            url = f"https://papers.ssrn.com/sol3/results.cfm?npage=1&q={query}"
            response = self.session.get(url, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find paper links
                titles = soup.find_all('div', class_='title')

                for title_div in titles[:30]:
                    link = title_div.find('a')
                    if link:
                        papers.append({
                            'title': link.get_text().strip(),
                            'url': f"https://papers.ssrn.com{link.get('href')}",
                            'source': 'SSRN',
                            'query': query
                        })

                print(f"[OK] SSRN '{query}': {len(papers)} papers")

        except Exception as e:
            print(f"[ERROR] SSRN: {e}")

        return papers

    def get_arxiv_papers(self, query: str = "quantitative finance") -> List[Dict]:
        """arXiv.org papers"""
        papers = []

        try:
            # arXiv API
            url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=50"
            response = self.session.get(url, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'xml')

                entries = soup.find_all('entry')

                for entry in entries:
                    title = entry.find('title').get_text().strip() if entry.find('title') else ''
                    link = entry.find('id').get_text().strip() if entry.find('id') else ''
                    summary = entry.find('summary').get_text().strip() if entry.find('summary') else ''

                    if title:
                        papers.append({
                            'title': title,
                            'url': link,
                            'summary': summary[:500],
                            'source': 'arXiv',
                            'query': query
                        })

                print(f"[OK] arXiv '{query}': {len(papers)} papers")

        except Exception as e:
            print(f"[ERROR] arXiv: {e}")

        return papers

    def get_institutional_research(self) -> List[Dict]:
        """Institutional research papers (public)"""
        papers = []

        institutions = [
            {
                'name': 'Quantopian',
                'url': 'https://www.quantopian.com/research',
                'pattern': r'/research/'
            },
            {
                'name': 'QuantConnect',
                'url': 'https://www.quantconnect.com/docs',
                'pattern': r'/docs/'
            }
        ]

        for inst in institutions:
            try:
                response = self.session.get(inst['url'], timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    links = soup.find_all('a', href=re.compile(inst['pattern']))

                    for link in links[:20]:
                        title = link.get_text().strip()
                        href = link.get('href')

                        if title and len(title) > 10:
                            papers.append({
                                'title': title,
                                'url': href if href.startswith('http') else f"{inst['url']}{href}",
                                'source': inst['name']
                            })

                    print(f"[OK] {inst['name']}: {len([p for p in papers if p['source'] == inst['name']])} papers")

            except Exception as e:
                print(f"[ERROR] {inst['name']}: {e}")

        return papers

    def get_known_institutional_papers(self) -> List[Dict]:
        """Curated list of famous institutional papers"""
        papers = [
            {
                'title': 'Momentum Strategies (Jegadeesh & Titman, 1993)',
                'url': 'https://doi.org/10.1111/j.1540-6261.1993.tb04702.x',
                'source': 'Classic',
                'topic': 'momentum'
            },
            {
                'title': 'Value and Momentum Everywhere (AQR, 2013)',
                'url': 'https://www.aqr.com/Insights/Research/Journal-Article/Value-and-Momentum-Everywhere',
                'source': 'AQR',
                'topic': 'factor investing'
            },
            {
                'title': 'Quality Minus Junk (AQR, 2014)',
                'url': 'https://www.aqr.com/Insights/Research/Journal-Article/Quality-Minus-Junk',
                'source': 'AQR',
                'topic': 'quality factor'
            },
            {
                'title': 'Betting Against Beta (AQR, 2014)',
                'url': 'https://www.aqr.com/Insights/Research/Journal-Article/Betting-Against-Beta',
                'source': 'AQR',
                'topic': 'low volatility'
            },
            {
                'title': 'Time Series Momentum (Moskowitz et al, 2012)',
                'url': 'https://doi.org/10.1016/j.jfineco.2011.11.003',
                'source': 'Academic',
                'topic': 'trend following'
            },
            {
                'title': 'Fact, Fiction and Momentum Investing',
                'url': 'https://www.aqr.com/Insights/Research/Alternative-Thinking/Fact-Fiction-and-Momentum-Investing',
                'source': 'AQR',
                'topic': 'momentum'
            },
            {
                'title': 'The Other Side of Value: Gross Profitability Premium',
                'url': 'https://doi.org/10.1016/j.jfineco.2013.01.003',
                'source': 'Academic',
                'topic': 'profitability'
            },
            {
                'title': 'A Century of Evidence on Trend-Following',
                'url': 'https://doi.org/10.1093/rfs/hhz127',
                'source': 'Academic',
                'topic': 'trend following'
            },
            {
                'title': 'Machine Learning and the Cross-Section of Expected Returns',
                'url': 'https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3004534',
                'source': 'SSRN',
                'topic': 'machine learning'
            },
            {
                'title': 'Factor Momentum and the Momentum Factor',
                'url': 'https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2435323',
                'source': 'SSRN',
                'topic': 'factor timing'
            }
        ]

        print(f"[OK] Curated institutional papers: {len(papers)}")
        return papers


def main():
    print("="*80)
    print("ACADEMIC WHITEPAPER DOWNLOADER")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    downloader = AcademicPaperDownloader()

    all_papers = {
        'timestamp': datetime.now().isoformat(),
        'aqr_papers': [],
        'ssrn_papers': [],
        'arxiv_papers': [],
        'institutional_research': [],
        'curated_classics': []
    }

    # AQR
    print("\n[1] Downloading AQR Capital papers...")
    all_papers['aqr_papers'] = downloader.get_aqr_papers()

    # SSRN - Multiple queries
    print("\n[2] Downloading SSRN papers...")
    queries = [
        "quantitative trading",
        "algorithmic trading",
        "factor investing",
        "momentum strategy",
        "mean reversion",
        "statistical arbitrage"
    ]
    for query in queries:
        papers = downloader.get_ssrn_papers(query)
        all_papers['ssrn_papers'].extend(papers)
        time.sleep(2)

    # arXiv
    print("\n[3] Downloading arXiv papers...")
    arxiv_queries = [
        "quantitative finance",
        "algorithmic trading",
        "portfolio optimization",
        "risk management trading"
    ]
    for query in arxiv_queries:
        papers = downloader.get_arxiv_papers(query)
        all_papers['arxiv_papers'].extend(papers)
        time.sleep(2)

    # Institutional research
    print("\n[4] Downloading institutional research...")
    all_papers['institutional_research'] = downloader.get_institutional_research()

    # Curated classics
    print("\n[5] Adding curated classics...")
    all_papers['curated_classics'] = downloader.get_known_institutional_papers()

    # Save
    output_dir = Path("data/academic_papers")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved to: {output_file}")

    # Summary
    total = sum([
        len(all_papers['aqr_papers']),
        len(all_papers['ssrn_papers']),
        len(all_papers['arxiv_papers']),
        len(all_papers['institutional_research']),
        len(all_papers['curated_classics'])
    ])

    print("\n" + "="*80)
    print(f"TOTAL PAPERS: {total}")
    print("="*80)
    print(f"\nAQR: {len(all_papers['aqr_papers'])}")
    print(f"SSRN: {len(all_papers['ssrn_papers'])}")
    print(f"arXiv: {len(all_papers['arxiv_papers'])}")
    print(f"Institutional: {len(all_papers['institutional_research'])}")
    print(f"Curated Classics: {len(all_papers['curated_classics'])}")

    # Create reading list
    reading_list = output_dir / "READING_LIST.md"
    with open(reading_list, 'w', encoding='utf-8') as f:
        f.write("# Institutional Research Papers - Priority Reading\n\n")

        f.write("## Curated Classics (Must Read)\n\n")
        for paper in all_papers['curated_classics']:
            f.write(f"- **{paper['title']}**\n")
            f.write(f"  - Source: {paper['source']}\n")
            f.write(f"  - Topic: {paper['topic']}\n")
            f.write(f"  - URL: {paper['url']}\n\n")

        f.write("\n## AQR Capital Research\n\n")
        for paper in all_papers['aqr_papers'][:20]:
            f.write(f"- {paper['title']}\n")
            f.write(f"  - {paper['url']}\n\n")

    print(f"\n[OK] Reading list: {reading_list}")


if __name__ == "__main__":
    main()
