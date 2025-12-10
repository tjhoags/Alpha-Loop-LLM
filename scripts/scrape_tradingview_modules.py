"""
Scrape TradingView Modules - Complete Run
Author: Tom Hogan | Alpha Loop Capital, LLC

Runs both scraping strategies:
A) Scrape by specific keywords
B) Scrape institutional indicators
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_sources.tradingview_scraper import TradingViewScraper


def main():
    print("="*80)
    print("TRADINGVIEW MODULE SCRAPER - COMPLETE RUN")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    scraper = TradingViewScraper(rate_limit=2.0)

    all_results = {
        'metadata': {
            'scraped_at': datetime.now().isoformat(),
            'scraper_version': '1.0',
            'total_scripts': 0
        },
        'option_a_keyword_searches': {},
        'option_b_institutional': [],
        'option_c_top_indicators': [],
        'option_d_top_strategies': []
    }

    # ================================================================
    # OPTION A: Scrape by Keyword
    # ================================================================
    print("\n" + "="*80)
    print("OPTION A: SCRAPE BY KEYWORD")
    print("="*80)

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
        "Cumulative Volume Delta",
        "VPOC",
        "Value Area",
        "Iceberg Orders",
        "Tape Reading",
        "Volume Weighted",
        "Order Blocks",
        "Fair Value Gap",
        "Supply Demand",
        "Wyckoff",
        "Institutional Order Flow"
    ]

    for i, keyword in enumerate(keywords, 1):
        print(f"\n[{i}/{len(keywords)}] Searching: '{keyword}'")
        try:
            results = scraper.search_by_keyword(keyword, limit=5)
            all_results['option_a_keyword_searches'][keyword] = results
            all_results['metadata']['total_scripts'] += len(results)
            print(f"  Found: {len(results)} scripts")

            # Print top 3
            for j, script in enumerate(results[:3], 1):
                print(f"    {j}. {script.get('title', 'Unknown')} by {script.get('author', 'Unknown')}")
                print(f"       Likes: {script.get('likes', 0)} | {script.get('url', 'N/A')}")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results['option_a_keyword_searches'][keyword] = []

    # ================================================================
    # OPTION B: Institutional Indicators
    # ================================================================
    print("\n" + "="*80)
    print("OPTION B: INSTITUTIONAL INDICATORS (Comprehensive)")
    print("="*80)

    try:
        institutional = scraper.get_institutional_indicators()
        all_results['option_b_institutional'] = institutional
        all_results['metadata']['total_scripts'] += len(institutional)
        print(f"\nFound {len(institutional)} institutional indicators")

        # Group by keyword
        by_keyword = {}
        for script in institutional:
            desc = script.get('description', '').lower()
            title = script.get('title', '').lower()

            for kw in ['volume profile', 'order flow', 'market profile', 'vwap', 'smart money', 'liquidity']:
                if kw in desc or kw in title:
                    if kw not in by_keyword:
                        by_keyword[kw] = []
                    by_keyword[kw].append(script)

        print("\nGrouped by category:")
        for kw, scripts in by_keyword.items():
            print(f"  {kw.upper()}: {len(scripts)} scripts")

    except Exception as e:
        print(f"ERROR: {e}")
        all_results['option_b_institutional'] = []

    # ================================================================
    # OPTION C: Top Indicators (Overall)
    # ================================================================
    print("\n" + "="*80)
    print("OPTION C: TOP 50 INDICATORS (All Categories)")
    print("="*80)

    try:
        top_indicators = scraper.get_top_indicators(limit=50)
        all_results['option_c_top_indicators'] = top_indicators
        all_results['metadata']['total_scripts'] += len(top_indicators)
        print(f"\nFound {len(top_indicators)} top indicators")

        print("\nTop 10:")
        for i, script in enumerate(top_indicators[:10], 1):
            print(f"  {i}. {script.get('title', 'Unknown')}")
            print(f"     Author: {script.get('author', 'Unknown')} | Likes: {script.get('likes', 0)}")

    except Exception as e:
        print(f"ERROR: {e}")
        all_results['option_c_top_indicators'] = []

    # ================================================================
    # OPTION D: Top Strategies
    # ================================================================
    print("\n" + "="*80)
    print("OPTION D: TOP 50 STRATEGIES")
    print("="*80)

    try:
        top_strategies = scraper.get_top_strategies(limit=50)
        all_results['option_d_top_strategies'] = top_strategies
        all_results['metadata']['total_scripts'] += len(top_strategies)
        print(f"\nFound {len(top_strategies)} top strategies")

        print("\nTop 10:")
        for i, script in enumerate(top_strategies[:10], 1):
            print(f"  {i}. {script.get('title', 'Unknown')}")
            print(f"     Author: {script.get('author', 'Unknown')} | Likes: {script.get('likes', 0)}")

    except Exception as e:
        print(f"ERROR: {e}")
        all_results['option_d_top_strategies'] = []

    # ================================================================
    # Save Results
    # ================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_dir = Path("data/tradingview")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save complete results
    complete_file = output_dir / f"complete_scrape_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(complete_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Complete results: {complete_file}")

    # Save individual files
    for key, data in [
        ('keyword_searches', all_results['option_a_keyword_searches']),
        ('institutional', all_results['option_b_institutional']),
        ('top_indicators', all_results['option_c_top_indicators']),
        ('top_strategies', all_results['option_d_top_strategies'])
    ]:
        filename = output_dir / f"{key}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[OK] {key}: {filename}")

    # ================================================================
    # Summary Statistics
    # ================================================================
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nTotal Scripts Scraped: {all_results['metadata']['total_scripts']}")
    print(f"\nBreakdown:")
    print(f"  Option A (Keyword Searches): {sum(len(v) for v in all_results['option_a_keyword_searches'].values())} scripts across {len(keywords)} keywords")
    print(f"  Option B (Institutional): {len(all_results['option_b_institutional'])} scripts")
    print(f"  Option C (Top Indicators): {len(all_results['option_c_top_indicators'])} scripts")
    print(f"  Option D (Top Strategies): {len(all_results['option_d_top_strategies'])} scripts")

    print(f"\nMost Popular Keywords:")
    keyword_counts = {k: len(v) for k, v in all_results['option_a_keyword_searches'].items()}
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    for kw, count in sorted_keywords[:10]:
        print(f"  {kw}: {count} scripts")

    # Top scripts by likes
    all_scripts_flat = []
    for scripts in all_results['option_a_keyword_searches'].values():
        all_scripts_flat.extend(scripts)
    all_scripts_flat.extend(all_results['option_b_institutional'])
    all_scripts_flat.extend(all_results['option_c_top_indicators'])
    all_scripts_flat.extend(all_results['option_d_top_strategies'])

    # Deduplicate by URL
    unique_scripts = {}
    for script in all_scripts_flat:
        url = script.get('url', '')
        if url and url not in unique_scripts:
            unique_scripts[url] = script

    print(f"\nUnique Scripts: {len(unique_scripts)}")

    # Sort by likes
    sorted_by_likes = sorted(unique_scripts.values(), key=lambda x: x.get('likes', 0), reverse=True)
    print(f"\nTop 10 Most Liked Scripts:")
    for i, script in enumerate(sorted_by_likes[:10], 1):
        print(f"  {i}. {script.get('title', 'Unknown')} - {script.get('likes', 0)} likes")
        print(f"     Author: {script.get('author', 'Unknown')}")
        print(f"     URL: {script.get('url', 'N/A')}")

    print("\n" + "="*80)
    print("SCRAPING COMPLETE")
    print("="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nAll data saved to: data/tradingview/")


if __name__ == "__main__":
    main()
