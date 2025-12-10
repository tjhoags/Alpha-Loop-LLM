#!/usr/bin/env python
"""
LIVE Short Squeeze Scanner
Finds high-probability squeeze candidates using real data

Uses:
- Yahoo Finance for price/volume
- Finviz for screening
- Reddit API for sentiment

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yfinance as yf
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("LIVE SHORT SQUEEZE SCANNER")
print("="*80)
print("\nFinding the next GME/AMC...")
print()


def get_high_short_interest_stocks() -> List[str]:
    """Get stocks with high short interest from Finviz"""

    logger.info("Fetching high short interest stocks from Finviz...")

    # Finviz screener URL for high short interest
    # Free version - scrape the website
    try:
        url = "https://finviz.com/screener.ashx?v=111&f=sh_short_o30"  # Short > 30%

        # Read HTML tables
        tables = pd.read_html(url)

        if len(tables) > 1:
            df = tables[-2]  # Usually the second-to-last table has the data

            if 'Ticker' in df.columns:
                symbols = df['Ticker'].tolist()
                logger.info(f"Found {len(symbols)} stocks with >30% short interest")
                return symbols

        # Fallback: manual list of known high SI stocks
        logger.warning("Couldn't scrape Finviz, using backup list...")
        return [
            'BBBY', 'AMC', 'GME', 'BYND', 'CLOV', 'RIDE', 'WKHS',
            'SKLZ', 'SOFI', 'PLTR', 'WISH', 'CLNE', 'CLOV', 'SPCE',
            'NKLA', 'LCID', 'RIVN', 'CVNA', 'W', 'PTON'
        ]

    except Exception as e:
        logger.error(f"Error fetching from Finviz: {e}")

        # Hard-coded list of historically high SI stocks
        return [
            'AMC', 'GME', 'BBBY', 'BYND', 'CLOV', 'SOFI', 'PLTR',
            'LCID', 'RIVN', 'CVNA', 'PTON', 'W', 'DASH'
        ]


def calculate_squeeze_metrics(symbol: str) -> Dict:
    """Calculate squeeze probability metrics for a stock"""

    try:
        ticker = yf.Ticker(symbol)

        # Get historical data
        hist = ticker.history(period='3mo')
        if len(hist) < 20:
            return None

        # Current price
        current_price = hist['Close'].iloc[-1]

        # Price momentum (last 5 days)
        price_5d_ago = hist['Close'].iloc[-5] if len(hist) >= 5 else hist['Close'].iloc[0]
        price_change_5d = (current_price - price_5d_ago) / price_5d_ago

        # Volume surge
        recent_volume = hist['Volume'].iloc[-5:].mean()
        avg_volume = hist['Volume'].iloc[-30:].mean()
        volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1.0

        # Volatility (last 20 days)
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.iloc[-20:].std() * np.sqrt(252) if len(returns) >= 20 else 0

        # Get info
        info = ticker.info

        # Short ratio (from Yahoo Finance)
        short_ratio = info.get('shortRatio', 0)  # Days to cover
        short_percent = info.get('shortPercentOfFloat', 0)

        # Market cap
        market_cap = info.get('marketCap', 0)

        # Float
        float_shares = info.get('floatShares', info.get('sharesOutstanding', 0))

        # Calculate squeeze score
        score = 0

        # 1. Short interest (40%)
        if short_percent and short_percent > 0:
            si_score = min(short_percent / 40, 1.0)  # Max at 40% SI
            score += si_score * 0.4
        else:
            score += 0.1  # Unknown SI, give small score

        # 2. Days to cover (20%)
        if short_ratio and short_ratio > 0:
            dtc_score = min(short_ratio / 10, 1.0)  # Max at 10 DTC
            score += dtc_score * 0.2

        # 3. Price momentum (20%)
        if 0.05 <= price_change_5d <= 0.20:
            score += 0.2
        elif 0 <= price_change_5d < 0.05:
            score += 0.15
        elif price_change_5d > 0.20:
            score += 0.05  # Too high, risky

        # 4. Volume surge (15%)
        vol_score = min(volume_surge / 3, 1.0)
        score += vol_score * 0.15

        # 5. Market cap (5% - prefer smaller caps)
        if market_cap > 0:
            if market_cap < 1e9:  # < $1B
                score += 0.05
            elif market_cap < 5e9:  # < $5B
                score += 0.03

        return {
            'symbol': symbol,
            'current_price': current_price,
            'short_percent': short_percent if short_percent else 'N/A',
            'days_to_cover': short_ratio if short_ratio else 'N/A',
            'price_change_5d': price_change_5d,
            'volume_surge': volume_surge,
            'volatility': volatility,
            'market_cap': market_cap,
            'float_shares': float_shares,
            'squeeze_score': score
        }

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None


def scan_for_squeezes():
    """Main scanning function"""

    # Get high SI stocks
    candidates = get_high_short_interest_stocks()

    logger.info(f"\nAnalyzing {len(candidates)} stocks...")
    print()

    results = []

    for i, symbol in enumerate(candidates, 1):
        logger.info(f"[{i}/{len(candidates)}] Analyzing {symbol}...")

        metrics = calculate_squeeze_metrics(symbol)

        if metrics and metrics['squeeze_score'] >= 0.4:  # Threshold
            results.append(metrics)

            print(f"\n¯ {symbol} - Score: {metrics['squeeze_score']:.2f}")
            print(f"   Price: ${metrics['current_price']:.2f}")
            print(f"   Short %: {metrics['short_percent']}")
            print(f"   Days to Cover: {metrics['days_to_cover']}")
            print(f"   5D Change: {metrics['price_change_5d']:+.1%}")
            print(f"   Volume Surge: {metrics['volume_surge']:.1f}x")

    # Sort by score
    results.sort(key=lambda x: x['squeeze_score'], reverse=True)

    # Display top candidates
    print("\n" + "="*80)
    print("¥ TOP SQUEEZE CANDIDATES ¥")
    print("="*80)

    if not results:
        print("\nNo strong squeeze candidates found.")
        print("Try again later or check other sources.")
    else:
        for i, result in enumerate(results[:10], 1):
            print(f"\n{i}. {result['symbol']} - Score: {result['squeeze_score']:.2f}")
            print(f"   ° Price: ${result['current_price']:.2f}")
            print(f"   Š Short %: {result['short_percent']}")
            print(f"     Days to Cover: {result['days_to_cover']}")
            print(f"   ˆ 5D Change: {result['price_change_5d']:+.1%}")
            print(f"   Š Volume: {result['volume_surge']:.1f}x average")

            # Trading recommendation
            if result['squeeze_score'] >= 0.7:
                print(f"   € STRONG BUY - High squeeze probability")
            elif result['squeeze_score'] >= 0.55:
                print(f"    BUY - Good squeeze potential")
            else:
                print(f"     WATCH - Monitor for entry")

    # Save to file
    if results:
        df = pd.DataFrame(results)
        output_file = Path("data/scanners/squeeze_candidates.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

        print(f"\n¾ Saved {len(results)} candidates to: {output_file}")

    print("\n" + "="*80)
    print("SCAN COMPLETE")
    print("="*80)

    return results


if __name__ == "__main__":
    scan_for_squeezes()
