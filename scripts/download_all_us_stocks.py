"""
Complete US Stock Universe Downloader
Author: Tom Hogan | Alpha Loop Capital, LLC

Downloads ALL US stocks including:
- NYSE, NASDAQ, AMEX (primary exchanges)
- OTC Markets (OTCQX, OTCQB, Pink Sheets)
- Historical price data (5+ years)
- Fundamentals, volume, market cap

Total universe: ~15,000-20,000 tickers
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
from typing import List, Dict
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompleteStockUniverseDownloader:
    """Downloads complete US stock universe"""

    def __init__(self, polygon_key: str = None):
        self.polygon_key = polygon_key
        self.session = requests.Session()

    def get_all_tickers_polygon(self) -> List[Dict]:
        """Get all tickers from Polygon.io (best source)"""
        if not self.polygon_key:
            logger.warning("No Polygon API key - skipping Polygon download")
            return []

        tickers = []

        try:
            # Polygon v3 tickers endpoint
            url = f"https://api.polygon.io/v3/reference/tickers"
            params = {
                'market': 'stocks',
                'active': 'true',
                'limit': 1000,
                'apiKey': self.polygon_key
            }

            while True:
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code != 200:
                    logger.error(f"Polygon API error: {response.status_code}")
                    break

                data = response.json()
                results = data.get('results', [])

                if not results:
                    break

                for ticker_data in results:
                    tickers.append({
                        'symbol': ticker_data.get('ticker'),
                        'name': ticker_data.get('name'),
                        'market': ticker_data.get('market'),
                        'primary_exchange': ticker_data.get('primary_exchange'),
                        'type': ticker_data.get('type'),
                        'currency': ticker_data.get('currency_name'),
                        'active': ticker_data.get('active'),
                        'source': 'polygon'
                    })

                # Pagination
                next_url = data.get('next_url')
                if not next_url:
                    break

                url = f"{next_url}&apiKey={self.polygon_key}"
                time.sleep(0.2)  # Rate limiting

            logger.info(f"Polygon: {len(tickers)} tickers")

        except Exception as e:
            logger.error(f"Polygon error: {e}")

        return tickers

    def get_all_tickers_nasdaq(self) -> List[Dict]:
        """Get all NASDAQ-listed stocks"""
        tickers = []

        try:
            # NASDAQ FTP download
            url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                lines = response.text.split('\n')

                for line in lines[1:]:  # Skip header
                    if not line.strip() or line.startswith('File'):
                        continue

                    parts = line.split('|')
                    if len(parts) >= 3:
                        symbol = parts[1].strip()
                        name = parts[2].strip()
                        exchange = parts[3].strip() if len(parts) > 3 else 'NASDAQ'

                        if symbol:
                            tickers.append({
                                'symbol': symbol,
                                'name': name,
                                'market': 'stocks',
                                'primary_exchange': exchange,
                                'type': 'CS',
                                'active': True,
                                'source': 'nasdaq'
                            })

                logger.info(f"NASDAQ: {len(tickers)} tickers")

        except Exception as e:
            logger.error(f"NASDAQ error: {e}")

        return tickers

    def get_all_tickers_nyse(self) -> List[Dict]:
        """Get all NYSE/AMEX-listed stocks"""
        tickers = []

        try:
            # NYSE/AMEX from various sources
            # Method 1: FMP API (if available)
            url = "https://financialmodelingprep.com/api/v3/stock/list"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()

                for item in data:
                    if item.get('exchange') in ['NYSE', 'AMEX', 'NYSE ARCA']:
                        tickers.append({
                            'symbol': item.get('symbol'),
                            'name': item.get('name'),
                            'market': 'stocks',
                            'primary_exchange': item.get('exchange'),
                            'type': 'CS',
                            'price': item.get('price'),
                            'active': True,
                            'source': 'fmp'
                        })

                logger.info(f"NYSE/AMEX: {len(tickers)} tickers")

        except Exception as e:
            logger.error(f"NYSE error: {e}")

        return tickers

    def get_otc_stocks(self) -> List[Dict]:
        """Get OTC Market stocks (OTCQX, OTCQB, Pink Sheets)"""
        tickers = []

        try:
            # OTC Markets Group provides CSV downloads
            otc_sources = [
                ('https://www.otcmarkets.com/research/stock-screener/api/downloadCSV?type=otcqx', 'OTCQX'),
                ('https://www.otcmarkets.com/research/stock-screener/api/downloadCSV?type=otcqb', 'OTCQB'),
                ('https://www.otcmarkets.com/research/stock-screener/api/downloadCSV?type=pink', 'PINK'),
            ]

            for url, market in otc_sources:
                try:
                    df = pd.read_csv(url, timeout=30)

                    for _, row in df.iterrows():
                        symbol = str(row.get('Symbol', '')).strip()
                        if symbol:
                            tickers.append({
                                'symbol': symbol,
                                'name': row.get('Security Name', ''),
                                'market': 'otc',
                                'primary_exchange': market,
                                'type': 'CS',
                                'tier': market,
                                'active': True,
                                'source': 'otc_markets'
                            })

                    logger.info(f"{market}: {len([t for t in tickers if t['primary_exchange'] == market])} tickers")
                    time.sleep(1)

                except Exception as e:
                    logger.warning(f"{market} error: {e}")

        except Exception as e:
            logger.error(f"OTC markets error: {e}")

        return tickers

    def get_all_tickers_combined(self) -> List[Dict]:
        """Combine all sources and deduplicate"""
        all_tickers = []

        # Gather from all sources
        all_tickers.extend(self.get_all_tickers_polygon())
        all_tickers.extend(self.get_all_tickers_nasdaq())
        all_tickers.extend(self.get_all_tickers_nyse())
        all_tickers.extend(self.get_otc_stocks())

        # Deduplicate by symbol
        unique_tickers = {}
        for ticker in all_tickers:
            symbol = ticker['symbol']
            if symbol not in unique_tickers:
                unique_tickers[symbol] = ticker
            else:
                # Prefer Polygon data if available
                if ticker['source'] == 'polygon':
                    unique_tickers[symbol] = ticker

        logger.info(f"\nTotal unique tickers: {len(unique_tickers)}")
        return list(unique_tickers.values())

    def download_historical_data_batch(
        self,
        tickers: List[str],
        start_date: str = '2020-01-01',
        end_date: str = None,
        batch_size: int = 50
    ) -> Dict[str, pd.DataFrame]:
        """Download historical data in batches using yfinance"""

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        all_data = {}
        total_batches = (len(tickers) + batch_size - 1) // batch_size

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"Downloading batch {batch_num}/{total_batches} ({len(batch)} tickers)...")

            try:
                # Download batch using yfinance
                data = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    group_by='ticker',
                    threads=True,
                    progress=False
                )

                # Parse results
                for ticker in batch:
                    try:
                        if len(batch) == 1:
                            ticker_data = data
                        else:
                            ticker_data = data[ticker] if ticker in data.columns.levels[0] else None

                        if ticker_data is not None and not ticker_data.empty:
                            all_data[ticker] = ticker_data
                    except Exception as e:
                        logger.debug(f"Error parsing {ticker}: {e}")

                time.sleep(1)  # Rate limiting

            except Exception as e:
                logger.error(f"Batch {batch_num} error: {e}")

        logger.info(f"Successfully downloaded {len(all_data)}/{len(tickers)} tickers")
        return all_data

    def save_to_parquet(self, data: Dict[str, pd.DataFrame], output_dir: Path):
        """Save data to efficient parquet format"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save by exchange for easier loading
        exchanges = {
            'NYSE': [],
            'NASDAQ': [],
            'AMEX': [],
            'OTC': [],
            'OTHER': []
        }

        for ticker, df in data.items():
            df['ticker'] = ticker

            # Determine exchange (simple heuristic)
            if ticker.endswith('.OTC') or len(ticker) > 5:
                exchanges['OTC'].append(df)
            else:
                exchanges['OTHER'].append(df)

        # Save each exchange to separate parquet file
        for exchange, dfs in exchanges.items():
            if dfs:
                combined = pd.concat(dfs)
                output_file = output_dir / f"{exchange.lower()}_stocks.parquet"
                combined.to_parquet(output_file, compression='snappy')
                logger.info(f"Saved {len(dfs)} tickers to {output_file}")


def main():
    print("="*80)
    print("COMPLETE US STOCK UNIVERSE DOWNLOADER")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize
    polygon_key = None  # Add your Polygon key to .env: POLYGON_API_KEY
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        polygon_key = os.getenv('POLYGON_API_KEY')
    except:
        pass

    downloader = CompleteStockUniverseDownloader(polygon_key=polygon_key)

    # Step 1: Get all tickers
    print("\n[1] Collecting all US tickers...")
    print("-" * 80)
    all_tickers = downloader.get_all_tickers_combined()

    # Save ticker list
    ticker_list_file = Path("data/stock_universe/all_us_tickers.json")
    ticker_list_file.parent.mkdir(parents=True, exist_ok=True)

    with open(ticker_list_file, 'w') as f:
        json.dump(all_tickers, f, indent=2)

    print(f"\n[OK] Ticker list saved: {ticker_list_file}")
    print(f"Total tickers: {len(all_tickers)}")

    # Breakdown by exchange
    exchanges = {}
    for ticker in all_tickers:
        exchange = ticker.get('primary_exchange', 'UNKNOWN')
        exchanges[exchange] = exchanges.get(exchange, 0) + 1

    print("\nBreakdown by exchange:")
    for exchange, count in sorted(exchanges.items(), key=lambda x: x[1], reverse=True):
        print(f"  {exchange}: {count:,}")

    # Step 2: Download historical data
    print("\n[2] Downloading historical price data...")
    print("-" * 80)
    print("This will take several hours for 15,000+ tickers...")
    print("Downloading last 5 years of data...")

    ticker_symbols = [t['symbol'] for t in all_tickers if t.get('active', True)]

    # Download in batches
    historical_data = downloader.download_historical_data_batch(
        ticker_symbols,
        start_date='2020-01-01',
        batch_size=50
    )

    # Step 3: Save data
    print("\n[3] Saving data to disk...")
    print("-" * 80)

    output_dir = Path("data/stock_universe/historical_prices")
    downloader.save_to_parquet(historical_data, output_dir)

    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print(f"\nTotal tickers collected: {len(all_tickers):,}")
    print(f"Historical data downloaded: {len(historical_data):,}")
    print(f"Success rate: {len(historical_data)/len(ticker_symbols)*100:.1f}%")

    print(f"\nFiles saved:")
    print(f"  - Ticker list: {ticker_list_file}")
    print(f"  - Historical data: {output_dir}")

    print(f"\nStorage:")
    total_size = sum(f.stat().st_size for f in output_dir.glob('*.parquet'))
    print(f"  - Total: {total_size / 1e9:.2f} GB")

    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
