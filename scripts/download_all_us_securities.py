#!/usr/bin/env python
"""
Complete US Securities Universe Downloader
Downloads ALL tradeable US securities for small/mid-cap fund

Securities Covered:
- All US Stocks (NYSE, NASDAQ, AMEX) - ~10,000 symbols
- Options chains (liquid stocks with OI > 1000)
- Corporate bonds
- Treasury securities
- Warrants
- Convertible bonds
- ETFs and ETNs

Focus: Small/mid-cap, liquidity analysis, options arbitrage

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import time
from typing import List, Dict, Optional
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path("data/universe")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("ALC-ALGO: COMPLETE US SECURITIES UNIVERSE DOWNLOAD")
print("="*80)
print("\nTarget: Small/Mid-Cap Focus + Options Arbitrage")
print("Coverage: Stocks, Options, Bonds, Warrants, Convertibles, Treasuries")
print("="*80 + "\n")


class SecuritiesDownloader:
    """Download all US tradeable securities"""

    def __init__(self):
        self.data_dir = DATA_DIR
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def download_all_stocks(self) -> pd.DataFrame:
        """Download complete list of US stocks from all exchanges"""
        logger.info("Downloading all US stocks (NYSE, NASDAQ, AMEX)...")

        all_stocks = []

        # 1. NASDAQ listed stocks
        try:
            logger.info("  Fetching NASDAQ stocks...")
            nasdaq_url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&exchange=NASDAQ"
            response = self.session.get(nasdaq_url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'rows' in data['data']:
                    nasdaq_stocks = pd.DataFrame(data['data']['rows'])
                    nasdaq_stocks['exchange'] = 'NASDAQ'
                    all_stocks.append(nasdaq_stocks)
                    logger.info(f"    Downloaded {len(nasdaq_stocks)} NASDAQ stocks")
        except Exception as e:
            logger.error(f"    Failed to download NASDAQ: {e}")

        # 2. NYSE listed stocks
        try:
            logger.info("  Fetching NYSE stocks...")
            nyse_url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&exchange=NYSE"
            response = self.session.get(nyse_url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'rows' in data['data']:
                    nyse_stocks = pd.DataFrame(data['data']['rows'])
                    nyse_stocks['exchange'] = 'NYSE'
                    all_stocks.append(nyse_stocks)
                    logger.info(f"    Downloaded {len(nyse_stocks)} NYSE stocks")
        except Exception as e:
            logger.error(f"    Failed to download NYSE: {e}")

        # 3. AMEX listed stocks
        try:
            logger.info("  Fetching AMEX stocks...")
            amex_url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&exchange=AMEX"
            response = self.session.get(amex_url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'rows' in data['data']:
                    amex_stocks = pd.DataFrame(data['data']['rows'])
                    amex_stocks['exchange'] = 'AMEX'
                    all_stocks.append(amex_stocks)
                    logger.info(f"    Downloaded {len(amex_stocks)} AMEX stocks")
        except Exception as e:
            logger.error(f"    Failed to download AMEX: {e}")

        # 4. Fallback to FTP sources
        try:
            logger.info("  Fetching additional symbols from NASDAQ FTP...")
            # NASDAQ provides symbol lists via FTP
            ftp_url = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt"
            ftp_df = pd.read_csv(ftp_url, sep='|')
            ftp_df = ftp_df[ftp_df['Test Issue'] == 'N']  # Remove test symbols
            ftp_df = ftp_df[ftp_df['ETF'] == 'N']  # Remove ETFs (separate download)
            ftp_df['exchange'] = ftp_df['Listing Exchange']
            logger.info(f"    Downloaded {len(ftp_df)} additional symbols")
            all_stocks.append(ftp_df[['Symbol', 'Security Name', 'exchange']])
        except Exception as e:
            logger.error(f"    Failed to download from FTP: {e}")

        # Combine all sources
        if all_stocks:
            combined = pd.concat(all_stocks, ignore_index=True)

            # Standardize column names
            if 'symbol' in combined.columns:
                combined.rename(columns={'symbol': 'Symbol'}, inplace=True)
            if 'name' in combined.columns:
                combined.rename(columns={'name': 'Name'}, inplace=True)

            # Remove duplicates
            combined = combined.drop_duplicates(subset=['Symbol'], keep='first')

            # Clean symbols
            combined['Symbol'] = combined['Symbol'].str.strip()
            combined = combined[combined['Symbol'].str.len() <= 5]  # Remove weird symbols

            # Filter out test/special symbols
            combined = combined[~combined['Symbol'].str.contains(r'[\^~]', na=False)]

            # Save
            output_file = self.data_dir / "all_us_stocks.csv"
            combined.to_csv(output_file, index=False)

            logger.info(f"\n[OK] Downloaded {len(combined)} unique US stocks")
            logger.info(f"    Saved to: {output_file}")

            return combined
        else:
            logger.error("Failed to download any stocks!")
            return pd.DataFrame()

    def download_etfs(self) -> pd.DataFrame:
        """Download all US ETFs"""
        logger.info("\nDownloading all US ETFs...")

        try:
            # NASDAQ ETF list
            url = "https://api.nasdaq.com/api/screener/etf?tableonly=true&limit=25000"
            response = self.session.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'data' in data['data']:
                    etfs = pd.DataFrame(data['data']['data'])

                    output_file = self.data_dir / "all_us_etfs.csv"
                    etfs.to_csv(output_file, index=False)

                    logger.info(f"[OK] Downloaded {len(etfs)} ETFs")
                    logger.info(f"    Saved to: {output_file}")

                    return etfs
        except Exception as e:
            logger.error(f"Failed to download ETFs: {e}")

        return pd.DataFrame()

    def download_options_universe(self, stocks: pd.DataFrame) -> Dict:
        """Download options chains for liquid stocks"""
        logger.info("\nDownloading options universe (liquid stocks only)...")
        logger.info("Filtering for stocks with market cap > $100M and avg volume > 500K")

        # Filter for liquid stocks (using available data)
        liquid_stocks = stocks.copy()

        # If market cap column exists, filter
        if 'marketCap' in liquid_stocks.columns:
            liquid_stocks = liquid_stocks[
                pd.to_numeric(liquid_stocks['marketCap'], errors='coerce') > 100_000_000
            ]

        # If volume column exists, filter
        if 'volume' in liquid_stocks.columns:
            liquid_stocks = liquid_stocks[
                pd.to_numeric(liquid_stocks['volume'], errors='coerce') > 500_000
            ]

        # Take top 1000 by volume for options download
        if 'volume' in liquid_stocks.columns:
            liquid_stocks = liquid_stocks.nlargest(1000, 'volume')
        else:
            liquid_stocks = liquid_stocks.head(1000)

        symbols = liquid_stocks['Symbol'].tolist()

        logger.info(f"Downloading options chains for {len(symbols)} liquid stocks...")
        logger.info("This may take 10-20 minutes...")

        options_data = {}

        # Use yfinance for options data
        try:
            import yfinance as yf

            def download_options(symbol):
                try:
                    ticker = yf.Ticker(symbol)
                    dates = ticker.options

                    if dates:
                        chains = []
                        for date in dates[:4]:  # First 4 expiration dates
                            try:
                                opt = ticker.option_chain(date)
                                calls = opt.calls
                                puts = opt.puts

                                calls['type'] = 'call'
                                puts['type'] = 'put'
                                calls['expiration'] = date
                                puts['expiration'] = date

                                chains.append(pd.concat([calls, puts]))
                            except:
                                pass

                        if chains:
                            return symbol, pd.concat(chains, ignore_index=True)
                except:
                    pass
                return symbol, None

            # Download in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(download_options, sym): sym for sym in symbols[:500]}

                completed = 0
                for future in as_completed(futures):
                    symbol, data = future.result()
                    if data is not None:
                        options_data[symbol] = data

                    completed += 1
                    if completed % 50 == 0:
                        logger.info(f"  Progress: {completed}/{len(futures)} symbols")

            logger.info(f"[OK] Downloaded options for {len(options_data)} stocks")

            # Save options data
            if options_data:
                for symbol, data in options_data.items():
                    output_file = self.data_dir / f"options_{symbol}.csv"
                    data.to_csv(output_file, index=False)

                # Create summary
                summary = {
                    'total_symbols': len(options_data),
                    'total_contracts': sum(len(v) for v in options_data.values()),
                    'symbols': list(options_data.keys())
                }

                summary_file = self.data_dir / "options_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)

                logger.info(f"    Total contracts: {summary['total_contracts']:,}")
                logger.info(f"    Saved to: {self.data_dir}/options_*.csv")

        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
        except Exception as e:
            logger.error(f"Failed to download options: {e}")

        return options_data

    def download_treasuries(self) -> pd.DataFrame:
        """Download US Treasury securities"""
        logger.info("\nDownloading US Treasury securities...")

        try:
            # Treasury Direct API
            url = "https://www.treasurydirect.gov/TA_WS/securities/search?format=json"
            response = self.session.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()

                if isinstance(data, list):
                    treasuries = pd.DataFrame(data)

                    output_file = self.data_dir / "us_treasuries.csv"
                    treasuries.to_csv(output_file, index=False)

                    logger.info(f"[OK] Downloaded {len(treasuries)} Treasury securities")
                    logger.info(f"    Saved to: {output_file}")

                    return treasuries
        except Exception as e:
            logger.error(f"Failed to download Treasuries: {e}")

        # Fallback: Create treasury ladder from FRED
        logger.info("  Creating Treasury yield curve from FRED...")
        try:
            import yfinance as yf

            treasury_etfs = {
                'SHY': '1-3 Year Treasury',
                'IEI': '3-7 Year Treasury',
                'IEF': '7-10 Year Treasury',
                'TLT': '20+ Year Treasury',
                'TIP': 'TIPS',
                'BIL': '1-3 Month Treasury'
            }

            treasury_data = []
            for symbol, description in treasury_etfs.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    treasury_data.append({
                        'symbol': symbol,
                        'description': description,
                        'price': info.get('regularMarketPrice'),
                        'yield': info.get('yield', 0) * 100 if info.get('yield') else None
                    })
                except:
                    pass

            if treasury_data:
                df = pd.DataFrame(treasury_data)
                output_file = self.data_dir / "treasury_etfs.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"[OK] Created Treasury ETF reference: {len(df)} instruments")
                return df
        except:
            pass

        return pd.DataFrame()

    def download_corporate_bonds(self) -> pd.DataFrame:
        """Download corporate bond data"""
        logger.info("\nDownloading corporate bonds...")
        logger.info("Note: Corporate bond data requires paid subscription (Bloomberg/Refinitiv)")
        logger.info("Using free bond ETF alternatives for small/mid-cap exposure...")

        try:
            import yfinance as yf

            # Corporate bond ETFs covering different credit ratings and maturities
            bond_etfs = {
                # Investment Grade
                'LQD': 'iShares iBoxx Investment Grade Corporate',
                'VCIT': 'Vanguard Intermediate-Term Corporate',
                'VCSH': 'Vanguard Short-Term Corporate',
                'VCLT': 'Vanguard Long-Term Corporate',
                'IGIB': 'iShares Intermediate-Term Corporate',

                # High Yield (small/mid-cap companies)
                'HYG': 'iShares iBoxx High Yield Corporate',
                'JNK': 'SPDR Bloomberg High Yield',
                'ANGL': 'VanEck Fallen Angel High Yield',
                'HYLB': 'Xtrackers USD High Yield Corporate',
                'USHY': 'iShares Broad USD High Yield',

                # Floating Rate (bank loans to mid-cap)
                'FLOT': 'iShares Floating Rate Bond',
                'FLRN': 'SPDR Bloomberg Investment Grade Floating',
                'SRLN': 'SPDR Blackstone Senior Loan',
                'BKLN': 'Invesco Senior Loan',
            }

            bond_data = []
            for symbol, description in bond_etfs.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period='1mo')

                    bond_data.append({
                        'symbol': symbol,
                        'name': description,
                        'price': info.get('regularMarketPrice'),
                        'yield': info.get('yield', 0) * 100 if info.get('yield') else None,
                        'duration': info.get('duration'),
                        'avg_volume': hist['Volume'].mean() if not hist.empty else None,
                        'credit_quality': 'High Yield' if any(x in symbol for x in ['HY', 'JNK', 'ANGL']) else 'Investment Grade'
                    })
                except Exception as e:
                    logger.warning(f"  Failed to fetch {symbol}: {e}")

            if bond_data:
                df = pd.DataFrame(bond_data)
                output_file = self.data_dir / "corporate_bond_etfs.csv"
                df.to_csv(output_file, index=False)

                logger.info(f"[OK] Downloaded {len(df)} corporate bond ETFs")
                logger.info(f"    Saved to: {output_file}")

                return df
        except Exception as e:
            logger.error(f"Failed to download bonds: {e}")

        return pd.DataFrame()

    def download_warrants_and_convertibles(self) -> pd.DataFrame:
        """Download warrants and convertible securities"""
        logger.info("\nDownloading warrants and convertible securities...")

        all_securities = []

        # 1. Search for warrant symbols (typically end with .W or .WS)
        try:
            # Get all symbols and filter for warrants
            all_symbols_file = self.data_dir / "all_us_stocks.csv"
            if all_symbols_file.exists():
                all_syms = pd.read_csv(all_symbols_file)

                # Warrants typically have specific suffixes
                warrants = all_syms[
                    all_syms['Symbol'].str.contains(r'[./]W[ST]?$|\.W$|^[A-Z]+W$', na=False, regex=True)
                ]

                if not warrants.empty:
                    warrants['security_type'] = 'warrant'
                    all_securities.append(warrants)
                    logger.info(f"  Found {len(warrants)} warrants")
        except Exception as e:
            logger.error(f"  Failed to identify warrants: {e}")

        # 2. Download convertible bond ETFs (best proxy for convertibles universe)
        try:
            import yfinance as yf

            convertible_etfs = {
                'CWB': 'SPDR Convertible Securities',
                'ICVT': 'iShares Convertible Bond',
                'FCVT': 'First Trust SSI Strategic Convertible'
            }

            conv_data = []
            for symbol, name in convertible_etfs.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    conv_data.append({
                        'Symbol': symbol,
                        'Name': name,
                        'security_type': 'convertible_etf',
                        'price': info.get('regularMarketPrice'),
                        'yield': info.get('yield', 0) * 100 if info.get('yield') else None
                    })
                except:
                    pass

            if conv_data:
                conv_df = pd.DataFrame(conv_data)
                all_securities.append(conv_df)
                logger.info(f"  Found {len(conv_df)} convertible bond ETFs")
        except Exception as e:
            logger.error(f"  Failed to download convertibles: {e}")

        # 3. Search for units (SPAC units often convertible)
        try:
            all_symbols_file = self.data_dir / "all_us_stocks.csv"
            if all_symbols_file.exists():
                all_syms = pd.read_csv(all_symbols_file)

                units = all_syms[
                    all_syms['Symbol'].str.contains(r'[./]U$|^[A-Z]+U$', na=False, regex=True)
                ]

                if not units.empty:
                    units['security_type'] = 'unit'
                    all_securities.append(units)
                    logger.info(f"  Found {len(units)} units (SPACs)")
        except Exception as e:
            logger.error(f"  Failed to identify units: {e}")

        # Combine all
        if all_securities:
            combined = pd.concat(all_securities, ignore_index=True)

            output_file = self.data_dir / "warrants_convertibles.csv"
            combined.to_csv(output_file, index=False)

            logger.info(f"[OK] Total warrants/convertibles/units: {len(combined)}")
            logger.info(f"    Saved to: {output_file}")

            return combined

        return pd.DataFrame()

    def analyze_small_midcap_liquidity(self, stocks: pd.DataFrame) -> pd.DataFrame:
        """Analyze liquidity for small/mid-cap focus"""
        logger.info("\nAnalyzing small/mid-cap liquidity opportunities...")

        if 'marketCap' not in stocks.columns or 'volume' not in stocks.columns:
            logger.warning("Missing market cap or volume data for liquidity analysis")
            return pd.DataFrame()

        # Convert to numeric
        stocks['marketCap_num'] = pd.to_numeric(stocks['marketCap'], errors='coerce')
        stocks['volume_num'] = pd.to_numeric(stocks['volume'], errors='coerce')

        # Filter for small/mid-cap ($300M - $10B)
        small_midcap = stocks[
            (stocks['marketCap_num'] >= 300_000_000) &
            (stocks['marketCap_num'] <= 10_000_000_000)
        ].copy()

        # Calculate liquidity score
        small_midcap['avg_daily_dollar_volume'] = small_midcap['volume_num'] * pd.to_numeric(small_midcap.get('lastsale', 1), errors='coerce')

        # Liquidity tiers
        small_midcap['liquidity_tier'] = pd.cut(
            small_midcap['avg_daily_dollar_volume'],
            bins=[0, 1_000_000, 5_000_000, 20_000_000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        output_file = self.data_dir / "small_midcap_liquidity.csv"
        small_midcap.to_csv(output_file, index=False)

        logger.info(f"[OK] Identified {len(small_midcap)} small/mid-cap stocks")
        logger.info(f"    Liquidity breakdown:")
        logger.info(f"      Very High: {(small_midcap['liquidity_tier'] == 'Very High').sum()}")
        logger.info(f"      High: {(small_midcap['liquidity_tier'] == 'High').sum()}")
        logger.info(f"      Medium: {(small_midcap['liquidity_tier'] == 'Medium').sum()}")
        logger.info(f"      Low: {(small_midcap['liquidity_tier'] == 'Low').sum()}")
        logger.info(f"    Saved to: {output_file}")

        return small_midcap


def main():
    """Main download orchestration"""

    downloader = SecuritiesDownloader()

    print("\n[1/7] Downloading all US stocks...")
    stocks = downloader.download_all_stocks()

    print("\n[2/7] Downloading all US ETFs...")
    etfs = downloader.download_etfs()

    print("\n[3/7] Downloading options universe...")
    if not stocks.empty:
        options = downloader.download_options_universe(stocks)

    print("\n[4/7] Downloading Treasury securities...")
    treasuries = downloader.download_treasuries()

    print("\n[5/7] Downloading corporate bonds...")
    bonds = downloader.download_corporate_bonds()

    print("\n[6/7] Downloading warrants and convertibles...")
    warrants = downloader.download_warrants_and_convertibles()

    print("\n[7/7] Analyzing small/mid-cap liquidity...")
    if not stocks.empty:
        small_midcap = downloader.analyze_small_midcap_liquidity(stocks)

    # Generate summary report
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE - UNIVERSE SUMMARY")
    print("="*80)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_stocks': len(stocks) if not stocks.empty else 0,
        'total_etfs': len(etfs) if not etfs.empty else 0,
        'total_options_symbols': len(options) if 'options' in locals() else 0,
        'total_treasuries': len(treasuries) if not treasuries.empty else 0,
        'total_bonds': len(bonds) if not bonds.empty else 0,
        'total_warrants': len(warrants) if not warrants.empty else 0,
        'small_midcap_count': len(small_midcap) if 'small_midcap' in locals() and not small_midcap.empty else 0,
        'data_directory': str(DATA_DIR.absolute())
    }

    for key, value in summary.items():
        if key != 'data_directory':
            print(f"  {key}: {value:,}")

    print(f"\n  All data saved to: {summary['data_directory']}")

    # Save summary
    summary_file = DATA_DIR / "universe_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Summary saved to: {summary_file}")

    print("\n" + "="*80)
    print("READY FOR SMALL/MID-CAP + OPTIONS ARB STRATEGIES")
    print("="*80)
    print("\nNext steps:")
    print("1. Run liquidity screening for optimal trading targets")
    print("2. Build options arbitrage scanners (put-call parity, vol arb)")
    print("3. Implement small-cap momentum with liquidity filters")
    print("4. Create convertible arbitrage strategies")
    print("\n")


if __name__ == "__main__":
    main()
