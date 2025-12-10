#!/usr/bin/env python
"""
Bulk Historical Price Data Downloader
Downloads OHLCV history for ALL US stocks in universe

WARNING: This will download 50-100GB of data
Expected time: 4-6 hours for full universe

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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Data directories
UNIVERSE_DIR = Path("data/universe")
PRICES_DIR = Path("data/historical_prices")
PRICES_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("BULK HISTORICAL PRICE DATA DOWNLOAD")
print("="*80)
print("\nWARNING: This will download 50-100GB of historical price data")
print("Expected time: 4-6 hours for ~7,000 stocks")
print("="*80 + "\n")


class BulkPriceDownloader:
    """Download historical prices for entire universe"""

    def __init__(
        self,
        years_of_history: int = 5,
        batch_size: int = 100,
        max_workers: int = 20
    ):
        self.years_of_history = years_of_history
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.start_date = datetime.now() - timedelta(days=365 * years_of_history)
        self.end_date = datetime.now()

        self.downloaded = 0
        self.failed = 0
        self.total = 0

    def download_single_stock(
        self,
        symbol: str,
        force_redownload: bool = False
    ) -> bool:
        """Download historical data for a single stock"""

        # Check if already downloaded
        output_file = PRICES_DIR / f"{symbol}.csv"
        if output_file.exists() and not force_redownload:
            # Check if file is recent and has data
            if output_file.stat().st_size > 1000:  # At least 1KB
                return True

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                actions=True  # Include dividends and splits
            )

            if not hist.empty and len(hist) > 20:  # At least 20 days of data
                # Save to CSV
                hist.to_csv(output_file)
                return True
            else:
                logger.warning(f"  {symbol}: No data available")
                return False

        except Exception as e:
            logger.error(f"  {symbol}: Failed - {str(e)[:50]}")
            return False

    def download_batch(
        self,
        symbols: List[str],
        batch_num: int,
        total_batches: int
    ) -> Dict[str, bool]:
        """Download a batch of stocks in parallel"""

        logger.info(f"\nBatch {batch_num}/{total_batches}: Processing {len(symbols)} symbols...")

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_single_stock, sym): sym
                for sym in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    success = future.result()
                    results[symbol] = success

                    if success:
                        self.downloaded += 1
                    else:
                        self.failed += 1

                except Exception as e:
                    logger.error(f"  {symbol}: Exception - {e}")
                    results[symbol] = False
                    self.failed += 1

                # Progress update
                completed = self.downloaded + self.failed
                if completed % 50 == 0:
                    logger.info(
                        f"  Progress: {completed}/{self.total} "
                        f"({self.downloaded} success, {self.failed} failed)"
                    )

        return results

    def download_all_stocks(self, universe_file: Path) -> pd.DataFrame:
        """Download historical prices for all stocks in universe"""

        # Load universe
        logger.info(f"Loading universe from {universe_file}...")
        universe = pd.read_csv(universe_file)

        symbols = universe['Symbol'].tolist()
        self.total = len(symbols)

        logger.info(f"Universe: {self.total} stocks")
        logger.info(f"Time period: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Batch size: {self.batch_size}, Workers: {self.max_workers}")

        # Split into batches
        batches = [
            symbols[i:i + self.batch_size]
            for i in range(0, len(symbols), self.batch_size)
        ]

        total_batches = len(batches)
        logger.info(f"Total batches: {total_batches}\n")

        # Download each batch
        start_time = time.time()
        all_results = {}

        for i, batch in enumerate(batches, 1):
            batch_results = self.download_batch(batch, i, total_batches)
            all_results.update(batch_results)

            # Estimate time remaining
            elapsed = time.time() - start_time
            rate = (self.downloaded + self.failed) / elapsed if elapsed > 0 else 0
            remaining = (self.total - (self.downloaded + self.failed)) / rate if rate > 0 else 0

            logger.info(
                f"\nBatch {i}/{total_batches} complete. "
                f"Total: {self.downloaded} success, {self.failed} failed. "
                f"Est. time remaining: {remaining/60:.1f} minutes"
            )

            # Small delay between batches to avoid rate limits
            if i < total_batches:
                time.sleep(2)

        # Summary
        elapsed_minutes = (time.time() - start_time) / 60

        logger.info("\n" + "="*80)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("="*80)
        logger.info(f"Total symbols processed: {self.total}")
        logger.info(f"Successfully downloaded: {self.downloaded}")
        logger.info(f"Failed: {self.failed}")
        logger.info(f"Success rate: {self.downloaded/self.total*100:.1f}%")
        logger.info(f"Total time: {elapsed_minutes:.1f} minutes")
        logger.info(f"Download rate: {self.downloaded/elapsed_minutes:.1f} stocks/minute")

        # Create results dataframe
        results_df = pd.DataFrame([
            {'symbol': sym, 'success': success}
            for sym, success in all_results.items()
        ])

        results_df.to_csv(PRICES_DIR / "download_results.csv", index=False)
        logger.info(f"\nResults saved to: {PRICES_DIR / 'download_results.csv'}")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in PRICES_DIR.glob("*.csv") if f.is_file())
        logger.info(f"Total data downloaded: {total_size / (1024**3):.2f} GB")

        return results_df

    def download_priority_symbols(
        self,
        priority_list: List[str],
        description: str = "priority symbols"
    ) -> Dict[str, bool]:
        """Download a specific list of symbols (e.g., S&P 500, liquid mid-caps)"""

        logger.info(f"\nDownloading {len(priority_list)} {description}...")
        self.total = len(priority_list)

        results = self.download_batch(priority_list, 1, 1)

        logger.info(f"\nCompleted {description}:")
        logger.info(f"  Success: {sum(results.values())}")
        logger.info(f"  Failed: {len(results) - sum(results.values())}")

        return results


def download_sp500_prices():
    """Download S&P 500 stocks first (highest priority)"""

    logger.info("=" * 80)
    logger.info("STEP 1: S&P 500 Historical Prices (High Priority)")
    logger.info("=" * 80)

    try:
        # Get S&P 500 list
        import yfinance as yf

        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500_symbols = sp500['Symbol'].str.replace('.', '-').tolist()

        downloader = BulkPriceDownloader(years_of_history=5, max_workers=20)
        results = downloader.download_priority_symbols(sp500_symbols, "S&P 500 stocks")

        return results

    except Exception as e:
        logger.error(f"Failed to download S&P 500: {e}")
        return {}


def download_liquid_midcaps():
    """Download liquid mid-caps (good for small/mid-cap strategies)"""

    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Liquid Mid-Caps ($500M-$10B)")
    logger.info("=" * 80)

    # Filter universe for mid-caps
    universe_file = UNIVERSE_DIR / "all_us_stocks.csv"
    if not universe_file.exists():
        logger.error("Universe file not found!")
        return {}

    # For now, just download top symbols alphabetically as placeholder
    # In production, would filter by market cap and volume
    universe = pd.read_csv(universe_file)

    # Take symbols from different parts of alphabet for diversity
    sample_symbols = []
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        letter_symbols = universe[universe['Symbol'].str.startswith(letter)]['Symbol'].head(20).tolist()
        sample_symbols.extend(letter_symbols)

    downloader = BulkPriceDownloader(years_of_history=5, max_workers=20)
    results = downloader.download_priority_symbols(sample_symbols[:500], "liquid mid-caps sample")

    return results


def download_full_universe():
    """Download entire universe (all 7,000+ stocks)"""

    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: FULL UNIVERSE (ALL 7,000+ STOCKS)")
    logger.info("=" * 80)

    response = input("\nThis will download ~70GB and take 4-6 hours. Continue? (yes/no): ")

    if response.lower() != 'yes':
        logger.info("Skipping full universe download.")
        return

    universe_file = UNIVERSE_DIR / "all_us_stocks.csv"
    if not universe_file.exists():
        logger.error("Universe file not found!")
        return

    downloader = BulkPriceDownloader(years_of_history=5, batch_size=100, max_workers=20)
    results = downloader.download_all_stocks(universe_file)

    return results


def main():
    """Main download orchestration"""

    print("Download Options:")
    print("1. S&P 500 only (~500 stocks, ~2GB, 20 minutes)")
    print("2. S&P 500 + Liquid Mid-Caps (~1,000 stocks, ~4GB, 40 minutes)")
    print("3. Full Universe (7,000+ stocks, ~70GB, 4-6 hours)")
    print()

    choice = input("Select option (1/2/3): ").strip()

    if choice == '1':
        # S&P 500 only
        download_sp500_prices()

    elif choice == '2':
        # S&P 500 + liquid mid-caps
        download_sp500_prices()
        download_liquid_midcaps()

    elif choice == '3':
        # Full universe
        download_sp500_prices()
        download_liquid_midcaps()
        download_full_universe()

    else:
        logger.error("Invalid choice. Exiting.")
        return

    logger.info("\n" + "="*80)
    logger.info("HISTORICAL PRICE DOWNLOAD COMPLETE")
    logger.info("="*80)
    logger.info(f"\nData saved to: {PRICES_DIR.absolute()}")
    logger.info("\nNext steps:")
    logger.info("1. Run liquidity analysis on downloaded stocks")
    logger.info("2. Calculate technical indicators (momentum, RSI, etc.)")
    logger.info("3. Backtest strategies on full historical data")
    logger.info("4. Train ML models on complete price history")
    print()


if __name__ == "__main__":
    main()
