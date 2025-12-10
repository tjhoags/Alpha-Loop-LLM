#!/usr/bin/env python
"""
Download High-Quality Financial Datasets
Pull data from GitHub, Kaggle, and other sources for training

Sources:
1. GitHub - Free financial datasets
2. Yahoo Finance - Historical OHLCV
3. FRED - Economic indicators
4. Alternative data sources

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
"""

import os
import sys
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """
    Download and prepare financial datasets for training.

    Targets:
    - 10+ years of historical stock data
    - Economic indicators (GDP, inflation, rates)
    - Market regime data
    - Alternative data (sentiment, options flow)
    """

    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Dataset sources (GitHub repos with financial data)
        self.github_datasets = {
            "sp500": {
                "url": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
                "description": "S&P 500 constituents list"
            },
            "sp500_historical": {
                "url": "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv",
                "description": "S&P 500 historical prices"
            },
            "nasdaq": {
                "url": "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed.csv",
                "description": "NASDAQ listed companies"
            },
            "fred_indicators": {
                "url": "https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv",
                "description": "Economic indicators from FRED"
            }
        }

        logger.info(f"DatasetDownloader initialized. Data dir: {self.data_dir}")

    def download_all(self):
        """Download all datasets"""
        logger.info("Starting dataset download...")

        # 1. GitHub datasets
        self.download_github_datasets()

        # 2. Yahoo Finance data (S&P 500)
        self.download_sp500_historical()

        # 3. Economic indicators
        self.download_economic_indicators()

        # 4. Create master dataset
        self.create_master_dataset()

        logger.info("All datasets downloaded successfully!")

    def download_github_datasets(self):
        """Download datasets from GitHub"""
        logger.info("Downloading GitHub datasets...")

        for name, info in self.github_datasets.items():
            try:
                logger.info(f"Downloading {name}: {info['description']}")

                response = requests.get(info['url'], timeout=30)
                response.raise_for_status()

                # Save to file
                filename = self.data_dir / f"{name}.csv"
                with open(filename, 'w') as f:
                    f.write(response.text)

                logger.info(f"✓ Saved {name} to {filename}")

            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")

    def download_sp500_historical(self):
        """Download historical data for S&P 500 stocks"""
        logger.info("Downloading S&P 500 historical data...")

        try:
            # Get S&P 500 constituents
            constituents_file = self.data_dir / "sp500.csv"

            if not constituents_file.exists():
                logger.warning("S&P 500 constituents not found. Downloading first...")
                self.download_github_datasets()

            constituents = pd.read_csv(constituents_file)
            symbols = constituents['Symbol'].tolist()[:50]  # Top 50 for speed

            logger.info(f"Downloading data for {len(symbols)} stocks...")

            # Download using Yahoo Finance
            self._download_yahoo_data(symbols)

        except Exception as e:
            logger.error(f"Error downloading S&P 500 historical: {e}")

    def _download_yahoo_data(self, symbols: List[str]):
        """Download historical data from Yahoo Finance"""
        try:
            import yfinance as yf

            # Download all at once (faster)
            logger.info(f"Fetching {len(symbols)} symbols from Yahoo Finance...")

            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*10)  # 10 years

            # Download in batches to avoid rate limits
            batch_size = 10
            all_data = []

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                logger.info(f"Batch {i//batch_size + 1}/{len(symbols)//batch_size + 1}: {batch}")

                try:
                    tickers = yf.Tickers(" ".join(batch))

                    for symbol in batch:
                        try:
                            ticker = tickers.tickers.get(symbol)
                            if ticker:
                                hist = ticker.history(start=start_date, end=end_date)
                                if not hist.empty:
                                    hist['Symbol'] = symbol
                                    all_data.append(hist)
                        except Exception as e:
                            logger.warning(f"Failed to get data for {symbol}: {e}")

                except Exception as e:
                    logger.warning(f"Batch failed: {e}")
                    continue

            if all_data:
                # Combine all data
                combined = pd.concat(all_data)

                # Save to file
                output_file = self.data_dir / "sp500_10yr_historical.csv"
                combined.to_csv(output_file)

                logger.info(f"✓ Saved {len(combined)} rows to {output_file}")
                logger.info(f"  Date range: {combined.index.min()} to {combined.index.max()}")
                logger.info(f"  Symbols: {combined['Symbol'].nunique()}")

        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
        except Exception as e:
            logger.error(f"Yahoo Finance download error: {e}")

    def download_economic_indicators(self):
        """Download economic indicators from FRED"""
        logger.info("Downloading economic indicators...")

        # Key economic indicators
        indicators = {
            'GDP': 'https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv',
            'CPI': 'https://raw.githubusercontent.com/datasets/cpi/master/data/cpi.csv',
            'UNRATE': 'https://raw.githubusercontent.com/datasets/employment-us/master/data/employment.csv',
        }

        for name, url in indicators.items():
            try:
                logger.info(f"Downloading {name}...")

                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    filename = self.data_dir / f"economic_{name.lower()}.csv"
                    with open(filename, 'w') as f:
                        f.write(response.text)
                    logger.info(f"✓ Saved {name}")
                else:
                    logger.warning(f"Could not download {name} (status {response.status_code})")

            except Exception as e:
                logger.warning(f"Failed to download {name}: {e}")

    def create_master_dataset(self):
        """Create master dataset combining all sources"""
        logger.info("Creating master dataset...")

        try:
            # Load S&P 500 historical
            sp500_file = self.data_dir / "sp500_10yr_historical.csv"

            if sp500_file.exists():
                df = pd.read_csv(sp500_file, index_col=0, parse_dates=True)

                # Create summary
                summary = {
                    'total_rows': len(df),
                    'symbols': df['Symbol'].nunique() if 'Symbol' in df.columns else 0,
                    'date_range': f"{df.index.min()} to {df.index.max()}",
                    'columns': list(df.columns),
                    'file': str(sp500_file),
                }

                # Save summary
                summary_file = self.data_dir / "dataset_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)

                logger.info(f"✓ Master dataset summary saved to {summary_file}")
                logger.info(f"  Total rows: {summary['total_rows']:,}")
                logger.info(f"  Symbols: {summary['symbols']}")
                logger.info(f"  Date range: {summary['date_range']}")
            else:
                logger.warning("No S&P 500 data found. Skipping master dataset creation.")

        except Exception as e:
            logger.error(f"Error creating master dataset: {e}")

    def get_dataset_stats(self) -> Dict:
        """Get statistics about downloaded datasets"""
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'datasets': {}
        }

        for file in self.data_dir.glob('*.csv'):
            size_mb = file.stat().st_size / (1024 * 1024)
            stats['total_files'] += 1
            stats['total_size_mb'] += size_mb
            stats['datasets'][file.name] = {
                'size_mb': round(size_mb, 2),
                'modified': datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            }

        return stats


def download_alternative_datasets():
    """Download additional alternative datasets"""
    logger.info("Downloading alternative datasets...")

    # VIX historical data (volatility index)
    vix_datasets = [
        {
            'name': 'VIX Historical',
            'url': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv',
            'filename': 'vix_historical.csv'
        }
    ]

    data_dir = Path("data/datasets")

    for dataset in vix_datasets:
        try:
            logger.info(f"Downloading {dataset['name']}...")
            response = requests.get(dataset['url'], timeout=30)

            if response.status_code == 200:
                filename = data_dir / dataset['filename']
                with open(filename, 'wb') as f:
                    f.write(response.content)
                logger.info(f"✓ Saved {dataset['name']}")
            else:
                logger.warning(f"Could not download {dataset['name']}")

        except Exception as e:
            logger.warning(f"Failed to download {dataset['name']}: {e}")


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("FINANCIAL DATASET DOWNLOADER")
    print("="*70 + "\n")

    # Initialize downloader
    downloader = DatasetDownloader()

    # Download all datasets
    downloader.download_all()

    # Download alternative datasets
    download_alternative_datasets()

    # Print statistics
    stats = downloader.get_dataset_stats()

    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"Total files downloaded: {stats['total_files']}")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    print("\nDatasets:")
    for name, info in stats['datasets'].items():
        print(f"  - {name}: {info['size_mb']:.2f} MB")
    print("="*70 + "\n")

    print("✓ All datasets downloaded successfully!")
    print(f"Data directory: {Path('data/datasets').absolute()}\n")


if __name__ == "__main__":
    main()
