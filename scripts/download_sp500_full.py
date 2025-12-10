#!/usr/bin/env python
"""
Download Full S&P 500 Historical Data
10 years of OHLCV data for all S&P 500 stocks

Author: Tom Hogan | Alpha Loop Capital, LLC
Date: 2025-12-09
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_sp500_full():
    """Download complete S&P 500 dataset"""

    # Load S&P 500 constituents
    data_dir = Path("data/datasets")
    constituents = pd.read_csv(data_dir / "sp500.csv")

    symbols = constituents['Symbol'].tolist()
    logger.info(f"Downloading {len(symbols)} S&P 500 stocks...")

    # Date range: 10 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)

    all_data = []
    failed = []

    # Download in batches
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        logger.info(f"Batch {i//batch_size + 1}: {len(batch)} symbols")

        try:
            # Download batch
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                group_by='ticker',
                auto_adjust=True,
                threads=True
            )

            # Process each symbol
            for symbol in batch:
                try:
                    if len(batch) == 1:
                        symbol_data = data
                    else:
                        symbol_data = data[symbol] if symbol in data.columns.levels[0] else None

                    if symbol_data is not None and not symbol_data.empty:
                        symbol_data = symbol_data.copy()
                        symbol_data['Symbol'] = symbol
                        all_data.append(symbol_data)
                    else:
                        failed.append(symbol)

                except Exception as e:
                    logger.warning(f"Failed {symbol}: {e}")
                    failed.append(symbol)

        except Exception as e:
            logger.error(f"Batch {i//batch_size + 1} failed: {e}")
            failed.extend(batch)

    # Combine all data
    if all_data:
        combined = pd.concat(all_data)

        # Save
        output_file = data_dir / "sp500_full_10yr.csv"
        combined.to_csv(output_file)

        logger.info(f"SUCCESS: {len(combined)} rows saved to {output_file}")
        logger.info(f"Symbols: {combined['Symbol'].nunique()}")
        logger.info(f"Failed: {len(failed)} symbols")

        return combined
    else:
        logger.error("No data downloaded!")
        return None


if __name__ == "__main__":
    download_sp500_full()
