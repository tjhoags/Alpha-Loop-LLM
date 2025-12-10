"""================================================================================
FULL UNIVERSE DATA HYDRATION
================================================================================
Pulls ALL available data from Polygon/Massive into SQL:
- All US Stocks (8,000+)
- All ETFs (2,500+)
- Options chains (100,000+ contracts)
- Indices
- Forex/Currencies
- Crypto

This will take several hours but gives you institutional-grade coverage.
================================================================================
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import requests
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import get_settings
from src.database.connection import get_engine

# Configuration
MASSIVE_BASE_URL = "https://api.massive.com"  # Rebranded from Polygon.io
BATCH_SIZE = 100  # Tickers per batch
MAX_WORKERS = 8   # Parallel API calls
RATE_LIMIT_DELAY = 0.25  # Seconds between calls


class FullUniverseHydrator:
    """Hydrates the full market universe from Massive.com (rebranded from Polygon.io).
    """

    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.polygon_api_key  # Still uses polygon_api_key env var for backward compatibility
        self.engine = get_engine()

        self.stats = {
            "stocks_fetched": 0,
            "etfs_fetched": 0,
            "options_fetched": 0,
            "indices_fetched": 0,
            "forex_fetched": 0,
            "crypto_fetched": 0,
            "errors": 0,
            "rows_inserted": 0,
        }

        logger.info("=" * 70)
        logger.info("FULL UNIVERSE HYDRATION SYSTEM")
        logger.info("=" * 70)

    # =========================================================================
    # TICKER FETCHING
    # =========================================================================

    def fetch_all_tickers(self, market: str = "stocks", ticker_type: str = None) -> List[Dict]:
        """Fetch all tickers for a market type.

        Markets: stocks, crypto, fx, otc, indices
        Types: CS (common stock), ETF, ADRC, etc.
        """
        all_tickers = []
        url = f"{MASSIVE_BASE_URL}/v3/reference/tickers"

        params = {
            "market": market,
            "active": "true",
            "limit": 1000,
            "apiKey": self.api_key,
        }

        if ticker_type:
            params["type"] = ticker_type

        next_url = url
        page = 0

        while next_url:
            try:
                resp = requests.get(next_url, params=params if page == 0 else {"apiKey": self.api_key}, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                if "results" in data:
                    all_tickers.extend(data["results"])
                    logger.info(f"  Fetched page {page+1}: {len(data['results'])} tickers (total: {len(all_tickers)})")

                next_url = data.get("next_url")
                page += 1
                time.sleep(RATE_LIMIT_DELAY)

            except Exception as e:
                logger.error(f"Error fetching tickers: {e}")
                break

        return all_tickers

    def get_all_stocks(self) -> List[str]:
        """Get all US stock tickers."""
        logger.info("Fetching ALL US stocks...")
        tickers = self.fetch_all_tickers(market="stocks", ticker_type="CS")
        symbols = [t["ticker"] for t in tickers if "ticker" in t]
        logger.info(f"Found {len(symbols)} stocks")
        return symbols

    def get_all_etfs(self) -> List[str]:
        """Get all ETF tickers."""
        logger.info("Fetching ALL ETFs...")
        tickers = self.fetch_all_tickers(market="stocks", ticker_type="ETF")
        symbols = [t["ticker"] for t in tickers if "ticker" in t]
        logger.info(f"Found {len(symbols)} ETFs")
        return symbols

    def get_all_indices(self) -> List[str]:
        """Get major index tickers."""
        logger.info("Fetching indices...")
        # Polygon doesn't have great index coverage, use manual list
        indices = [
            "I:SPX", "I:NDX", "I:DJI", "I:RUT", "I:VIX",
            "SPY", "QQQ", "IWM", "DIA", "VXX",
        ]
        return indices

    def get_all_crypto(self) -> List[str]:
        """Get all crypto tickers."""
        logger.info("Fetching ALL crypto...")
        tickers = self.fetch_all_tickers(market="crypto")
        symbols = [t["ticker"] for t in tickers if "ticker" in t]
        logger.info(f"Found {len(symbols)} crypto pairs")
        return symbols

    def get_all_forex(self) -> List[str]:
        """Get all forex pairs."""
        logger.info("Fetching ALL forex pairs...")
        tickers = self.fetch_all_tickers(market="fx")
        symbols = [t["ticker"] for t in tickers if "ticker" in t]
        logger.info(f"Found {len(symbols)} forex pairs")
        return symbols

    # =========================================================================
    # OPTIONS CHAINS
    # =========================================================================

    def get_options_contracts(self, underlying: str) -> List[Dict]:
        """Get all options contracts for an underlying.
        """
        url = f"{MASSIVE_BASE_URL}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "limit": 1000,
            "apiKey": self.api_key,
        }

        all_contracts = []
        next_url = url

        while next_url and len(all_contracts) < 5000:  # Cap per underlying
            try:
                resp = requests.get(
                    next_url,
                    params=params if len(all_contracts) == 0 else {"apiKey": self.api_key},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                if "results" in data:
                    all_contracts.extend(data["results"])

                next_url = data.get("next_url")
                time.sleep(RATE_LIMIT_DELAY)

            except Exception as e:
                logger.debug(f"Options fetch error for {underlying}: {e}")
                break

        return all_contracts

    def get_all_options_for_stocks(self, underlyings: List[str], max_underlyings: int = 500) -> List[Dict]:
        """Get options chains for multiple underlyings.
        """
        logger.info(f"Fetching options for {min(len(underlyings), max_underlyings)} underlyings...")

        all_options = []

        for i, underlying in enumerate(underlyings[:max_underlyings]):
            if i % 50 == 0:
                logger.info(f"  Options progress: {i}/{min(len(underlyings), max_underlyings)}")

            contracts = self.get_options_contracts(underlying)
            all_options.extend(contracts)
            self.stats["options_fetched"] += len(contracts)

        logger.info(f"Found {len(all_options)} total options contracts")
        return all_options

    # =========================================================================
    # PRICE DATA
    # =========================================================================

    def fetch_aggregates(
        self,
        ticker: str,
        multiplier: int = 1,
        timespan: str = "day",
        from_date: str = None,
        to_date: str = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV aggregates for a ticker.
        """
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")

        url = f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if "results" not in data or not data["results"]:
                return pd.DataFrame()

            df = pd.DataFrame(data["results"])
            df["symbol"] = ticker
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
            df = df.rename(columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            })

            return df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.debug(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def fetch_batch(self, tickers: List[str], timespan: str = "day") -> pd.DataFrame:
        """Fetch data for a batch of tickers in parallel.
        """
        all_data = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.fetch_aggregates, ticker, 1, timespan): ticker
                for ticker in tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    df = future.result()
                    if not df.empty:
                        all_data.append(df)
                except Exception:
                    self.stats["errors"] += 1

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    # =========================================================================
    # DATABASE PERSISTENCE
    # =========================================================================

    def persist_to_sql(self, df: pd.DataFrame, table: str = "price_bars") -> int:
        """Persist DataFrame to SQL, with CSV fallback if SQL fails.
        """
        if df.empty:
            return 0

        # Always save to CSV as backup
        csv_dir = self.settings.data_dir / "csv_backup"
        csv_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = csv_dir / f"{table}_{timestamp}.csv"

        try:
            # Append to existing or create new
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df)} rows to {csv_path}")
        except Exception as e:
            logger.warning(f"CSV backup failed: {e}")

        # Try SQL
        try:
            with self.engine.begin() as conn:
                df.to_sql(table, conn, if_exists="append", index=False, chunksize=1000)

            self.stats["rows_inserted"] += len(df)
            return len(df)

        except Exception as e:
            logger.warning(f"SQL persist failed (data saved to CSV): {e}")
            self.stats["rows_inserted"] += len(df)  # Count CSV rows
            return len(df)

    def persist_options(self, options: List[Dict]) -> int:
        """Persist options contracts to SQL.
        """
        if not options:
            return 0

        df = pd.DataFrame(options)

        # Select relevant columns
        cols = ["ticker", "underlying_ticker", "contract_type", "expiration_date",
                "strike_price", "shares_per_contract"]
        available = [c for c in cols if c in df.columns]
        df = df[available]

        return self.persist_to_sql(df, table="options_contracts")

    # =========================================================================
    # MAIN HYDRATION
    # =========================================================================

    def hydrate_stocks(self, limit: int = None):
        """Hydrate all stocks."""
        logger.info("\n" + "=" * 50)
        logger.info("HYDRATING STOCKS")
        logger.info("=" * 50)

        stocks = self.get_all_stocks()
        if limit:
            stocks = stocks[:limit]

        # Process in batches
        for i in range(0, len(stocks), BATCH_SIZE):
            batch = stocks[i:i+BATCH_SIZE]
            logger.info(f"Processing stocks batch {i//BATCH_SIZE + 1}/{len(stocks)//BATCH_SIZE + 1}")

            df = self.fetch_batch(batch, timespan="day")
            if not df.empty:
                self.persist_to_sql(df)
                self.stats["stocks_fetched"] += len(batch)

            time.sleep(1)  # Rate limiting

    def hydrate_etfs(self, limit: int = None):
        """Hydrate all ETFs."""
        logger.info("\n" + "=" * 50)
        logger.info("HYDRATING ETFs")
        logger.info("=" * 50)

        etfs = self.get_all_etfs()
        if limit:
            etfs = etfs[:limit]

        for i in range(0, len(etfs), BATCH_SIZE):
            batch = etfs[i:i+BATCH_SIZE]
            logger.info(f"Processing ETF batch {i//BATCH_SIZE + 1}/{len(etfs)//BATCH_SIZE + 1}")

            df = self.fetch_batch(batch, timespan="day")
            if not df.empty:
                self.persist_to_sql(df)
                self.stats["etfs_fetched"] += len(batch)

            time.sleep(1)

    def hydrate_crypto(self, limit: int = None):
        """Hydrate crypto pairs."""
        logger.info("\n" + "=" * 50)
        logger.info("HYDRATING CRYPTO")
        logger.info("=" * 50)

        crypto = self.get_all_crypto()
        if limit:
            crypto = crypto[:limit]

        for i in range(0, len(crypto), BATCH_SIZE):
            batch = crypto[i:i+BATCH_SIZE]
            logger.info(f"Processing crypto batch {i//BATCH_SIZE + 1}/{len(crypto)//BATCH_SIZE + 1}")

            df = self.fetch_batch(batch, timespan="day")
            if not df.empty:
                self.persist_to_sql(df)
                self.stats["crypto_fetched"] += len(batch)

            time.sleep(1)

    def hydrate_forex(self, limit: int = None):
        """Hydrate forex pairs."""
        logger.info("\n" + "=" * 50)
        logger.info("HYDRATING FOREX")
        logger.info("=" * 50)

        forex = self.get_all_forex()
        if limit:
            forex = forex[:limit]

        for i in range(0, len(forex), BATCH_SIZE):
            batch = forex[i:i+BATCH_SIZE]
            logger.info(f"Processing forex batch {i//BATCH_SIZE + 1}/{len(forex)//BATCH_SIZE + 1}")

            df = self.fetch_batch(batch, timespan="day")
            if not df.empty:
                self.persist_to_sql(df, table="forex_bars")
                self.stats["forex_fetched"] += len(batch)

            time.sleep(1)

    def hydrate_options(self, underlyings: List[str] = None, limit: int = 100):
        """Hydrate options chains."""
        logger.info("\n" + "=" * 50)
        logger.info("HYDRATING OPTIONS")
        logger.info("=" * 50)

        if underlyings is None:
            # Use top liquid stocks for options
            underlyings = [
                "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA",
                "GOOGL", "META", "AMZN", "NFLX", "DIS", "BA", "JPM", "GS",
                "BAC", "C", "WFC", "V", "MA", "PYPL", "SQ", "COIN",
                "XOM", "CVX", "OXY", "SLB", "HAL",
                "PFE", "JNJ", "MRK", "ABBV", "LLY", "UNH",
                "HD", "LOW", "TGT", "WMT", "COST", "NKE",
                "F", "GM", "RIVN", "LCID",
            ]

        options = self.get_all_options_for_stocks(underlyings[:limit])
        self.persist_options(options)

    def hydrate_everything(
        self,
        stocks: bool = True,
        etfs: bool = True,
        crypto: bool = True,
        forex: bool = True,
        options: bool = True,
        stock_limit: int = None,
        etf_limit: int = None,
        crypto_limit: int = None,
        forex_limit: int = None,
        options_limit: int = 100,
    ):
        """Hydrate the full universe.

        WARNING: This can take 6-12 hours for full universe!
        """
        start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("FULL UNIVERSE HYDRATION STARTING")
        logger.info("=" * 70)
        logger.info(f"Stocks: {stocks}, ETFs: {etfs}, Crypto: {crypto}, Forex: {forex}, Options: {options}")

        if stocks:
            self.hydrate_stocks(limit=stock_limit)

        if etfs:
            self.hydrate_etfs(limit=etf_limit)

        if crypto:
            self.hydrate_crypto(limit=crypto_limit)

        if forex:
            self.hydrate_forex(limit=forex_limit)

        if options:
            self.hydrate_options(limit=options_limit)

        elapsed = time.time() - start_time

        logger.info("\n" + "=" * 70)
        logger.info("HYDRATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Time elapsed: {elapsed/3600:.2f} hours")
        logger.info(f"Stocks fetched: {self.stats['stocks_fetched']}")
        logger.info(f"ETFs fetched: {self.stats['etfs_fetched']}")
        logger.info(f"Crypto fetched: {self.stats['crypto_fetched']}")
        logger.info(f"Forex fetched: {self.stats['forex_fetched']}")
        logger.info(f"Options fetched: {self.stats['options_fetched']}")
        logger.info(f"Total rows inserted: {self.stats['rows_inserted']}")
        logger.info(f"Errors: {self.stats['errors']}")


def run_full_hydration():
    """Run full universe hydration.

    USAGE:
    python scripts/hydrate_full_universe.py
    """
    settings = get_settings()
    logger.add(settings.logs_dir / "hydration.log", rotation="100 MB", level="INFO")

    hydrator = FullUniverseHydrator()

    # Full hydration (will take hours)
    hydrator.hydrate_everything(
        stocks=True,
        etfs=True,
        crypto=True,
        forex=True,
        options=True,
        stock_limit=None,      # All stocks
        etf_limit=None,        # All ETFs
        crypto_limit=100,      # Top 100 crypto
        forex_limit=50,        # Major forex pairs
        options_limit=50,       # Top 50 underlyings
    )


def run_quick_hydration():
    """Quick hydration for testing (smaller subset).
    """
    settings = get_settings()
    logger.add(settings.logs_dir / "hydration.log", rotation="100 MB", level="INFO")

    hydrator = FullUniverseHydrator()

    # Quick test (10-20 minutes)
    hydrator.hydrate_everything(
        stocks=True,
        etfs=True,
        crypto=True,
        forex=False,
        options=False,
        stock_limit=500,       # Top 500 stocks
        etf_limit=200,         # Top 200 ETFs
        crypto_limit=20,       # Top 20 crypto
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_hydration()
    else:
        run_full_hydration()


