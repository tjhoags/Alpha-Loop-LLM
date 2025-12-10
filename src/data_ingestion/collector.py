"""================================================================================
DATA COLLECTOR - Multi-Source Market Data Collection
================================================================================
Collects market data from multiple sources in parallel:
- Polygon.io: Premium market data
- Alpha Vantage: Stock data
- Coinbase: Cryptocurrency data
- FRED: Macroeconomic indicators

Features:
- Parallel collection with ThreadPoolExecutor
- Rate limiting and retry logic
- Deduplication and data quality checks
================================================================================
"""

import concurrent.futures
import time
from typing import List

import pandas as pd
from loguru import logger

from src.config.settings import get_settings
from src.data_ingestion.sources.alpha_vantage import fetch_intraday as av_fetch
from src.data_ingestion.sources.coinbase import fetch_candles as coinbase_fetch
from src.data_ingestion.sources.fred import FredClient
from src.data_ingestion.sources.polygon import fetch_aggregates as polygon_fetch
from src.database.connection import get_engine

# Rate limiting settings
ALPHA_VANTAGE_DELAY = 12.5  # 5 calls/minute = 12 seconds between calls
MAX_WORKERS = 5  # Reduced from 10 to respect API limits


def collect_equities(symbols: List[str]) -> pd.DataFrame:
    """Collects data for all symbols with rate limiting.

    Args:
        symbols: List of stock ticker symbols

    Returns:
        DataFrame with combined OHLCV data from all sources
    """
    all_frames = []
    settings = get_settings()

    def process_symbol(sym: str) -> pd.DataFrame:
        frames = []
        try:
            logger.info(f"Collecting data for {sym}...")

            # Polygon: 1-minute bars (if API key configured)
            if settings.polygon_api_key:
                try:
                    df_poly = polygon_fetch(sym, timespan="minute", multiplier=1, lookback_days=730)
                    if df_poly is not None and not df_poly.empty:
                        df_poly["source"] = "polygon"
                        frames.append(df_poly)
                except Exception as e:
                    logger.warning(f"Polygon fetch failed for {sym}: {e}")

            # Alpha Vantage: 1-minute bars (with rate limiting)
            if settings.alpha_vantage_api_key:
                try:
                    time.sleep(ALPHA_VANTAGE_DELAY)  # Rate limit
                    df_av = av_fetch(sym, interval="1min")
                    if df_av is not None and not df_av.empty:
                        df_av["source"] = "alpha_vantage"
                        frames.append(df_av)
                except Exception as e:
                    logger.warning(f"Alpha Vantage fetch failed for {sym}: {e}")

            if frames:
                return pd.concat(frames, ignore_index=True)

        except Exception as exc:
            logger.error(f"Failed equity collection for {sym}: {exc}")

        return pd.DataFrame()

    # Use ThreadPool with limited workers to respect rate limits
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_symbol, symbols))

    for res in results:
        if res is not None and not res.empty:
            all_frames.append(res)

    if not all_frames:
        logger.warning("No equity data collected")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df.sort_values(["symbol", "timestamp"], inplace=True)
    df.drop_duplicates(subset=["symbol", "timestamp"], keep="last", inplace=True)

    logger.info(f"Collected {len(df)} total equity rows for {len(symbols)} symbols")
    return df


def collect_macro() -> pd.DataFrame:
    """Collect macroeconomic data from FRED."""
    settings = get_settings()
    if not settings.fred_api_key:
        logger.warning("FRED API key not configured, skipping macro data")
        return pd.DataFrame()

    try:
        fred = FredClient()
        df = fred.fetch_core_macro()
        logger.info(f"Collected {len(df)} macro indicator rows")
        return df
    except Exception as e:
        logger.error(f"Macro collection failed: {e}")
        return pd.DataFrame()


def collect_crypto(symbols: List[str] = None) -> pd.DataFrame:
    """Collect cryptocurrency data from Coinbase.

    Args:
        symbols: List of crypto symbols (default: ["BTC-USD", "ETH-USD"])

    Returns:
        DataFrame with crypto OHLCV data
    """
    settings = get_settings()
    if not settings.coinbase_api_key:
        logger.warning("Coinbase API key not configured, skipping crypto data")
        return pd.DataFrame()

    symbols = symbols or ["BTC-USD", "ETH-USD"]
    all_frames = []

    for symbol in symbols:
        try:
            logger.info(f"Collecting crypto data for {symbol}...")
            df = coinbase_fetch(symbol)
            if df is not None and not df.empty:
                all_frames.append(df)
        except Exception as exc:
            logger.error(f"Failed crypto collection for {symbol}: {exc}")

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    logger.info(f"Collected {len(df)} crypto rows")
    return df


def persist(df: pd.DataFrame, table: str = "price_bars") -> int:
    """Persist DataFrame to database.

    Args:
        df: DataFrame to persist
        table: Target table name

    Returns:
        Number of rows persisted
    """
    if df is None or df.empty:
        logger.warning(f"No data to persist to {table}")
        return 0

    try:
        engine = get_engine()
        chunksize = 1000
        with engine.begin() as conn:
            df.to_sql(table, conn, if_exists="append", index=False, chunksize=chunksize)
        logger.info(f"Persisted {len(df)} rows into {table}")
        return len(df)
    except Exception as e:
        logger.error(f"Failed to persist data to {table}: {e}")
        return 0


def main() -> None:
    """Main entry point for data collection."""
    settings = get_settings()
    logger.add(settings.logs_dir / "data_collection.log", rotation="50 MB", level=settings.log_level)

    # Log configuration status at startup
    settings.log_configuration_status()
    logger.info("Starting market data collection...")

    target_symbols = settings.target_symbols
    if not target_symbols:
        logger.warning("No target symbols configured. Check settings.target_symbols")
        return

    logger.info(f"Collecting data for {len(target_symbols)} symbols")

    # Collect macro indicators
    macro_df = collect_macro()
    if not macro_df.empty:
        persist(macro_df, table="macro_indicators")

    # Collect equities
    eq_df = collect_equities(target_symbols)

    # Collect crypto
    crypto_symbols = [s for s in target_symbols if "-USD" in s]
    crypto_df = collect_crypto(crypto_symbols) if crypto_symbols else pd.DataFrame()

    # Combine and persist
    frames = [df for df in [eq_df, crypto_df] if not df.empty]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        count = persist(combined, table="price_bars")
        logger.info(f"Data collection complete. Total rows stored: {count}")
    else:
        logger.warning("No data collected from any source")


if __name__ == "__main__":
    main()
