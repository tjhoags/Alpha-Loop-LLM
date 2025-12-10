"""================================================================================
DATA COLLECTOR - Optimized Multi-Source Data Ingestion
================================================================================

HOW TO RUN:
-----------
Windows (PowerShell):
    cd "C:\\Users\\tom\\.cursor\\worktrees\\Alpha-Loop-LLM-1\\sii"
    .\\venv\\Scripts\\Activate.ps1
    python src/data_ingestion/collector.py

Mac (Terminal):
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
    source venv/bin/activate
    caffeinate -d python src/data_ingestion/collector.py

WHAT THIS MODULE DOES:
----------------------
1. Collects price data from multiple sources (Polygon, Alpha Vantage, Coinbase)
2. Collects macro indicators from FRED
3. Validates and normalizes all data to consistent types
4. Persists data to Azure SQL with deduplication
5. Handles rate limits and retries automatically

DATA FLOW:
----------
    Sources (APIs) → Collector → Validation → Normalization → Database

PERFORMANCE OPTIMIZATIONS:
--------------------------
- Parallel API calls using ThreadPoolExecutor
- Chunked database inserts (1000 rows per batch)
- Connection pooling for database
- Type validation before insert
- Deduplication on (symbol, timestamp)

================================================================================
"""

from __future__ import annotations

import concurrent.futures
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import pandas as pd
from loguru import logger

from src.config.settings import get_settings
from src.data_ingestion.data_types import (
    AssetType,
    DataSource,
    TimeFrame,
    validate_price_dataframe,
    normalize_source_dataframe,
    POLYGON_MAPPING,
    ALPHA_VANTAGE_MAPPING,
)
from src.data_ingestion.sources.alpha_vantage import fetch_intraday as av_fetch
from src.data_ingestion.sources.coinbase import fetch_candles as coinbase_fetch
from src.data_ingestion.sources.fred import FredClient
from src.data_ingestion.sources.polygon import fetch_aggregates as polygon_fetch
from src.database.connection import get_engine

T = TypeVar("T")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Maximum parallel workers for API calls (IO-bound)
MAX_WORKERS = 10

# Database batch size for inserts
DB_CHUNK_SIZE = 1000

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


# =============================================================================
# DATA COLLECTION FUNCTIONS
# =============================================================================

def _with_retry(
    func: Callable[..., T],
    *args,
    max_retries: int = MAX_RETRIES,
    **kwargs
) -> Optional[T]:
    """
    Execute function with retry logic.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        max_retries: Maximum number of retry attempts
        **kwargs: Keyword arguments
        
    Returns:
        Function result or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))  # Exponential backoff
    return None


def _process_symbol(symbol: str) -> pd.DataFrame:
    """
    Process a single symbol by fetching from multiple sources.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Combined DataFrame from all sources
    """
    frames: List[pd.DataFrame] = []
    
    try:
        logger.info(f"📊 Collecting data for {symbol}...")
        
        # Polygon: 1-minute bars, last 2 years
        df_poly = _with_retry(
            polygon_fetch,
            symbol,
            timespan="minute",
            multiplier=1,
            lookback_days=730
        )
        if df_poly is not None and not df_poly.empty:
            df_poly = normalize_source_dataframe(
                df_poly,
                source=DataSource.POLYGON,
                column_mapping=POLYGON_MAPPING
            )
            df_poly["symbol"] = symbol
            df_poly["asset_type"] = AssetType.EQUITY.value
            frames.append(df_poly)
            logger.debug(f"  Polygon: {len(df_poly)} rows for {symbol}")

        # Alpha Vantage: 1-minute bars, full recent history
        df_av = _with_retry(av_fetch, symbol, interval="1min")
        if df_av is not None and not df_av.empty:
            df_av = normalize_source_dataframe(
                df_av,
                source=DataSource.ALPHA_VANTAGE,
                column_mapping=ALPHA_VANTAGE_MAPPING
            )
            df_av["symbol"] = symbol
            df_av["asset_type"] = AssetType.EQUITY.value
            frames.append(df_av)
            logger.debug(f"  Alpha Vantage: {len(df_av)} rows for {symbol}")

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            # Validate the combined data
            combined = validate_price_dataframe(combined, strict=False)
            return combined
            
    except Exception as exc:
        logger.error(f"❌ Failed equity collection for {symbol}: {exc}")
    
    return pd.DataFrame()


def collect_equities(symbols: List[str]) -> pd.DataFrame:
    """
    Collect equity data for all symbols in parallel.
    
    Uses ThreadPoolExecutor for concurrent API calls since these
    are IO-bound operations.
    
    Args:
        symbols: List of stock ticker symbols
        
    Returns:
        Combined DataFrame with all price data
    """
    all_frames: List[pd.DataFrame] = []
    
    logger.info(f"🚀 Starting parallel collection for {len(symbols)} symbols...")
    start_time = time.time()
    
    # Use ThreadPool for IO-bound API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(_process_symbol, sym): sym 
            for sym in symbols
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if not result.empty:
                    all_frames.append(result)
            except Exception as exc:
                logger.error(f"❌ {symbol} generated exception: {exc}")

    if not all_frames:
        logger.warning("⚠️ No equity data collected")
        return pd.DataFrame()

    # Combine all frames
    df = pd.concat(all_frames, ignore_index=True)
    
    # Sort and deduplicate
    df.sort_values(["symbol", "timestamp"], inplace=True)
    df.drop_duplicates(subset=["symbol", "timestamp"], keep="last", inplace=True)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Collected {len(df):,} rows in {elapsed:.1f}s")
    
    return df


def collect_macro() -> pd.DataFrame:
    """
    Collect macroeconomic indicators from FRED.
    
    Returns:
        DataFrame with macro indicator data
    """
    try:
        logger.info("📈 Collecting macro indicators from FRED...")
        fred = FredClient()
        df = fred.fetch_core_macro()
        if not df.empty:
            df["source"] = DataSource.FRED.value
            logger.info(f"✅ Collected {len(df)} macro indicator rows")
        return df
    except Exception as e:
        logger.error(f"❌ Macro collection failed: {e}")
        return pd.DataFrame()


def collect_crypto(symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Collect cryptocurrency data from Coinbase.
    
    Args:
        symbols: List of crypto pairs (default: BTC-USD)
        
    Returns:
        DataFrame with crypto price data
    """
    if symbols is None:
        symbols = ["BTC-USD", "ETH-USD"]
    
    frames: List[pd.DataFrame] = []
    
    try:
        logger.info(f"₿ Collecting crypto data for {symbols}...")
        
        for symbol in symbols:
            df = _with_retry(coinbase_fetch, symbol)
            if df is not None and not df.empty:
                df["symbol"] = symbol
                df["source"] = DataSource.COINBASE.value
                df["asset_type"] = AssetType.CRYPTO.value
                frames.append(df)
                logger.debug(f"  Coinbase: {len(df)} rows for {symbol}")
        
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = validate_price_dataframe(combined, strict=False)
            logger.info(f"✅ Collected {len(combined)} crypto rows")
            return combined
            
    except Exception as exc:
        logger.error(f"❌ Crypto collection failed: {exc}")
    
    return pd.DataFrame()


# =============================================================================
# DATABASE PERSISTENCE
# =============================================================================

def persist(
    df: pd.DataFrame,
    table: str = "price_bars",
    chunk_size: int = DB_CHUNK_SIZE
) -> int:
    """
    Persist DataFrame to database with chunked inserts.
    
    Args:
        df: DataFrame to persist
        table: Target table name
        chunk_size: Number of rows per insert batch
        
    Returns:
        Number of rows persisted
    """
    if df.empty:
        logger.warning("⚠️ No data to persist.")
        return 0
    
    engine = get_engine()
    
    try:
        with engine.begin() as conn:
            df.to_sql(
                table,
                conn,
                if_exists="append",
                index=False,
                chunksize=chunk_size
            )
        logger.info(f"💾 Persisted {len(df):,} rows to {table}")
        return len(df)
    except Exception as e:
        logger.error(f"❌ Failed to persist to {table}: {e}")
        return 0


def persist_with_upsert(
    df: pd.DataFrame,
    table: str = "price_bars",
    conflict_columns: List[str] = None
) -> int:
    """
    Persist DataFrame with upsert logic (insert or update on conflict).
    
    Args:
        df: DataFrame to persist
        table: Target table name
        conflict_columns: Columns that define uniqueness
        
    Returns:
        Number of rows affected
    """
    if conflict_columns is None:
        conflict_columns = ["symbol", "timestamp"]
    
    if df.empty:
        logger.warning("⚠️ No data to persist.")
        return 0
    
    # For now, use simple append with deduplication in DataFrame
    # TODO: Implement proper SQL MERGE for upsert
    df = df.drop_duplicates(subset=conflict_columns, keep="last")
    return persist(df, table)


# =============================================================================
# MAIN COLLECTION ORCHESTRATOR
# =============================================================================

def run_collection_cycle(
    symbols: Optional[List[str]] = None,
    include_macro: bool = True,
    include_crypto: bool = True
) -> Dict[str, int]:
    """
    Run a complete data collection cycle.
    
    Args:
        symbols: List of equity symbols (default: from settings)
        include_macro: Whether to collect macro indicators
        include_crypto: Whether to collect crypto data
        
    Returns:
        Dictionary with row counts per data type
    """
    settings = get_settings()
    
    # Configure logging
    logger.add(
        settings.logs_dir / "data_collection.log",
        rotation="50 MB",
        level=settings.log_level
    )
    
    logger.info("=" * 60)
    logger.info("🚀 Starting TOTAL MARKET data collection cycle")
    logger.info("=" * 60)
    
    results: Dict[str, int] = {}
    
    # 1. Collect Macro Indicators
    if include_macro:
        macro_df = collect_macro()
        results["macro"] = persist(macro_df, table="macro_indicators")
    
    # 2. Get target symbols
    if symbols is None:
        symbols = settings.target_symbols
    
    logger.info(f"📋 Target symbols: {len(symbols)}")
    
    # 3. Collect Equities
    eq_df = collect_equities(symbols)
    
    # 4. Collect Crypto
    crypto_df = pd.DataFrame()
    if include_crypto:
        crypto_df = collect_crypto()
    
    # 5. Combine and persist
    combined_frames = [eq_df, crypto_df]
    combined = pd.concat([f for f in combined_frames if not f.empty], ignore_index=True)
    
    if not combined.empty:
        results["price_bars"] = persist_with_upsert(combined, table="price_bars")
    else:
        results["price_bars"] = 0
    
    # Summary
    total = sum(results.values())
    logger.info("=" * 60)
    logger.info(f"✅ Collection cycle complete. Total rows: {total:,}")
    for table, count in results.items():
        logger.info(f"   {table}: {count:,} rows")
    logger.info("=" * 60)
    
    return results


def main() -> None:
    """
    Main entry point for data collection.
    
    Run from command line:
        Windows:  python src/data_ingestion/collector.py
        Mac:      caffeinate -d python src/data_ingestion/collector.py
    """
    results = run_collection_cycle()
    
    # Report summary
    print("\n" + "=" * 40)
    print("DATA COLLECTION SUMMARY")
    print("=" * 40)
    for table, count in results.items():
        print(f"  {table}: {count:,} rows")
    print("=" * 40)


if __name__ == "__main__":
    main()
