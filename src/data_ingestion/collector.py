import concurrent.futures
from typing import List

import pandas as pd
from loguru import logger

from src.config.settings import get_settings
from src.data_ingestion.sources.alpha_vantage import fetch_intraday as av_fetch
from src.data_ingestion.sources.coinbase import fetch_candles as coinbase_fetch
from src.data_ingestion.sources.fred import FredClient
from src.data_ingestion.sources.polygon import fetch_aggregates as massive_fetch
from src.database.connection import get_engine


def collect_equities(symbols: List[str]) -> pd.DataFrame:
    """Collects data for all symbols in PARALLEL.
    """
    all_frames = []

    def process_symbol(sym):
        frames = []
        try:
            logger.info(f"Collecting PREMIUM data for {sym}...")
            # Massive.com: 1-minute bars, last 2 years
            df_massive = massive_fetch(sym, timespan="minute", multiplier=1, lookback_days=730)
            if not df_massive.empty:
                frames.append(df_massive)

            # Alpha Vantage: 1-minute bars, full recent history
            df_av = av_fetch(sym, interval="1min")
            if not df_av.empty:
                frames.append(df_av)

            if frames:
                return pd.concat(frames, ignore_index=True)
        except Exception as exc:
            logger.error(f"Failed equity collection for {sym}: {exc}")
        return pd.DataFrame()

    # Use ThreadPool to hit APIs concurrently (IO-bound)
    # Max workers = 10 for Full Throttle
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_symbol, symbols)

    for res in results:
        if not res.empty:
            all_frames.append(res)

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df.sort_values(["symbol", "timestamp"], inplace=True)
    df.drop_duplicates(subset=["symbol", "timestamp"], keep="last", inplace=True)
    return df


def collect_macro() -> pd.DataFrame:
    try:
        fred = FredClient()
        return fred.fetch_core_macro()
    except Exception as e:
        logger.error(f"Macro collection failed: {e}")
        return pd.DataFrame()


def collect_crypto() -> pd.DataFrame:
    try:
        logger.info("Collecting crypto data...")
        return coinbase_fetch("BTC-USD")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed crypto collection: {exc}")
        return pd.DataFrame()


def persist(df: pd.DataFrame, table: str = "price_bars") -> int:
    if df.empty:
        logger.warning("No data to persist.")
        return 0
    engine = get_engine()
    chunksize = 1000
    with engine.begin() as conn:
        df.to_sql(table, conn, if_exists="append", index=False, chunksize=chunksize)
    logger.info(f"Persisted {len(df)} rows into {table}")
    return len(df)


def main() -> None:
    settings = get_settings()
    logger.add(settings.logs_dir / "data_collection.log", rotation="50 MB", level=settings.log_level)
    logger.info("Starting TOTAL MARKET data collection.")

    # 1. Expand Universe (Dynamic)
    # WARNING: fetching all 10k tickers might take too long for tonight.
    # For tonight, let's stick to the target list + top 100 tech.
    # To enable full universe, uncomment the line below:
    # target_symbols = fetch_all_tickers()

    # Using settings list for immediate safety, but you can swap this.
    target_symbols = settings.target_symbols

    # 2. Collect Macro
    macro_df = collect_macro()
    persist(macro_df, table="macro_indicators") # New table for macro

    # 3. Collect Equities & Crypto
    eq_df = collect_equities(target_symbols)
    crypto_df = collect_crypto()

    combined = pd.concat([eq_df, crypto_df], ignore_index=True)
    count = persist(combined, table="price_bars")

    logger.info(f"Data collection cycle completed. Rows stored: {count}")


if __name__ == "__main__":
    main()
