"""
================================================================================
COMPREHENSIVE ALPHA VANTAGE DATA HYDRATION
================================================================================
Pulls ALL historical data from Alpha Vantage premium tier:
- Stocks (intraday + daily + fundamentals)
- Indices (S&P 500, NASDAQ, Dow)
- Currencies/Forex (major pairs)
- Options (if available)
- Runs continuously to backfill and maintain fresh data
================================================================================
"""

from datetime import datetime, timedelta
from typing import List
import time

import pandas as pd
from loguru import logger

from src.config.settings import get_settings
from src.data_ingestion.collector import persist
from src.data_ingestion.sources.alpha_vantage_premium import get_av_premium


# Major indices
MAJOR_INDICES = ["SPX", "NDX", "DJI", "VIX", "RUT", "VXX", "SPY", "QQQ", "IWM", "DIA"]

# Major forex pairs
FOREX_PAIRS = [
    ("USD", "EUR"), ("USD", "GBP"), ("USD", "JPY"), ("USD", "CHF"),
    ("USD", "CAD"), ("USD", "AUD"), ("EUR", "GBP"), ("EUR", "JPY"),
    ("GBP", "JPY"), ("AUD", "USD"), ("NZD", "USD"), ("USD", "CHF"), 
    ("USD", "MXN"), ("USD", "BRL"), ("USD", "ARS"), ("USD", "CLP"), 
    ("USD", "COP"), ("USD", "PEN"), ("USD", "MXN"), ("USD", "BRL"), 
    ("USD", "ARS"), ("USD", "CLP"), ("USD", "COP"), ("USD", "PEN"),
    ("USD", "MXN"), ("USD", "BRL"), ("USD", "ARS"), ("USD", "CLP"), 
    ("USD", "COP"), ("USD", "PEN"), ("USD", "MXN"), ("USD", "BRL"), 

]

# Extended stock universe (add your small/mid-cap focus here)
EXTENDED_STOCKS = [
    # Add your small/mid-cap universe here
    # Example: "AAPL", "MSFT", "NVDA", etc.
]


def hydrate_stocks(symbols: List[str], include_fundamentals: bool = True):
    """Hydrate stock data: intraday, daily, and fundamentals."""
    av = get_av_premium()
    all_data = []
    
    for symbol in symbols:
        try:
            logger.info(f"🔄 Hydrating STOCK: {symbol}")
            
            # 1. Intraday (1-minute, full history)
            df_intraday = av.fetch_stock_intraday(symbol, interval="1min", outputsize="full")
            if not df_intraday.empty:
                all_data.append(df_intraday)
                logger.info(f"  ✓ Intraday: {len(df_intraday)} bars")
            
            # 2. Daily (20+ years)
            df_daily = av.fetch_stock_daily(symbol, outputsize="full")
            if not df_daily.empty:
                all_data.append(df_daily)
                logger.info(f"  ✓ Daily: {len(df_daily)} bars")
            
            # 3. Fundamentals (valuation metrics)
            if include_fundamentals:
                fundamentals = av.fetch_fundamental_data(symbol)
                if fundamentals:
                    # Store in separate table
                    fund_df = pd.DataFrame([fundamentals])
                    persist(fund_df, table="fundamental_data")
                    logger.info(f"  ✓ Fundamentals: stored")
            
            # Rate limiting: Alpha Vantage premium = 75 calls/minute
            # We're being conservative with 12s delay
            time.sleep(12.1)
            
        except Exception as e:
            logger.error(f"❌ Error hydrating {symbol}: {e}")
            continue
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.sort_values(["symbol", "timestamp"], inplace=True)
        combined.drop_duplicates(subset=["symbol", "timestamp"], keep="last", inplace=True)
        count = persist(combined, table="price_bars")
        logger.success(f"✅ Stock hydration complete: {count} total rows stored")
    else:
        logger.warning("No stock data collected")


def hydrate_indices(indices: List[str]):
    """Hydrate index data."""
    av = get_av_premium()
    all_data = []
    
    for index_symbol in indices:
        try:
            logger.info(f"🔄 Hydrating INDEX: {index_symbol}")
            
            # Daily index data
            df = av.fetch_index(index_symbol, interval="daily")
            if not df.empty:
                all_data.append(df)
                logger.info(f"  ✓ Index: {len(df)} bars")
            
            time.sleep(12.1)
            
        except Exception as e:
            logger.error(f"❌ Error hydrating {index_symbol}: {e}")
            continue
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        count = persist(combined, table="price_bars")
        logger.success(f"✅ Index hydration complete: {count} total rows stored")


def hydrate_forex(pairs: List[tuple]):
    """Hydrate forex/currency pair data."""
    av = get_av_premium()
    all_data = []
    
    for from_curr, to_curr in pairs:
        try:
            logger.info(f"🔄 Hydrating FOREX: {from_curr}/{to_curr}")
            
            # 1-minute forex data
            df = av.fetch_forex(from_curr, to_curr, interval="1min")
            if not df.empty:
                all_data.append(df)
                logger.info(f"  ✓ Forex: {len(df)} bars")
            
            time.sleep(12.1)
            
        except Exception as e:
            logger.error(f"❌ Error hydrating {from_curr}/{to_curr}: {e}")
            continue
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        count = persist(combined, table="price_bars")
        logger.success(f"✅ Forex hydration complete: {count} total rows stored")


def main():
    """Main hydration loop - runs continuously."""
    settings = get_settings()
    logger.add(settings.logs_dir / "alpha_vantage_hydration.log", rotation="100 MB", level="INFO")
    logger.info("🚀 Starting COMPREHENSIVE Alpha Vantage hydration...")
    
    # Get target symbols from settings
    target_stocks = settings.target_symbols
    
    # Add extended universe if configured
    if settings.use_full_universe:
        # You can expand this to fetch from Polygon/Massive
        target_stocks.extend(EXTENDED_STOCKS)
    
    # Remove duplicates
    target_stocks = list(set(target_stocks))
    
    logger.info(f"📊 Hydrating {len(target_stocks)} stocks, {len(MAJOR_INDICES)} indices, {len(FOREX_PAIRS)} forex pairs")
    
    # Run hydration cycles
    cycle = 0
    while True:
        cycle += 1
        logger.info(f"\n{'='*80}")
        logger.info(f"HYDRATION CYCLE #{cycle} - {datetime.now()}")
        logger.info(f"{'='*80}\n")
        
        try:
            # 1. Stocks (intraday + daily + fundamentals)
            hydrate_stocks(target_stocks, include_fundamentals=True)
            
            # 2. Indices
            hydrate_indices(MAJOR_INDICES)
            
            # 3. Forex
            hydrate_forex(FOREX_PAIRS)
            
            logger.success(f"✅ Cycle #{cycle} complete. Sleeping 1 hour before next cycle...")
            time.sleep(3600)  # Wait 1 hour before next cycle
            
        except KeyboardInterrupt:
            logger.info("🛑 Hydration stopped by user")
            break
        except Exception as e:
            logger.exception(f"❌ Fatal error in cycle #{cycle}: {e}")
            logger.info("⏳ Waiting 5 minutes before retry...")
            time.sleep(300)


if __name__ == "__main__":
    main()

