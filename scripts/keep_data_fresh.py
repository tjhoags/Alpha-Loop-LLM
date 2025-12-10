"""================================================================================
KEEP DATA FRESH - Continuous Data Updater
================================================================================
Runs continuously to:
1. Pull latest market data every hour
2. Refresh sentiment/news data
3. Update fundamentals daily
4. Sync across Azure SQL

Run with: python scripts/keep_data_fresh.py
================================================================================
"""
import os
import sys
import time
from datetime import datetime, timedelta

import schedule
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import get_settings
from src.database.connection import get_engine

settings = get_settings()
logger.add(settings.logs_dir / "data_updater.log", rotation="100 MB", level="INFO")


def update_intraday_data():
    """Pull latest intraday data for all symbols."""
    logger.info("=" * 60)
    logger.info("UPDATING INTRADAY DATA")
    logger.info("=" * 60)

    try:
        import pandas as pd
        import requests
        from sqlalchemy import text

        api_key = settings.polygon_api_key
        engine = get_engine()

        # Get symbols from database
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT symbol FROM price_bars"))
            symbols = [row[0] for row in result.fetchall()]

        logger.info(f"Updating {len(symbols)} symbols...")

        # Get last 2 hours of data for each (recent updates)
        end = datetime.now()
        start = end - timedelta(hours=2)

        updated = 0
        for symbol in symbols[:500]:  # Batch of 500 at a time
            try:
                url = f"https://api.massive.com/v2/aggs/ticker/{symbol}/range/5/minute/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
                resp = requests.get(url, params={"apiKey": api_key, "limit": 50000})

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("results"):
                        df = pd.DataFrame(data["results"])
                        df["symbol"] = symbol
                        df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
                        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
                        df = df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]

                        # Upsert to database
                        df.to_sql("price_bars", engine, if_exists="append", index=False)
                        updated += 1

                time.sleep(0.15)  # Rate limit

            except Exception as e:
                logger.debug(f"Error updating {symbol}: {e}")

        logger.info(f"Updated {updated} symbols with latest data")

    except Exception as e:
        logger.error(f"Error in intraday update: {e}")


def update_daily_fundamentals():
    """Update fundamental data (runs daily)."""
    logger.info("=" * 60)
    logger.info("UPDATING FUNDAMENTALS")
    logger.info("=" * 60)

    # TODO: Add fundamental data updates from financial APIs
    logger.info("Fundamental updates - placeholder")


def update_sentiment():
    """Update sentiment/news data."""
    logger.info("=" * 60)
    logger.info("UPDATING SENTIMENT")
    logger.info("=" * 60)

    try:
        # Run research ingestion
        os.system("python scripts/ingest_research.py")
    except Exception as e:
        logger.error(f"Error updating sentiment: {e}")


def check_data_health():
    """Check data completeness and freshness."""
    logger.info("=" * 60)
    logger.info("DATA HEALTH CHECK")
    logger.info("=" * 60)

    try:
        import pyodbc
        conn = pyodbc.connect(
            "Driver={ODBC Driver 17 for SQL Server};"
            "Server=alc-sql-server.database.windows.net;"
            "Database=alc_market_data;"
            "UID=CloudSAb3fcbb35;"
            "PWD=ALCadmin27!",
        )
        cursor = conn.cursor()

        # Total rows
        cursor.execute("SELECT COUNT(*) FROM price_bars")
        total = cursor.fetchone()[0]

        # Unique symbols
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM price_bars")
        symbols = cursor.fetchone()[0]

        # Latest timestamp
        cursor.execute("SELECT MAX(timestamp) FROM price_bars")
        latest = cursor.fetchone()[0]

        logger.info(f"Total rows: {total:,}")
        logger.info(f"Unique symbols: {symbols}")
        logger.info(f"Latest data: {latest}")

        conn.close()

    except Exception as e:
        logger.error(f"Error checking data health: {e}")


def run_scheduled_updates():
    """Run the scheduler."""
    logger.info("=" * 60)
    logger.info("DATA UPDATER STARTED")
    logger.info("=" * 60)
    logger.info("Schedule:")
    logger.info("  - Intraday data: Every hour")
    logger.info("  - Fundamentals: Daily at 6:00 AM")
    logger.info("  - Sentiment: Every 4 hours")
    logger.info("  - Health check: Every 30 minutes")
    logger.info("")

    # Schedule tasks
    schedule.every().hour.do(update_intraday_data)
    schedule.every().day.at("06:00").do(update_daily_fundamentals)
    schedule.every(4).hours.do(update_sentiment)
    schedule.every(30).minutes.do(check_data_health)

    # Run health check immediately
    check_data_health()

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    import sys

    if "--once" in sys.argv:
        # Run all updates once
        check_data_health()
        update_intraday_data()
    else:
        # Run continuous scheduler
        run_scheduled_updates()


