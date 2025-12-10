"""================================================================================
ALPHA VANTAGE DATA HYDRATION
================================================================================

WHAT THIS DOES:
    Pulls comprehensive market data from Alpha Vantage Premium API:
    - Stock price data (intraday 1min, 5min, daily)
    - Fundamental data (P/E, EV/EBITDA, margins, growth rates)
    - Forex pairs (EUR/USD, GBP/USD, etc.)
    - Index data (SPX, NDX, DJI)
    - Earnings calendars and estimates

WHY USE ALPHA VANTAGE:
    - Premium tier: 75 API calls/minute (vs 5 for free)
    - Up to 20+ years of daily data
    - Up to 2 years of intraday data
    - Fundamental data unavailable elsewhere
    - Good for US small/mid cap companies

RATE LIMITS:
    - Premium: 75 calls/minute, 500 calls/day (can request more)
    - This script enforces 12-second delays to stay safe

WHAT IT PULLS:
    1. STOCKS: Daily adjusted (20+ years), Intraday (2 years)
    2. FUNDAMENTALS: Company overview with 30+ valuation metrics
    3. FOREX: Major currency pairs
    4. INDICES: SPX, NDX, DJI (via proxy ETFs)
    5. EARNINGS: Upcoming earnings dates and estimates

================================================================================
HOW TO USE:
================================================================================

STEP 1: Make sure your .env has Alpha Vantage API key:
    ALPHA_VANTAGE_API_KEY=your_key_here

STEP 2: Activate virtual environment:

    Windows (PowerShell):
    ---------------------
    cd C:\\Users\\tom\\Alpha-Loop-LLM\\Alpha-Loop-LLM-1
    .\\venv\\Scripts\\activate

    Mac (Terminal):
    ---------------
    cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1
    source venv/bin/activate

STEP 3: Run hydration:

    # Full hydration (takes 2-4 hours due to rate limits)
    python scripts/hydrate_alpha_vantage.py

    # Quick test (10-15 minutes)
    python scripts/hydrate_alpha_vantage.py --quick

    # Specific asset class
    python scripts/hydrate_alpha_vantage.py --stocks-only
    python scripts/hydrate_alpha_vantage.py --forex-only
    python scripts/hydrate_alpha_vantage.py --fundamentals-only

STEP 4: Check outputs:
    - Data saved to Azure SQL (price_bars, fundamentals tables)
    - Backup CSVs in data/csv_backup/
    - Log file: logs/alpha_vantage_hydration.log

================================================================================
DATA STORED:
================================================================================

price_bars table:
    symbol, timestamp, open, high, low, close, volume, source, asset_type

fundamentals table:
    symbol, timestamp, pe_ratio, peg_ratio, price_to_book, price_to_sales,
    ev_to_revenue, ev_to_ebitda, profit_margin, operating_margin, etc.

forex_bars table:
    symbol, timestamp, open, high, low, close, source

================================================================================
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.data_ingestion.sources.alpha_vantage_premium import (
    get_av_premium,
)
from src.database.connection import get_engine


class AlphaVantageHydrator:
    """Comprehensive Alpha Vantage data hydration.
    """

    def __init__(self):
        self.settings = get_settings()
        self.engine = get_engine()
        self.av = get_av_premium()

        # Check API key
        if not self.settings.alpha_vantage_api_key:
            raise ValueError(
                "Alpha Vantage API key not found!\n"
                "Add ALPHA_VANTAGE_API_KEY to your .env file",
            )

        self.stats = {
            "stocks_daily": 0,
            "stocks_intraday": 0,
            "fundamentals": 0,
            "forex": 0,
            "indices": 0,
            "errors": 0,
            "rows_saved": 0,
        }

        # Create backup directory
        self.backup_dir = Path(self.settings.data_dir) / "csv_backup" / "alpha_vantage"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("ALPHA VANTAGE DATA HYDRATION")
        logger.info("=" * 70)
        logger.info(f"API Key: {self.settings.alpha_vantage_api_key[:8]}...")
        logger.info(f"Backup dir: {self.backup_dir}")

    def get_target_stocks(self, limit: Optional[int] = None) -> List[str]:
        """Get list of stocks to hydrate."""
        # Default universe: US small/mid cap focus
        # You can customize this list
        stocks = [
            # Large Cap Tech
            "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AMD", "TSLA",
            "NFLX", "ADBE", "CRM", "ORCL", "INTC", "CSCO", "QCOM", "TXN",

            # Financials
            "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "V", "MA", "PYPL",
            "SQ", "COIN", "HOOD", "SCHW", "BLK", "BX", "KKR",

            # Healthcare
            "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "DHR", "BMY",
            "AMGN", "GILD", "MRNA", "BNTX", "REGN", "VRTX",

            # Consumer
            "WMT", "COST", "HD", "LOW", "TGT", "NKE", "SBUX", "MCD", "DIS",
            "CMCSA", "NFLX", "ABNB", "UBER", "LYFT", "DASH",

            # Energy
            "XOM", "CVX", "COP", "OXY", "SLB", "HAL", "EOG", "MPC", "VLO",

            # Industrials
            "BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "LMT", "RTX",

            # Small/Mid Cap (your focus)
            "DDOG", "NET", "ZS", "CRWD", "SNOW", "MDB", "PLTR", "U", "RBLX",
            "AFRM", "SOFI", "UPST", "OPEN", "HOOD", "RIVN", "LCID",
            "AI", "PATH", "GTLB", "CFLT", "SAMSARA", "DUOL",

            # ETFs for indices
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
        ]

        if limit:
            return stocks[:limit]
        return stocks

    def get_forex_pairs(self) -> List[tuple]:
        """Get forex pairs to hydrate."""
        return [
            ("EUR", "USD"),
            ("GBP", "USD"),
            ("USD", "JPY"),
            ("USD", "CHF"),
            ("AUD", "USD"),
            ("USD", "CAD"),
            ("NZD", "USD"),
            ("EUR", "GBP"),
            ("EUR", "JPY"),
            ("GBP", "JPY"),
        ]

    def save_to_sql(self, df: pd.DataFrame, table: str) -> int:
        """Save DataFrame to SQL with CSV backup."""
        if df.empty:
            return 0

        # CSV backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.backup_dir / f"{table}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        # SQL insert
        try:
            with self.engine.begin() as conn:
                df.to_sql(table, conn, if_exists="append", index=False, chunksize=1000)
            self.stats["rows_saved"] += len(df)
            logger.info(f"Saved {len(df)} rows to {table}")
            return len(df)
        except Exception as e:
            logger.error(f"SQL save failed (CSV backup at {csv_path}): {e}")
            return 0

    def hydrate_stock_daily(self, symbol: str) -> bool:
        """Hydrate daily stock data."""
        try:
            df = self.av.fetch_stock_daily(symbol, outputsize="full")
            if not df.empty:
                self.save_to_sql(df, "price_bars")
                self.stats["stocks_daily"] += 1
                return True
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {e}")
            self.stats["errors"] += 1
        return False

    def hydrate_stock_intraday(self, symbol: str, interval: str = "5min") -> bool:
        """Hydrate intraday stock data."""
        try:
            df = self.av.fetch_stock_intraday(symbol, interval=interval, outputsize="full")
            if not df.empty:
                self.save_to_sql(df, "price_bars")
                self.stats["stocks_intraday"] += 1
                return True
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            self.stats["errors"] += 1
        return False

    def hydrate_fundamentals(self, symbol: str) -> bool:
        """Hydrate fundamental data."""
        try:
            data = self.av.fetch_fundamental_data(symbol)
            if data:
                df = pd.DataFrame([data])
                self.save_to_sql(df, "fundamentals")
                self.stats["fundamentals"] += 1
                return True
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            self.stats["errors"] += 1
        return False

    def hydrate_forex(self, from_currency: str, to_currency: str) -> bool:
        """Hydrate forex pair data."""
        try:
            df = self.av.fetch_forex(from_currency, to_currency, interval="1min")
            if not df.empty:
                self.save_to_sql(df, "forex_bars")
                self.stats["forex"] += 1
                return True
        except Exception as e:
            logger.error(f"Error fetching forex {from_currency}/{to_currency}: {e}")
            self.stats["errors"] += 1
        return False

    def hydrate_stocks(self, symbols: List[str], include_intraday: bool = False):
        """Hydrate all stock data."""
        logger.info(f"\nHydrating {len(symbols)} stocks...")

        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] Processing {symbol}")

            # Daily data
            self.hydrate_stock_daily(symbol)

            # Intraday (optional, uses more API calls)
            if include_intraday:
                self.hydrate_stock_intraday(symbol)

            # Fundamentals
            self.hydrate_fundamentals(symbol)

    def hydrate_all_forex(self):
        """Hydrate all forex pairs."""
        pairs = self.get_forex_pairs()
        logger.info(f"\nHydrating {len(pairs)} forex pairs...")

        for i, (from_curr, to_curr) in enumerate(pairs):
            logger.info(f"[{i+1}/{len(pairs)}] Processing {from_curr}/{to_curr}")
            self.hydrate_forex(from_curr, to_curr)

    def run_full_hydration(
        self,
        stocks: bool = True,
        fundamentals: bool = True,
        forex: bool = True,
        intraday: bool = False,
        stock_limit: Optional[int] = None,
    ):
        """Run full hydration."""
        start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("STARTING FULL HYDRATION")
        logger.info("=" * 70)
        logger.info(f"Stocks: {stocks}, Fundamentals: {fundamentals}, Forex: {forex}, Intraday: {intraday}")

        if stocks or fundamentals:
            target_stocks = self.get_target_stocks(limit=stock_limit)
            self.hydrate_stocks(target_stocks, include_intraday=intraday)

        if forex:
            self.hydrate_all_forex()

        elapsed = time.time() - start_time

        logger.info("\n" + "=" * 70)
        logger.info("HYDRATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
        logger.info(f"Stocks daily: {self.stats['stocks_daily']}")
        logger.info(f"Stocks intraday: {self.stats['stocks_intraday']}")
        logger.info(f"Fundamentals: {self.stats['fundamentals']}")
        logger.info(f"Forex pairs: {self.stats['forex']}")
        logger.info(f"Total rows saved: {self.stats['rows_saved']}")
        logger.info(f"Errors: {self.stats['errors']}")

    def run_quick_hydration(self):
        """Quick hydration for testing."""
        logger.info("Running quick hydration (10 stocks, no intraday)...")
        self.run_full_hydration(
            stocks=True,
            fundamentals=True,
            forex=False,
            intraday=False,
            stock_limit=10,
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Alpha Vantage Data Hydration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/hydrate_alpha_vantage.py              # Full hydration
    python scripts/hydrate_alpha_vantage.py --quick      # Quick test
    python scripts/hydrate_alpha_vantage.py --stocks-only
    python scripts/hydrate_alpha_vantage.py --fundamentals-only
    python scripts/hydrate_alpha_vantage.py --forex-only
        """,
    )

    parser.add_argument("--quick", action="store_true", help="Quick test (10 stocks)")
    parser.add_argument("--stocks-only", action="store_true", help="Hydrate stocks only")
    parser.add_argument("--fundamentals-only", action="store_true", help="Hydrate fundamentals only")
    parser.add_argument("--forex-only", action="store_true", help="Hydrate forex only")
    parser.add_argument("--with-intraday", action="store_true", help="Include intraday data")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of stocks")

    args = parser.parse_args()

    # Setup logging
    settings = get_settings()
    log_file = settings.logs_dir / "alpha_vantage_hydration.log"
    logger.add(log_file, rotation="50 MB", level="INFO")

    try:
        hydrator = AlphaVantageHydrator()

        if args.quick:
            hydrator.run_quick_hydration()
        elif args.stocks_only:
            stocks = hydrator.get_target_stocks(limit=args.limit)
            hydrator.hydrate_stocks(stocks, include_intraday=args.with_intraday)
        elif args.fundamentals_only:
            stocks = hydrator.get_target_stocks(limit=args.limit)
            for s in stocks:
                hydrator.hydrate_fundamentals(s)
        elif args.forex_only:
            hydrator.hydrate_all_forex()
        else:
            hydrator.run_full_hydration(
                stocks=True,
                fundamentals=True,
                forex=True,
                intraday=args.with_intraday,
                stock_limit=args.limit,
            )

    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Hydration interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()

