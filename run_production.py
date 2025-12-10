#!/usr/bin/env python3
"""
ALC-Algo Production Runner
Author: Tom Hogan | Alpha Loop Capital, LLC

Main entry point for production trading system.
Runs before market open and continues through close.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(f"data/logs/production_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

from src.core.trading_engine import TradingEngine


def check_prerequisites():
    """Check that all prerequisites are met"""
    issues = []

    # Check database connection
    if not os.getenv("DATABASE_URL"):
        issues.append("⚠️  DATABASE_URL not set - will use file logging only")

    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("❌ OPENAI_API_KEY not set")

    if not os.getenv("ANTHROPIC_API_KEY"):
        issues.append("⚠️  ANTHROPIC_API_KEY not set")

    # Check market data
    if not os.getenv("POLYGON_API_KEY") and not os.getenv("ALPHA_VANTAGE_API_KEY"):
        issues.append("⚠️  No market data API configured")

    if issues:
        print("\n" + "=" * 60)
        print("Prerequisites Check:")
        print("=" * 60)
        for issue in issues:
            print(issue)
        print("=" * 60)
        print()

        if any("❌" in issue for issue in issues):
            print("FATAL: Critical prerequisites missing!")
            return False

    return True


def main():
    """Main production entry point"""
    logger.info("=" * 60)
    logger.info("ALC-ALGO PRODUCTION SYSTEM STARTING")
    logger.info("=" * 60)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working Dir: {os.getcwd()}")
    logger.info("=" * 60)

    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed - exiting")
        return 1

    # Initialize trading engine
    logger.info("Initializing Trading Engine...")

    # Get config from environment
    initial_capital = float(os.getenv("INITIAL_CAPITAL", "100000"))
    paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"

    engine = TradingEngine(
        initial_capital=initial_capital,
        paper_trading=paper_trading
    )

    logger.info(f"Trading Engine initialized | Capital: ${initial_capital:,.2f} | Paper: {paper_trading}")

    # Determine what to run based on time
    now = datetime.now()
    hour = now.hour
    minute = now.minute

    try:
        if hour < 9 or (hour == 9 and minute < 30):
            # Pre-market: Run morning scan
            logger.info("PRE-MARKET: Running morning scan...")
            engine.run_morning_scan()
            logger.info("Morning scan complete. Waiting for market open...")

        elif hour >= 16:
            # After hours: Run EOD analysis
            logger.info("AFTER HOURS: Running EOD analysis...")
            engine.run_eod_analysis()
            logger.info("EOD analysis complete.")

        else:
            # Market hours: Run trading loop
            logger.info("MARKET HOURS: Starting trading loop...")
            engine.run_trading_loop()
            logger.info("Trading loop complete.")

    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error in production system: {e}", exc_info=True)
        return 1

    logger.info("=" * 60)
    logger.info("ALC-ALGO PRODUCTION SYSTEM STOPPED")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
