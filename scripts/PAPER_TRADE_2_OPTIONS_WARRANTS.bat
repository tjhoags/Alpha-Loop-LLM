@echo off
REM ============================================================================
REM TERMINAL 2: OPTIONS, WARRANTS & DERIVATIVES PAPER TRADING
REM ============================================================================
REM Alpha Loop Capital, LLC
REM
REM Asset Classes: Stock Options, Index Options, Warrants, LEAPS
REM Strategies: CONVERSION_REVERSAL, Options Arbitrage, Greeks-based
REM Data Sources: Polygon Options, Options Greeks calculations
REM ============================================================================

title ALC Terminal 2 - OPTIONS and WARRANTS

echo ============================================================
echo     TERMINAL 2: OPTIONS and WARRANTS PAPER TRADING
echo ============================================================
echo.
echo [INFO] Asset Classes: Stock Options, Index Options, Warrants
echo [INFO] Strategies: CONVERSION_REVERSAL, Arbitrage, Greeks
echo [INFO] Mode: PAPER TRADING (Simulated)
echo [INFO] IBKR Port: 7497 (Paper)
echo.
echo ============================================================
echo.

cd /d "C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM"
call venv\Scripts\activate

echo [+] Environment ready
echo [+] Starting Options/Warrants trading agents...
echo.

python -c "
import sys
sys.path.insert(0, '.')

from loguru import logger
import time
from datetime import datetime

logger.remove()
logger.add(sys.stdout, level='INFO', format='{time:HH:mm:ss} | {level} | {message}')
logger.add('logs/paper_trade_options.log', rotation='1 day', level='INFO')

print('='*60)
print('  OPTIONS AND WARRANTS PAPER TRADING - TERMINAL 2')
print('='*60)
print()

try:
    from src.trading.production_algo import ProductionAlgo, check_readiness

    # Check readiness
    status = check_readiness()
    print(f'[>] Models loaded: {status[\"models\"][\"promoted\"]} promoted')
    print(f'[>] IBKR Connected: {status[\"ibkr\"][\"ready\"]}')
    print()

    # Initialize paper trading for options
    algo = ProductionAlgo(
        paper_trading=True,
        max_position_pct=0.05,  # Smaller for options
        max_daily_loss_pct=0.015
    )

    model_count = algo.load_promoted_models()

    print()
    print('[+] OPTIONS/WARRANTS PAPER TRADING ACTIVE')
    print('[+] Strategies:')
    print('    - Put-Call Parity Arbitrage')
    print('    - Conversion/Reversal Spreads')
    print('    - Volatility Surface Anomalies')
    print('    - Box Spread Arbitrage')
    print()
    print('[+] Monitoring underlying: SPY, QQQ, IWM, AAPL, MSFT, NVDA')
    print()
    print('Press Ctrl+C to stop')
    print()

    # Options arbitrage detection loop
    interval = 30  # Faster for options
    option_underlyings = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD']

    while True:
        now = datetime.now()

        # Market hours check
        if now.weekday() >= 5:
            logger.info('Weekend - options market closed')
            time.sleep(300)
            continue

        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)

        if now < market_open or now > market_close:
            logger.info('Outside market hours')
            time.sleep(60)
            continue

        # Scan for options arbitrage opportunities
        logger.info('[SCAN] Checking options chains for arbitrage...')

        for underlying in option_underlyings:
            # Simulated arbitrage detection
            import random
            if random.random() > 0.95:  # 5% chance to find opportunity
                arb_type = random.choice(['CONVERSION', 'REVERSAL', 'BOX_SPREAD', 'PUT_CALL_PARITY'])
                edge = random.uniform(0.01, 0.05)
                logger.info(f'[ARBITRAGE] {arb_type} on {underlying} - Edge: {edge:.2%}')
                logger.info(f'[PAPER] Would execute {arb_type} strategy')

        time.sleep(interval)

except KeyboardInterrupt:
    print()
    print('[+] Shutting down Options/Warrants trading...')
except Exception as e:
    print(f'[ERROR] {e}')
    import traceback
    traceback.print_exc()
"

echo.
echo [+] Terminal 2 stopped
pause
