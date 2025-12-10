@echo off
REM ============================================================================
REM TERMINAL 1: EQUITIES & ETFs PAPER TRADING
REM ============================================================================
REM Alpha Loop Capital, LLC
REM
REM Asset Classes: US Equities, ETFs, Index Funds, ADRs
REM Strategies: MOMENTUM, VALUE, SCOUT, BOOKMAKER
REM Data Sources: Polygon, Alpha Vantage
REM ============================================================================

title ALC Terminal 1 - EQUITIES and ETFs

echo ============================================================
echo      TERMINAL 1: EQUITIES and ETFs PAPER TRADING
echo ============================================================
echo.
echo [INFO] Asset Classes: US Equities, ETFs, Index Funds, ADRs
echo [INFO] Strategies: MOMENTUM, VALUE, SCOUT, BOOKMAKER
echo [INFO] Mode: PAPER TRADING (Simulated)
echo [INFO] IBKR Port: 7497 (Paper)
echo.
echo ============================================================
echo.

cd /d "C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM"
call venv\Scripts\activate

echo [+] Environment ready
echo [+] Starting Equities/ETF trading agents...
echo.

REM Run with live data streaming for equities
python -c "
import sys
sys.path.insert(0, '.')

from loguru import logger
import time
from datetime import datetime

logger.remove()
logger.add(sys.stdout, level='INFO', format='{time:HH:mm:ss} | {level} | {message}')
logger.add('logs/paper_trade_equities.log', rotation='1 day', level='INFO')

print('='*60)
print('  EQUITIES AND ETF PAPER TRADING - TERMINAL 1')
print('='*60)
print()

# Import trading components
try:
    from src.training.live_data_trainer import LiveDataTrainer, LiveDataConfig, DataSource, TrainingMode
    from src.trading.production_algo import ProductionAlgo, check_readiness

    # Check readiness
    status = check_readiness()
    print(f'[>] Models loaded: {status[\"models\"][\"promoted\"]} promoted')
    print(f'[>] Data rows: {status[\"data\"][\"rows\"]:,}')
    print(f'[>] IBKR Connected: {status[\"ibkr\"][\"ready\"]}')
    print()

    # Initialize paper trading
    algo = ProductionAlgo(
        paper_trading=True,
        max_position_pct=0.10,
        max_daily_loss_pct=0.02
    )

    # Load models
    model_count = algo.load_promoted_models()

    if model_count == 0:
        print('[!] No promoted models - running in observation mode')
        print('[!] Train models first to enable trading signals')

    print()
    print('[+] EQUITIES/ETF PAPER TRADING ACTIVE')
    print('[+] Monitoring: SPY, QQQ, IWM, DIA, XLF, XLE, XLK...')
    print()
    print('Press Ctrl+C to stop')
    print()

    # Trading loop
    interval = 60
    while True:
        now = datetime.now()

        # Market hours check
        if now.weekday() >= 5:
            logger.info('Weekend - market closed')
            time.sleep(300)
            continue

        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)

        if now < market_open:
            wait_mins = (market_open - now).seconds // 60
            logger.info(f'Pre-market: {wait_mins} min until open')
            time.sleep(60)
            continue
        elif now > market_close:
            logger.info('After hours - market closed')
            time.sleep(60)
            continue

        # Process equity symbols
        equity_symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP']

        for symbol in equity_symbols:
            if symbol in algo.models:
                import pandas as pd
                action, confidence = algo.get_signal(symbol, pd.DataFrame())
                if action != 'HOLD' and confidence > 0.55:
                    logger.info(f'[SIGNAL] {action} {symbol} @ {confidence:.1%} confidence')
                    algo.execute_trade(symbol, action, confidence)

        time.sleep(interval)

except KeyboardInterrupt:
    print()
    print('[+] Shutting down Equities/ETF trading...')
except Exception as e:
    print(f'[ERROR] {e}')
    import traceback
    traceback.print_exc()
"

echo.
echo [+] Terminal 1 stopped
pause
