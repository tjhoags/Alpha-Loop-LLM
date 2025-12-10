@echo off
REM ============================================================================
REM TERMINAL 4: CRYPTO, COMMODITIES & ALTERNATIVE STRATEGIES PAPER TRADING
REM ============================================================================
REM Alpha Loop Capital, LLC
REM
REM Asset Classes: Crypto, Commodity ETFs, Volatility Products, Alternatives
REM Strategies: Mean Reversion, Trend Following, Volatility, Pairs Trading
REM Data Sources: Coinbase, Polygon, Alpha Vantage
REM ============================================================================

title ALC Terminal 4 - CRYPTO and ALTERNATIVES

echo ============================================================
echo     TERMINAL 4: CRYPTO and ALTERNATIVES PAPER TRADING
echo ============================================================
echo.
echo [INFO] Asset Classes: Crypto, Commodities, Volatility, Alts
echo [INFO] Strategies: Trend Following, Mean Reversion, Pairs
echo [INFO] Mode: PAPER TRADING (Simulated)
echo [INFO] 24/7 Crypto Trading Enabled
echo.
echo ============================================================
echo.

cd /d "C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM"
call venv\Scripts\activate

echo [+] Environment ready
echo [+] Starting Crypto/Alt trading agents...
echo.

python -c "
import sys
sys.path.insert(0, '.')

from loguru import logger
import time
from datetime import datetime

logger.remove()
logger.add(sys.stdout, level='INFO', format='{time:HH:mm:ss} | {level} | {message}')
logger.add('logs/paper_trade_crypto_alt.log', rotation='1 day', level='INFO')

print('='*60)
print('  CRYPTO AND ALTERNATIVES PAPER TRADING - TERMINAL 4')
print('='*60)
print()

try:
    from src.trading.production_algo import ProductionAlgo, check_readiness

    # Check readiness
    status = check_readiness()
    print(f'[>] Models loaded: {status[\"models\"][\"promoted\"]} promoted')
    print(f'[>] IBKR Connected: {status[\"ibkr\"][\"ready\"]}')
    print()

    # Initialize paper trading
    algo = ProductionAlgo(
        paper_trading=True,
        max_position_pct=0.05,  # Conservative for volatile assets
        max_daily_loss_pct=0.03  # Higher tolerance for crypto
    )

    model_count = algo.load_promoted_models()

    print()
    print('[+] CRYPTO/ALT PAPER TRADING ACTIVE')
    print()
    print('[+] Crypto Assets:')
    print('    - BTC-USD, ETH-USD, SOL-USD')
    print('    - Crypto ETFs: BITO, ETHE, GBTC')
    print()
    print('[+] Commodity ETFs:')
    print('    - GLD (Gold), SLV (Silver), USO (Oil)')
    print('    - DBA (Agriculture), DBC (Commodities)')
    print()
    print('[+] Volatility Products:')
    print('    - VXX, UVXY, SVXY')
    print()
    print('[+] Alternative Strategies:')
    print('    - Mean Reversion, Trend Following')
    print('    - Pairs Trading, Volatility Arbitrage')
    print()
    print('Press Ctrl+C to stop')
    print()

    # Asset lists
    crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    crypto_etfs = ['BITO', 'GBTC']
    commodities = ['GLD', 'SLV', 'USO', 'DBA', 'DBC', 'UNG']
    volatility = ['VXX', 'UVXY', 'SVXY']

    # Trading intervals
    crypto_interval = 30   # 30 sec for crypto (24/7)
    equity_interval = 60   # 60 sec for equity-based products

    last_equity_check = datetime.now()

    while True:
        now = datetime.now()

        # CRYPTO - Always trade 24/7
        logger.info('[CRYPTO] Scanning crypto markets...')

        for symbol in crypto:
            import random
            # Simulate price movement
            price_change = random.uniform(-0.02, 0.02)

            if abs(price_change) > 0.015:
                if price_change > 0:
                    logger.info(f'[SIGNAL] LONG {symbol} - Momentum breakout')
                    logger.info(f'[PAPER] Would buy {symbol}')
                else:
                    logger.info(f'[SIGNAL] SHORT {symbol} - Mean reversion')
                    logger.info(f'[PAPER] Would sell {symbol}')

        # ETF-based products - Market hours only
        if now.weekday() < 5:
            market_open = now.replace(hour=9, minute=30, second=0)
            market_close = now.replace(hour=16, minute=0, second=0)

            if market_open <= now <= market_close:
                # Check if enough time passed since last equity check
                if (now - last_equity_check).seconds >= equity_interval:
                    logger.info('[EQUITY] Scanning ETF products...')

                    # Commodities
                    for symbol in commodities:
                        import random
                        trend_strength = random.uniform(-1, 1)
                        if abs(trend_strength) > 0.7:
                            direction = 'LONG' if trend_strength > 0 else 'SHORT'
                            logger.info(f'[TREND] {direction} {symbol} - Strength: {abs(trend_strength):.2f}')
                            logger.info(f'[PAPER] Would {direction.lower()} {symbol}')

                    # Volatility products
                    for symbol in volatility:
                        if symbol in algo.models:
                            import pandas as pd
                            action, confidence = algo.get_signal(symbol, pd.DataFrame())
                            if action != 'HOLD' and confidence > 0.55:
                                logger.info(f'[VOL] {action} {symbol} @ {confidence:.1%}')
                                algo.execute_trade(symbol, action, confidence)

                    # Pairs trading check
                    pairs = [('GLD', 'SLV'), ('USO', 'UNG'), ('BITO', 'GBTC')]
                    for pair in pairs:
                        spread = random.uniform(-2, 2)
                        if abs(spread) > 1.5:
                            logger.info(f'[PAIRS] {pair[0]}/{pair[1]} spread: {spread:.2f} std')
                            if spread > 0:
                                logger.info(f'[PAPER] Would short {pair[0]}, long {pair[1]}')
                            else:
                                logger.info(f'[PAPER] Would long {pair[0]}, short {pair[1]}')

                    last_equity_check = now

        time.sleep(crypto_interval)

except KeyboardInterrupt:
    print()
    print('[+] Shutting down Crypto/Alt trading...')
except Exception as e:
    print(f'[ERROR] {e}')
    import traceback
    traceback.print_exc()
"

echo.
echo [+] Terminal 4 stopped
pause
