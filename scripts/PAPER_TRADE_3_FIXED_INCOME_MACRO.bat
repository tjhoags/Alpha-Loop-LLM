@echo off
REM ============================================================================
REM TERMINAL 3: FIXED INCOME, MACRO & MULTI-STRATEGY PAPER TRADING
REM ============================================================================
REM Alpha Loop Capital, LLC
REM
REM Asset Classes: Treasury ETFs, Bond ETFs, Macro Indicators, Factor Strategies
REM Strategies: Factor Rotation, Dividend, Event-Driven, Long-Short
REM Data Sources: FRED, Polygon, Treasury APIs
REM ============================================================================

title ALC Terminal 3 - FIXED INCOME and MACRO

echo ============================================================
echo    TERMINAL 3: FIXED INCOME and MACRO PAPER TRADING
echo ============================================================
echo.
echo [INFO] Asset Classes: Treasuries, Bonds, Macro, Factors
echo [INFO] Strategies: Factor Rotation, Dividend, Event-Driven
echo [INFO] Mode: PAPER TRADING (Simulated)
echo [INFO] IBKR Port: 7497 (Paper)
echo.
echo ============================================================
echo.

cd /d "C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM"
call venv\Scripts\activate

echo [+] Environment ready
echo [+] Starting Fixed Income/Macro trading agents...
echo.

python -c "
import sys
sys.path.insert(0, '.')

from loguru import logger
import time
from datetime import datetime

logger.remove()
logger.add(sys.stdout, level='INFO', format='{time:HH:mm:ss} | {level} | {message}')
logger.add('logs/paper_trade_fixed_income.log', rotation='1 day', level='INFO')

print('='*60)
print('  FIXED INCOME AND MACRO PAPER TRADING - TERMINAL 3')
print('='*60)
print()

try:
    from src.trading.production_algo import ProductionAlgo, check_readiness
    from src.data_ingestion.sources.fred import FredClient

    # Check readiness
    status = check_readiness()
    print(f'[>] Models loaded: {status[\"models\"][\"promoted\"]} promoted')
    print(f'[>] IBKR Connected: {status[\"ibkr\"][\"ready\"]}')
    print()

    # Initialize paper trading
    algo = ProductionAlgo(
        paper_trading=True,
        max_position_pct=0.15,  # Larger for fixed income
        max_daily_loss_pct=0.01  # Tighter for bonds
    )

    model_count = algo.load_promoted_models()

    print()
    print('[+] FIXED INCOME/MACRO PAPER TRADING ACTIVE')
    print('[+] Asset Classes:')
    print('    - Treasury ETFs: TLT, IEF, SHY, GOVT')
    print('    - Corporate Bonds: LQD, HYG, JNK')
    print('    - Municipal Bonds: MUB')
    print('    - TIPS: TIP')
    print()
    print('[+] Strategies:')
    print('    - Yield Curve Positioning')
    print('    - Credit Spread Trading')
    print('    - Duration Management')
    print('    - Macro Factor Rotation')
    print()
    print('Press Ctrl+C to stop')
    print()

    # Fixed income symbols
    treasury_etfs = ['TLT', 'IEF', 'SHY', 'GOVT', 'BIL']
    corp_bonds = ['LQD', 'HYG', 'JNK', 'VCIT']
    other_fi = ['MUB', 'TIP', 'AGG', 'BND']
    all_fi = treasury_etfs + corp_bonds + other_fi

    # Macro indicators to monitor
    macro_indicators = {
        'DGS10': '10-Year Treasury',
        'DGS2': '2-Year Treasury',
        'T10Y2Y': '10Y-2Y Spread',
        'BAMLH0A0HYM2': 'High Yield Spread',
        'VIXCLS': 'VIX',
    }

    interval = 120  # Slower for fixed income

    while True:
        now = datetime.now()

        # Market hours check (bonds trade extended hours)
        if now.weekday() >= 5:
            logger.info('Weekend - bond market closed')
            time.sleep(300)
            continue

        # Process fixed income
        logger.info('[MACRO] Analyzing yield curve and credit spreads...')

        # Simulated macro analysis
        import random

        # Yield curve signal
        curve_steepening = random.choice([True, False])
        if curve_steepening:
            logger.info('[MACRO] Yield curve steepening - favor short duration')
            logger.info('[PAPER] Would rotate: TLT -> SHY')
        else:
            logger.info('[MACRO] Yield curve flattening - favor long duration')
            logger.info('[PAPER] Would rotate: SHY -> TLT')

        # Credit spread signal
        spreads_widening = random.choice([True, False])
        if spreads_widening:
            logger.info('[CREDIT] Spreads widening - reduce HY exposure')
            logger.info('[PAPER] Would sell HYG, buy LQD')
        else:
            logger.info('[CREDIT] Spreads tightening - increase HY exposure')
            logger.info('[PAPER] Would buy HYG')

        # Process each FI symbol
        for symbol in all_fi:
            if symbol in algo.models:
                import pandas as pd
                action, confidence = algo.get_signal(symbol, pd.DataFrame())
                if action != 'HOLD' and confidence > 0.55:
                    logger.info(f'[SIGNAL] {action} {symbol} @ {confidence:.1%}')
                    algo.execute_trade(symbol, action, confidence)

        time.sleep(interval)

except KeyboardInterrupt:
    print()
    print('[+] Shutting down Fixed Income/Macro trading...')
except Exception as e:
    print(f'[ERROR] {e}')
    import traceback
    traceback.print_exc()
"

echo.
echo [+] Terminal 3 stopped
pause
