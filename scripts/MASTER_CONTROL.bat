@echo off
REM ============================================================================
REM ALPHA LOOP CAPITAL - MASTER CONTROL CENTER
REM ============================================================================
REM
REM One-stop launcher for all system components:
REM   1. Data Ingestion (all sources)
REM   2. Model Training (all strategies)
REM   3. Paper Trading (all asset classes)
REM   4. Dashboard (monitoring)
REM
REM ============================================================================

:MENU
cls
echo ============================================================
echo       ALPHA LOOP CAPITAL - MASTER CONTROL CENTER
echo ============================================================
echo.
echo   [1] Launch Dashboard (monitoring)
echo   [2] Launch Data Ingestion (all sources)
echo   [3] Launch Model Training (all strategies)
echo   [4] Launch Paper Trading (all terminals)
echo   [5] Launch EVERYTHING (full system)
echo.
echo   [6] Quick Data Refresh (single run)
echo   [7] Check System Status
echo.
echo   [0] Exit
echo.
echo ============================================================
set /p choice="Enter choice [0-7]: "

if "%choice%"=="1" goto DASHBOARD
if "%choice%"=="2" goto INGEST
if "%choice%"=="3" goto TRAIN
if "%choice%"=="4" goto TRADE
if "%choice%"=="5" goto EVERYTHING
if "%choice%"=="6" goto QUICKDATA
if "%choice%"=="7" goto STATUS
if "%choice%"=="0" goto EXIT
goto MENU

:DASHBOARD
echo.
echo [+] Launching Dashboard...
start "ALC Dashboard" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/dashboard.py --web --port 5000"
echo [+] Dashboard launched at http://localhost:5000
timeout /t 3
goto MENU

:INGEST
echo.
echo [+] Launching Data Ingestion (3 terminals)...
call scripts\LAUNCH_DATA_INGESTION.bat
goto MENU

:TRAIN
echo.
echo [+] Launching Model Training (3 terminals)...
call scripts\LAUNCH_ALL_TRAINING.bat
goto MENU

:TRADE
echo.
echo [+] Launching Paper Trading (4 terminals)...
call scripts\LAUNCH_ALL_PAPER_TRADING.bat
goto MENU

:EVERYTHING
echo.
echo [+] Launching FULL SYSTEM...
echo.

REM Dashboard first
start "ALC Dashboard" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/dashboard.py --web --port 5000"
echo [+] Dashboard launched
timeout /t 2 > nul

REM Data Ingestion
start "ALC Ingest - Polygon" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/ingest_full_universe.py --stocks --options --massive --continuous --interval 60"
echo [+] Polygon/Massive ingestion launched
timeout /t 1 > nul

start "ALC Ingest - AlphaVantage" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/ingest_full_universe.py --alpha-vantage --macro --continuous --interval 120"
echo [+] Alpha Vantage/FRED ingestion launched
timeout /t 1 > nul

start "ALC Ingest - IBKR" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/ingest_full_universe.py --ibkr --crypto --continuous --interval 30"
echo [+] IBKR/Crypto ingestion launched
timeout /t 1 > nul

REM Training
start "ALC Train - Strategies" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/train_all_strategies.py --continuous --interval 15 --parallel"
echo [+] Strategy training launched
timeout /t 1 > nul

start "ALC Train - Options" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/train_options_strategies.py --continuous --interval 15 --parallel"
echo [+] Options training launched
timeout /t 1 > nul

start "ALC Train - SmallMidCap" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/train_small_mid_cap.py --continuous --interval 15 --full-universe"
echo [+] Small/Mid Cap training launched
timeout /t 1 > nul

REM Paper Trading
start "ALC Trade - Equities" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM\scripts && PAPER_TRADE_1_EQUITIES_ETFS.bat"
echo [+] Equities trading launched
timeout /t 1 > nul

start "ALC Trade - Options" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM\scripts && PAPER_TRADE_2_OPTIONS_WARRANTS.bat"
echo [+] Options trading launched
timeout /t 1 > nul

start "ALC Trade - FixedIncome" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM\scripts && PAPER_TRADE_3_FIXED_INCOME_MACRO.bat"
echo [+] Fixed Income trading launched
timeout /t 1 > nul

start "ALC Trade - Crypto" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM\scripts && PAPER_TRADE_4_CRYPTO_ALT.bat"
echo [+] Crypto trading launched

echo.
echo ============================================================
echo       FULL SYSTEM LAUNCHED - 11 TERMINALS ACTIVE
echo ============================================================
echo.
echo   Dashboard: http://localhost:5000
echo.
echo   Data Ingestion: 3 terminals
echo   Model Training: 3 terminals
echo   Paper Trading:  4 terminals
echo.
echo   Press any key to return to menu...
pause > nul
goto MENU

:QUICKDATA
echo.
echo [+] Running quick data refresh (single pass)...
cd /d "C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM"
call .venv\Scripts\activate
python scripts/ingest_full_universe.py --all --lookback 7 --max-stocks 100
echo.
echo [+] Data refresh complete
pause
goto MENU

:STATUS
echo.
echo [+] Checking system status...
cd /d "C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM"
call .venv\Scripts\activate
python -c "
from src.config.settings import get_settings
settings = get_settings()
print()
print('API CONFIGURATION STATUS')
print('=' * 50)
api_status = settings.validate_required_apis()
for api, configured in api_status.items():
    status = '[OK]' if configured else '[MISSING]'
    print(f'  {api:20s} {status}')
print()

# Check models
from pathlib import Path
models_dir = settings.models_dir
if models_dir.exists():
    models = list(models_dir.glob('*.pkl'))
    print(f'Models: {len(models)} trained')
else:
    print('Models: 0 (directory missing)')
print()

# Check database
try:
    from src.database.connection import get_engine
    import pandas as pd
    engine = get_engine()
    df = pd.read_sql('SELECT COUNT(*) as cnt FROM price_bars', engine)
    print(f'Database: {df[\"cnt\"].iloc[0]:,} price bars')
except Exception as e:
    print(f'Database: Error - {e}')

# Check IBKR
import socket
for port, name in [(7497, 'Paper'), (7496, 'Live')]:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', port))
        status = 'CONNECTED' if result == 0 else 'offline'
        sock.close()
    except:
        status = 'error'
    print(f'IBKR {name} ({port}): {status}')
"
echo.
pause
goto MENU

:EXIT
echo.
echo Goodbye!
exit /b 0
