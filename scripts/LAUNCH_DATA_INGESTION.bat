@echo off
REM ============================================================================
REM DATA INGESTION LAUNCHER - Multiple Terminals
REM ============================================================================
REM Alpha Loop Capital, LLC
REM
REM Launches 3 data ingestion terminals:
REM   Terminal 1: Polygon + Massive (bulk data)
REM   Terminal 2: Alpha Vantage + FRED (premium + macro)
REM   Terminal 3: IBKR + Crypto (real-time)
REM ============================================================================

echo ============================================================
echo       ALPHA LOOP CAPITAL - DATA INGESTION LAUNCHER
echo ============================================================
echo.
echo [INFO] This will launch 3 data ingestion terminals
echo.
echo ============================================================
echo.
echo Press any key to launch all ingestion terminals...
pause > nul

cd /d "C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM"

echo.
echo [+] Launching Terminal 1: Polygon + Massive (stocks, options, bulk)...
start "ALC Ingest - Polygon/Massive" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/ingest_full_universe.py --stocks --options --massive --continuous --interval 60"

timeout /t 3 > nul

echo [+] Launching Terminal 2: Alpha Vantage + FRED (premium + macro)...
start "ALC Ingest - AlphaVantage/FRED" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/ingest_full_universe.py --alpha-vantage --macro --continuous --interval 120"

timeout /t 3 > nul

echo [+] Launching Terminal 3: IBKR + Crypto (real-time)...
start "ALC Ingest - IBKR/Crypto" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/ingest_full_universe.py --ibkr --crypto --continuous --interval 30"

echo.
echo ============================================================
echo       ALL 3 INGESTION TERMINALS LAUNCHED
echo ============================================================
echo.
echo [+] Terminal 1: Polygon + Massive (60s interval)
echo [+] Terminal 2: Alpha Vantage + FRED (120s interval)
echo [+] Terminal 3: IBKR + Crypto (30s interval)
echo.
echo [INFO] Make sure IBKR TWS/Gateway is running for Terminal 3
echo [INFO] Close individual windows to stop ingestion
echo.
echo This window will close in 10 seconds...
timeout /t 10
