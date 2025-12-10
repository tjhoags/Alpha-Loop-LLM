@echo off
REM ============================================================================
REM FULL UNIVERSE DATA INGESTION - All US Stocks and Options
REM ============================================================================
REM Alpha Loop Capital, LLC
REM
REM Data Sources:
REM   - Polygon.io: Stocks, ETFs, Options
REM   - Massive S3: Polygon flat files (bulk historical)
REM   - Alpha Vantage: Stocks, Forex, Fundamentals
REM   - IBKR: Real-time quotes and options chains
REM   - Coinbase: Crypto prices
REM   - FRED: Macro indicators
REM
REM USAGE:
REM   INGEST_ALL_DATA.bat           - Run once (all sources)
REM   INGEST_ALL_DATA.bat continuous - Run continuously (60s interval)
REM ============================================================================

echo ============================================================
echo       ALPHA LOOP CAPITAL - FULL UNIVERSE INGESTION
echo ============================================================
echo.
echo [INFO] Data Sources:
echo        - Polygon.io (stocks, options, ETFs)
echo        - Massive S3 (bulk historical data)
echo        - Alpha Vantage (premium data + fundamentals)
echo        - IBKR (if TWS running - real-time quotes)
echo        - Coinbase (crypto)
echo        - FRED (macro indicators)
echo.
echo ============================================================
echo.

cd /d "C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM"
call .venv\Scripts\activate

if "%1"=="continuous" (
    echo [+] Starting CONTINUOUS ingestion (60 second intervals)...
    echo.
    python scripts/ingest_full_universe.py --all --continuous --interval 60 --lookback 30 --workers 10
) else (
    echo [+] Starting SINGLE ingestion run...
    echo.
    python scripts/ingest_full_universe.py --all --lookback 30 --workers 10
)

echo.
echo [+] Ingestion complete
pause
