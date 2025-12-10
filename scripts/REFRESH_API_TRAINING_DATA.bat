@echo off
REM ============================================================================
REM REFRESH API TRAINING DATA - Pull fresh data from all sources
REM ============================================================================
REM Alpha Loop Capital, LLC
REM
REM This script pulls fresh market data from all API sources:
REM   - Polygon.io (equities, options)
REM   - Alpha Vantage (stocks intraday)
REM   - FRED (macro indicators)
REM   - Coinbase (crypto)
REM   - SEC Edgar (filings)
REM ============================================================================

echo ============================================================
echo           ALPHA LOOP CAPITAL - DATA REFRESH
echo ============================================================
echo.
echo [INFO] Starting API training data refresh...
echo [INFO] Timestamp: %date% %time%
echo.

REM Navigate to project directory
cd /d "C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM"

REM Activate virtual environment
call venv\Scripts\activate

echo [+] Virtual environment activated
echo [+] Starting multi-source data collection...
echo.

REM Run the main data collector
python -c "from src.data_ingestion.collector import main; main()"

echo.
echo [+] Data collection complete!
echo.
echo ============================================================
echo           DATA REFRESH SUMMARY
echo ============================================================
echo.
echo Press any key to exit...
pause > nul
