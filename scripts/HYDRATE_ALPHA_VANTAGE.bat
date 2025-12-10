@echo off
REM ============================================================================
REM ALPHA VANTAGE DATA HYDRATION
REM ============================================================================
REM This script pulls market data from Alpha Vantage Premium API.
REM
REM WHAT IT PULLS:
REM   - Stock daily prices (20+ years history)
REM   - Stock fundamentals (30+ valuation metrics)
REM   - Forex pairs (10 major pairs)
REM
REM RATE LIMITS: 75 calls/minute (Premium tier)
REM EXPECTED TIME: 2-4 hours for full hydration
REM
REM OUTPUTS:
REM   - SQL: price_bars, fundamentals, forex_bars tables
REM   - CSV Backup: data/csv_backup/alpha_vantage/
REM   - Log: logs/alpha_vantage_hydration.log
REM ============================================================================

echo ============================================================================
echo ALPHA VANTAGE DATA HYDRATION
echo ============================================================================
echo.

cd /d "%~dp0.."

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Choose an option:
echo   1. Full hydration (2-4 hours)
echo   2. Quick test (10 stocks, ~15 minutes)
echo   3. Stocks only
echo   4. Fundamentals only
echo   5. Forex only
echo.

set /p choice="Enter choice (1-5): "

if "%choice%"=="1" (
    echo Running full hydration...
    python scripts/hydrate_alpha_vantage.py
) else if "%choice%"=="2" (
    echo Running quick test...
    python scripts/hydrate_alpha_vantage.py --quick
) else if "%choice%"=="3" (
    echo Hydrating stocks...
    python scripts/hydrate_alpha_vantage.py --stocks-only
) else if "%choice%"=="4" (
    echo Hydrating fundamentals...
    python scripts/hydrate_alpha_vantage.py --fundamentals-only
) else if "%choice%"=="5" (
    echo Hydrating forex...
    python scripts/hydrate_alpha_vantage.py --forex-only
) else (
    echo Invalid choice. Running full hydration...
    python scripts/hydrate_alpha_vantage.py
)

echo.
echo ============================================================================
echo Hydration complete! Check logs/alpha_vantage_hydration.log for details.
echo ============================================================================

pause

