@echo off
REM ============================================
REM QUICK DATA HYDRATION (10-20 minutes)
REM ============================================
REM 
REM WHAT THIS DOES:
REM   Pulls a smaller subset for quick testing:
REM   - Top 500 stocks
REM   - Top 200 ETFs
REM   - Top 20 crypto
REM
REM HOW TO RUN:
REM   Option 1: Double-click this file
REM   Option 2: Open PowerShell/Terminal and run:
REM      cd C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1
REM      scripts\HYDRATE_QUICK.bat
REM
REM TIME: 10-20 minutes
REM ============================================

echo.
echo ========================================
echo    QUICK DATA HYDRATION
echo ========================================
echo.
echo WHAT THIS DOES:
echo   - Pulls Top 500 stocks from Polygon/Alpha Vantage
echo   - Pulls Top 200 ETFs
echo   - Pulls Top 20 crypto from Coinbase
echo   - Saves to Azure SQL database
echo.
echo TIME: 10-20 minutes
echo.
pause

REM Step 1: Navigate to project folder (relative to script location)
cd /d "%~dp0.."

REM Step 2: Activate Python virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo.
    echo ERROR: Virtual environment not found!
    echo.
    echo Run these commands first:
    echo   python -m venv venv
    echo   venv\Scripts\activate.bat
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Step 3: Load API keys from .env file
set DOTENV_PATH=C:\Users\tom\Alphaloopcapital Dropbox\ALC Tech Agents\API - Dec 2025.env
if exist "%DOTENV_PATH%" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%DOTENV_PATH%") do (
        set "%%a=%%b"
    )
) else (
    echo.
    echo WARNING: .env file not found at %DOTENV_PATH%
    echo Continuing with existing environment variables...
    echo.
)

REM Step 4: Run the hydration script
python scripts\hydrate_full_universe.py --quick

echo.
echo ========================================
echo    QUICK HYDRATION COMPLETE
echo ========================================
echo.
echo NEXT STEPS:
echo   1. Run TRAIN_MASSIVE.bat to train models
echo   2. Or run CHECK_MODEL_GRADES.bat to see results
echo.
pause
