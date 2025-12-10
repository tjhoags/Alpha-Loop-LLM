@echo off
echo ========================================
echo    *** LIVE TRADING - REAL MONEY ***
echo ========================================
echo.
echo WARNING: This will trade with REAL money!
echo Make sure:
echo   1. Models are promoted (check with CHECK_TRADING_READY.bat)
echo   2. IBKR TWS/Gateway is running
echo   3. You have reviewed the model grades
echo.
set /p confirm="Type YES to start live trading: "
if "%confirm%"=="YES" (
    REM Navigate to project folder (relative to script location)
    cd /d "%~dp0.."
    
    REM Activate Python virtual environment
    if exist "venv\Scripts\activate.bat" (
        call venv\Scripts\activate.bat
    ) else (
        echo.
        echo ERROR: Virtual environment not found!
        echo Run: python -m venv venv
        echo Then: venv\Scripts\activate.bat
        echo Then: pip install -r requirements.txt
        pause
        exit /b 1
    )
    
    REM Load API keys from .env file
    set DOTENV_PATH=C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env
    if exist "%DOTENV_PATH%" (
        for /f "usebackq tokens=1,* delims==" %%a in ("%DOTENV_PATH%") do (
            set "%%a=%%b"
        )
    )
    
    REM Start live trading
    echo.
    echo Starting LIVE TRADING...
    echo.
    python src\trading\production_algo.py --live
) else (
    echo.
    echo Aborted. No trades executed.
)
pause


