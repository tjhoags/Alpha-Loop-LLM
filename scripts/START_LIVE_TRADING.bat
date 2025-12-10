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
    cd /d "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
    call venv\Scripts\activate
    python src/trading/production_algo.py --live
) else (
    echo Aborted.
)
pause


