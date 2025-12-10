@echo off
REM =============================================================================
REM TRADING ENGINE - Run this at 9:15 AM after training completes
REM =============================================================================
echo.
echo ========================================
echo   ALPHA LOOP TRADING ENGINE
echo ========================================
echo.
echo WARNING: This will execute trades via IBKR!
echo Make sure TWS/Gateway is running first.
echo.
echo Port 7497 = PAPER TRADING (safe)
echo Port 7496 = LIVE TRADING (real money)
echo.
pause

cd /d "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
call venv\Scripts\activate.bat
python src/trading/execution_engine.py
pause

