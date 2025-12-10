@echo off
REM ============================================
REM QUICK DATA HYDRATION (10-20 minutes)
REM ============================================
REM Pulls a smaller subset for quick testing:
REM - Top 500 stocks
REM - Top 200 ETFs
REM - Top 20 crypto
REM ============================================

echo.
echo ========================================
echo    QUICK DATA HYDRATION
echo ========================================
echo.
echo This will pull (10-20 min):
echo   - Top 500 stocks
echo   - Top 200 ETFs
echo   - Top 20 crypto
echo.

cd /d "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
call venv\Scripts\activate.bat

set DOTENV_PATH=C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env
for /f "usebackq tokens=1,* delims==" %%a in ("%DOTENV_PATH%") do (
    set "%%a=%%b"
)

python scripts\hydrate_full_universe.py --quick

echo.
echo QUICK HYDRATION COMPLETE
pause


