@echo off
REM ============================================
REM FULL UNIVERSE DATA HYDRATION
REM ============================================
REM This pulls ALL data from Polygon:
REM - 8,000+ stocks
REM - 2,500+ ETFs  
REM - 100+ crypto
REM - Options chains
REM 
REM WARNING: Takes 6-12 hours for full universe!
REM ============================================

echo.
echo ========================================
echo    FULL UNIVERSE DATA HYDRATION
echo ========================================
echo.
echo This will pull:
echo   - ALL US stocks (8,000+)
echo   - ALL ETFs (2,500+)
echo   - Crypto pairs
echo   - Options chains
echo.
echo WARNING: This takes 6-12 HOURS!
echo.
pause

cd /d "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Load environment
set DOTENV_PATH=C:\Users\tom\Alphaloopcapital Dropbox\ALC Tech Agents\API - Dec 2025.env
for /f "usebackq tokens=1,* delims==" %%a in ("%DOTENV_PATH%") do (
    set "%%a=%%b"
)

REM Run full hydration
python scripts\hydrate_full_universe.py

echo.
echo ========================================
echo    HYDRATION COMPLETE
echo ========================================
pause


