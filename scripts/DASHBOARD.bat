@echo off
REM ============================================================================
REM ALPHA LOOP CAPITAL - DASHBOARD LAUNCHER
REM ============================================================================
REM
REM Options:
REM   DASHBOARD.bat           - Terminal dashboard (rich)
REM   DASHBOARD.bat web       - Web dashboard at http://localhost:5000
REM   DASHBOARD.bat simple    - Simple text dashboard
REM ============================================================================

echo ============================================================
echo       ALPHA LOOP CAPITAL - DASHBOARD
echo ============================================================
echo.

cd /d "C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM"
call .venv\Scripts\activate

if "%1"=="web" (
    echo [+] Launching WEB dashboard at http://localhost:5000
    echo [+] Open browser to: http://localhost:5000
    echo.
    python scripts/dashboard.py --web --port 5000
) else if "%1"=="simple" (
    echo [+] Launching SIMPLE text dashboard
    echo.
    python scripts/dashboard.py --simple --refresh 5
) else (
    echo [+] Launching TERMINAL dashboard (rich)
    echo [+] Install rich if needed: pip install rich
    echo.
    python scripts/dashboard.py --refresh 5
)

pause
