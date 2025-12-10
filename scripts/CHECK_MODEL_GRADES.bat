@echo off
REM ============================================
REM CHECK MODEL GRADES
REM ============================================
REM
REM WHAT THIS DOES:
REM   Displays current model grades and performance:
REM   - Grade per model (S/A/B/C/D/F)
REM   - Performance metrics (Sharpe, returns, etc.)
REM   - Comparison vs institutional benchmarks
REM
REM HOW TO RUN:
REM   Option 1: Double-click this file
REM   Option 2: Open PowerShell/Terminal and run:
REM      cd C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1
REM      scripts\CHECK_MODEL_GRADES.bat
REM
REM ============================================

echo.
echo ========================================
echo    CHECKING MODEL GRADES
echo ========================================
echo.

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

REM Step 3: Run the dashboard
python scripts\model_dashboard.py

pause
