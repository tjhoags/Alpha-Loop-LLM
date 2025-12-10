@echo off
REM ============================================
REM CHECK TRAINING STATUS
REM ============================================
REM
REM WHAT THIS DOES:
REM   Shows comprehensive training status:
REM   - Total models trained
REM   - Models promoted to production
REM   - Active training processes
REM   - Data status
REM   - Launch readiness
REM
REM HOW TO RUN:
REM   Double-click this file
REM
REM ============================================

echo.
echo ========================================
echo    TRAINING STATUS DASHBOARD
echo ========================================
echo.

REM Navigate to project folder
cd /d "%~dp0.."

REM Activate Python virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    pause
    exit /b 1
)

REM Run the status dashboard
python scripts\training_status.py

pause

