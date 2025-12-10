@echo off
REM ============================================
REM MASSIVE PARALLEL TRAINING
REM ============================================
REM 
REM WHAT THIS DOES:
REM   Trains ML models on ENTIRE universe:
REM   - All symbols in database
REM   - Technical + Behavioral features
REM   - XGBoost, LightGBM, CatBoost
REM   - Auto-checkpoint/resume
REM
REM HOW TO RUN:
REM   Option 1: Double-click this file
REM   Option 2: Open PowerShell/Terminal and run:
REM      cd C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1
REM      scripts\TRAIN_MASSIVE.bat
REM
REM TIME: Several hours (depends on data size)
REM ============================================

echo.
echo ========================================
echo    MASSIVE PARALLEL TRAINING
echo ========================================
echo.
echo This will train on ALL symbols:
echo   - 100+ features (technical + behavioral)
echo   - 3 model types per symbol
echo   - Auto-saves progress (can resume)
echo.
echo Press Ctrl+C anytime to pause
echo Run again to resume from checkpoint
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

REM Step 4: Run the training script (use all cores except 1)
python src\ml\massive_trainer.py --batch-size 50

echo.
echo ========================================
echo    TRAINING SESSION ENDED
echo ========================================
echo.
echo Run again to resume from checkpoint
echo.
pause
