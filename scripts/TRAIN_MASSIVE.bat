@echo off
REM ============================================
REM MASSIVE PARALLEL TRAINING
REM ============================================
REM Trains ML models on ENTIRE universe:
REM - All symbols in database
REM - Technical + Behavioral features
REM - XGBoost, LightGBM, CatBoost
REM - Auto-checkpoint/resume
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

cd /d "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
call venv\Scripts\activate.bat

set DOTENV_PATH=C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env
for /f "usebackq tokens=1,* delims==" %%a in ("%DOTENV_PATH%") do (
    set "%%a=%%b"
)

REM Use all cores except 1
python src\ml\massive_trainer.py --batch-size 50

echo.
echo TRAINING SESSION ENDED
echo Run again to resume from checkpoint
pause


