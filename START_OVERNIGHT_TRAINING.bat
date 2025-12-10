@echo off
REM =============================================================================
REM DOUBLE-CLICK THIS FILE TO START OVERNIGHT TRAINING
REM =============================================================================
REM This will:
REM - Prevent your computer from sleeping
REM - Run training OUTSIDE of Cursor
REM - Auto-restart if it crashes
REM =============================================================================

echo.
echo ========================================
echo   ALPHA LOOP OVERNIGHT TRAINING
echo ========================================
echo.
echo This will run ALL NIGHT. Do NOT close this window!
echo.
echo Press any key to start training...
pause > nul

powershell -ExecutionPolicy Bypass -File "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1\scripts\overnight_training_robust.ps1"

