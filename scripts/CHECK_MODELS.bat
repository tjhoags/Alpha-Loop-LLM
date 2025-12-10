@echo off
REM =============================================================================
REM CHECK TRAINED MODELS
REM =============================================================================
echo.
echo ========================================
echo   TRAINED MODELS STATUS
echo ========================================
echo.

cd /d "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"

echo Models in folder:
dir /b models\*.pkl 2>nul || echo No models found yet!

echo.
echo Total count:
dir /b models\*.pkl 2>nul | find /c /v ""

echo.
echo ========================================
echo.
pause

