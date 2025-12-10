@echo off
REM =============================================================================
REM DATA COLLECTION - Run this FIRST before training
REM =============================================================================
echo.
echo ========================================
echo   ALPHA LOOP DATA COLLECTION
echo ========================================
echo.
echo This pulls data from Polygon, Alpha Vantage, Coinbase into SQL
echo.
pause

cd /d "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
call venv\Scripts\activate.bat
python src/data_ingestion/collector.py
pause

