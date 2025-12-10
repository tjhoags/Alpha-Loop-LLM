@echo off
echo ========================================
echo    STARTING PAPER TRADING
echo ========================================
echo.
echo This runs the algorithm with FAKE money.
echo Good for testing before going live.
echo.
cd /d "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
call venv\Scripts\activate
python src/trading/production_algo.py --paper


