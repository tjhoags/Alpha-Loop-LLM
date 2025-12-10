@echo off
echo ========================================
echo    TRADING READINESS CHECK
echo ========================================
cd /d "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
call venv\Scripts\activate
python src/trading/production_algo.py --check
pause


