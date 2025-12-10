@echo off
echo ========================================
echo    MODEL PERFORMANCE DASHBOARD
echo ========================================
cd /d "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
call venv\Scripts\activate
python scripts/model_dashboard.py
pause


