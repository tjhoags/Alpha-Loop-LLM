@echo off
echo ================================================================================
echo ALPHA LOOP CAPITAL - FUND OPERATIONS AGENT TRAINING
echo ================================================================================
echo.
echo Training SANTAS_HELPER and CPA agents...
echo These agents report directly to Chris Friedman.
echo.
echo SANTAS_HELPER: Fund Accounting Operations Lead
echo CPA: Tax, Audit, Reporting ^& Compliance Authority
echo.
echo ================================================================================

cd /d "%~dp0\.."

:: Check if venv exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo WARNING: Virtual environment not found. Using system Python.
)

echo.
echo Starting training...
echo.

python -m src.training.fund_ops_training full --verbose

echo.
echo ================================================================================
echo Training complete!
echo.
echo To start the Chris Friedman communication interface:
echo   python -m src.interfaces.chris_interface.web_app
echo.
echo Or run: START_CHRIS_INTERFACE.bat
echo ================================================================================
pause

