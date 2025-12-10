@echo off
echo ================================================================================
echo ALPHA LOOP CAPITAL - CHRIS FRIEDMAN AGENT INTERFACE
echo ================================================================================
echo.
echo Starting the Fund Operations communication portal...
echo.
echo Chris (and Tom) can communicate directly with:
echo   - SANTAS_HELPER: Fund Accounting Operations Lead
echo   - CPA: Tax, Audit, Reporting ^& Compliance Authority
echo.
echo The interface will be available at: http://127.0.0.1:5000
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
echo Launching web interface...
echo Press Ctrl+C to stop the server.
echo.

python -m src.interfaces.chris_interface.web_app --host 127.0.0.1 --port 5000

pause

