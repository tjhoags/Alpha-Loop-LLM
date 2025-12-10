@echo off
echo ================================================================================
echo ALPHA LOOP CAPITAL - AGENT CHAT INTERFACE
echo ================================================================================
echo.
echo Starting agent communication interface...
echo.

cd /d "%~dp0.."

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found. Using system Python.
)

echo.
echo Starting web interface on http://localhost:8080
echo.
echo Agents available:
echo   - SANTAS_HELPER (Fund Operations)
echo   - CPA (Tax ^& Audit)
echo.
echo Press Ctrl+C to stop.
echo.

python -m src.ui.agent_chat --port 8080

pause

