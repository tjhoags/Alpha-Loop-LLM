@echo off
REM ================================================================================
REM ALPHA-LOOP-LLM CODE REVIEW LAUNCHER
REM ================================================================================
REM
REM HOW TO USE:
REM -----------
REM 1. Double-click this file, or
REM 2. Open PowerShell and run: .\scripts\REVIEW_CODE.bat
REM
REM OPTIONS:
REM --------
REM --scope=src/agents    Review only specific directory
REM --fix                 Automatically apply approved changes
REM --dry-run             Show changes without applying
REM --consensus=3         Require 3 agents to agree (default: 2)
REM --strict              Require unanimous agreement
REM
REM ================================================================================

echo ======================================================================
echo ALPHA-LOOP-LLM MULTI-AGENT CODE REVIEW
echo ======================================================================
echo.

cd /d "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting code review with multi-agent consensus...
echo.
echo Options: %*
echo.

python -m src.review.orchestrator %*

echo.
echo ======================================================================
echo REVIEW COMPLETE
echo ======================================================================
echo.
echo Check logs\review\ for detailed reports.
echo.

pause
