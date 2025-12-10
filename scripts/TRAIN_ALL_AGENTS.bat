@echo off
REM ============================================================================
REM TRAIN ALL AGENTS
REM ============================================================================
REM This script trains ALL agents using the elite grading system.
REM 
REM WHAT IT DOES:
REM   1. Activates the virtual environment
REM   2. Runs the agent trainer on all configured agents
REM   3. Grades each agent using elite thresholds
REM   4. Saves results to data/training_results/
REM
REM EXPECTED TIME: 1-4 hours depending on data volume
REM
REM OUTPUTS:
REM   - Training results: data/training_results/*.json
REM   - Log file: logs/agent_training.log
REM ============================================================================

echo ============================================================================
echo AGENT TRAINING SYSTEM
echo ============================================================================
echo.

cd /d "%~dp0.."

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Starting agent training...
echo This will train all agents and grade them using elite thresholds.
echo.

python -m src.training.agent_trainer --all

echo.
echo ============================================================================
echo Training complete! Check logs/agent_training.log for details.
echo Results saved to data/training_results/
echo ============================================================================

pause

