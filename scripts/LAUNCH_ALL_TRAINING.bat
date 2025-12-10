@echo off
REM ============================================================================
REM MASTER TRAINING LAUNCHER - All Strategy and Options Training
REM ============================================================================
REM Alpha Loop Capital, LLC
REM
REM Launches training terminals:
REM   Terminal 1: Strategy Training (momentum, mean_reversion, value, etc.)
REM   Terminal 2: Options Training (conversion/reversal, put-call parity, etc.)
REM   Terminal 3: Small/Mid Cap Training (retail arbitrage, meme stocks)
REM
REM All run with 15-second intervals
REM ============================================================================

echo ============================================================
echo       ALPHA LOOP CAPITAL - TRAINING LAUNCHER
echo ============================================================
echo.
echo [INFO] This will launch 3 training terminals
echo [INFO] All running with 15-second intervals
echo.
echo ============================================================
echo.
echo Press any key to launch all training terminals...
pause > nul

cd /d "C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM"

echo.
echo [+] Launching Terminal 1: Strategy Training...
start "ALC Training - Strategies" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/train_all_strategies.py --continuous --interval 15 --parallel"

timeout /t 2 > nul

echo [+] Launching Terminal 2: Options Training...
start "ALC Training - Options" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/train_options_strategies.py --continuous --interval 15 --parallel"

timeout /t 2 > nul

echo [+] Launching Terminal 3: Small/Mid Cap Training...
start "ALC Training - Small/Mid Cap" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM && call .venv\Scripts\activate && python scripts/train_small_mid_cap.py --continuous --interval 15 --full-universe"

echo.
echo ============================================================
echo       ALL 3 TRAINING TERMINALS LAUNCHED
echo ============================================================
echo.
echo [+] Terminal 1: Strategy Training (7 strategies)
echo [+] Terminal 2: Options Training (6 strategies)
echo [+] Terminal 3: Small/Mid Cap (65 symbols)
echo.
echo [INFO] All training every 15 seconds
echo [INFO] Close individual windows to stop
echo.
echo This window will close in 10 seconds...
timeout /t 10
