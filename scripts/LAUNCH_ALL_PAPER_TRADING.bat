@echo off
REM ============================================================================
REM MASTER LAUNCHER: ALL 4 PAPER TRADING TERMINALS
REM ============================================================================
REM Alpha Loop Capital, LLC
REM
REM Launches 4 separate terminal windows for paper trading:
REM   Terminal 1: Equities and ETFs
REM   Terminal 2: Options and Warrants
REM   Terminal 3: Fixed Income and Macro
REM   Terminal 4: Crypto and Alternatives
REM
REM USAGE: Double-click this file to launch all 4 terminals
REM ============================================================================

echo ============================================================
echo       ALPHA LOOP CAPITAL - PAPER TRADING LAUNCHER
echo ============================================================
echo.
echo [INFO] This will launch 4 paper trading terminals
echo [INFO] Each terminal covers different asset classes
echo.
echo [WARNING] Make sure IBKR TWS/Gateway is running on port 7497
echo.
echo ============================================================
echo.
echo Press any key to launch all 4 terminals...
pause > nul

echo.
echo [+] Launching Terminal 1: Equities and ETFs...
start "ALC Terminal 1 - Equities" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM\scripts && PAPER_TRADE_1_EQUITIES_ETFS.bat"

timeout /t 2 > nul

echo [+] Launching Terminal 2: Options and Warrants...
start "ALC Terminal 2 - Options" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM\scripts && PAPER_TRADE_2_OPTIONS_WARRANTS.bat"

timeout /t 2 > nul

echo [+] Launching Terminal 3: Fixed Income and Macro...
start "ALC Terminal 3 - Fixed Income" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM\scripts && PAPER_TRADE_3_FIXED_INCOME_MACRO.bat"

timeout /t 2 > nul

echo [+] Launching Terminal 4: Crypto and Alternatives...
start "ALC Terminal 4 - Crypto" cmd /k "cd /d C:\Users\tom\OneDrive\Alpha Loop LLM\alpha-loop-llm\Alpha-Loop-LLM\scripts && PAPER_TRADE_4_CRYPTO_ALT.bat"

echo.
echo ============================================================
echo       ALL 4 TERMINALS LAUNCHED SUCCESSFULLY
echo ============================================================
echo.
echo [+] Terminal 1: Equities, ETFs, Index Funds
echo [+] Terminal 2: Options, Warrants, Derivatives
echo [+] Terminal 3: Fixed Income, Bonds, Macro
echo [+] Terminal 4: Crypto, Commodities, Volatility
echo.
echo [INFO] Close individual terminal windows to stop trading
echo [INFO] Or press Ctrl+C in each terminal
echo.
echo This window will close in 10 seconds...
timeout /t 10
