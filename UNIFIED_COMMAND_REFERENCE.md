# UNIFIED COMMAND REFERENCE - Windows & Mac

## Complete Natural Language Guide for All Operations

This document provides step-by-step instructions in plain English for both Windows (PowerShell) and MacBook Pro (Terminal) users.

---

## TABLE OF CONTENTS

1. [Opening Your Terminal](#1-opening-your-terminal)
2. [Navigating to Project](#2-navigating-to-project)
3. [Setting Up Environment](#3-setting-up-environment)
4. [Installing Dependencies](#4-installing-dependencies)
5. [Copying Configuration Files](#5-copying-configuration-files)
6. [Testing Database Connection](#6-testing-database-connection)
7. [Starting Data Collection](#7-starting-data-collection)
8. [Running Model Training](#8-running-model-training)
9. [Starting Trading Engine](#9-starting-trading-engine)
10. [Monitoring & Logs](#10-monitoring--logs)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. OPENING YOUR TERMINAL

### Windows (PowerShell)

**Option A - Windows Terminal (Recommended):**
1. Press `Windows + X` on your keyboard
2. Click "Terminal" or "Windows Terminal" from the menu
3. A blue PowerShell window will open

**Option B - Search:**
1. Press `Windows key` and type "PowerShell"
2. Click "Windows PowerShell" or "Windows Terminal"

**Option C - From Cursor IDE:**
1. Open Cursor
2. Go to menu: Terminal → New Terminal
3. Terminal opens at bottom of IDE

### MacBook Pro (Terminal)

**Option A - Spotlight (Fastest):**
1. Press `Cmd + Space` to open Spotlight
2. Type "Terminal"
3. Press `Enter` when Terminal appears

**Option B - Finder:**
1. Open Finder
2. Go to Applications → Utilities → Terminal
3. Double-click Terminal

**Option C - From Cursor IDE:**
1. Open Cursor
2. Go to menu: Terminal → New Terminal
3. Terminal opens at bottom of IDE

---

## 2. NAVIGATING TO PROJECT

### Windows (PowerShell)

**Step 1: Change to project directory**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"
```

**Plain English:** Type `cd` (change directory), then the full path in quotes, then press Enter.

**Alternative path (if different location):**
```powershell
cd "C:\Users\tom\Alpha-Loop-LLM\Alpha-Loop-LLM-1"
```

**Verify you're in the right place:**
```powershell
Get-Location
```
This shows your current directory. You should see the project path.

### MacBook Pro (Terminal)

**Step 1: Change to project directory**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
```

**Plain English:** Type `cd` (change directory), then `~` (your home folder), then the path, then press Enter.

**Alternative paths:**
```bash
# If project is directly in home folder
cd ~/Alpha-Loop-LLM-1/gkv

# If project is in Documents
cd ~/Documents/Alpha-Loop-LLM/gkv
```

**Verify you're in the right place:**
```bash
pwd
```
This prints working directory. You should see the project path.

---

## 3. SETTING UP ENVIRONMENT

### Windows (PowerShell)

**Step 1: Create virtual environment**
```powershell
python -m venv venv
```
**Plain English:** This creates a folder called "venv" that will hold all Python packages separate from your system.

**Step 2: Activate the virtual environment**
```powershell
.\venv\Scripts\Activate.ps1
```
**Plain English:** This "turns on" the virtual environment. You'll see `(venv)` appear at the start of your prompt.

**If you get an execution policy error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
**Plain English:** This allows PowerShell to run local scripts. Type `Y` and press Enter when asked.

**Then try activating again:**
```powershell
.\venv\Scripts\Activate.ps1
```

### MacBook Pro (Terminal)

**Step 1: Create virtual environment**
```bash
python3 -m venv venv
```
**Plain English:** Same as Windows - creates isolated Python environment. Note: Mac uses `python3` not `python`.

**Step 2: Activate the virtual environment**
```bash
source venv/bin/activate
```
**Plain English:** This activates the environment. You'll see `(venv)` at the start of your prompt.

---

## 4. INSTALLING DEPENDENCIES

### Windows (PowerShell)

**Make sure venv is activated first (you should see (venv) in prompt)**

**Step 1: Upgrade pip (optional but recommended)**
```powershell
python -m pip install --upgrade pip
```

**Step 2: Install all required packages**
```powershell
pip install -r requirements.txt
```
**Plain English:** This reads the requirements.txt file and installs all listed packages. Takes 2-5 minutes.

### MacBook Pro (Terminal)

**Make sure venv is activated first (you should see (venv) in prompt)**

**Step 1: Upgrade pip (optional but recommended)**
```bash
python3 -m pip install --upgrade pip
```

**Step 2: Install all required packages**
```bash
pip install -r requirements.txt
```
**Plain English:** Same as Windows - installs all required Python packages.

---

## 5. COPYING CONFIGURATION FILES

### Windows (PowerShell)

**Step 1: Copy the .env file from OneDrive**
```powershell
Copy-Item "C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env" -Destination ".env"
```
**Plain English:** This copies your API keys file into the project folder as ".env".

**Verify the file exists:**
```powershell
Test-Path ".env"
```
Should return `True`.

**Alternative - if .env is elsewhere:**
```powershell
# From Dropbox
Copy-Item "C:\Users\tom\Dropbox\API Keys\.env" -Destination ".env"

# Or create manually
notepad .env
```

### MacBook Pro (Terminal)

**Step 1: Copy the .env file from OneDrive**
```bash
cp ~/OneDrive/Alpha\ Loop\ LLM/API\ -\ Dec\ 2025.env .env
```
**Plain English:** This copies your API keys file. Note the backslashes before spaces in the path.

**Alternative paths:**
```bash
# From Dropbox
cp ~/Dropbox/API\ Keys/.env .env

# From iCloud
cp ~/Library/Mobile\ Documents/com~apple~CloudDocs/.env .env

# Or manually create
nano .env
```

**Verify the file exists:**
```bash
ls -la .env
```
Should show the file with its size.

---

## 6. TESTING DATABASE CONNECTION

### Windows (PowerShell)

**Step 1: Run the test script**
```powershell
python scripts/test_db_connection.py
```
**Plain English:** This tests if Python can connect to the Azure SQL database.

**What to look for:**
- ✅ "Connection successful" = Good!
- ❌ "Connection failed" = Check your .env file credentials

### MacBook Pro (Terminal)

**Step 1: Run the test script**
```bash
python scripts/test_db_connection.py
```
**Plain English:** Same test - verifies database connectivity from Mac.

---

## 7. STARTING DATA COLLECTION

### Windows (PowerShell)

**Option A: Quick Data Collection (standard)**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"
.\venv\Scripts\Activate.ps1
python src/data_ingestion/collector.py
```

**Option B: Full Universe Hydration (comprehensive)**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"
.\venv\Scripts\Activate.ps1
python scripts/hydrate_full_universe.py
```

**Option C: Alpha Vantage Premium Data**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"
.\venv\Scripts\Activate.ps1
python scripts/hydrate_alpha_vantage.py
```

**Option D: Massive S3 Deep Historical Data**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"
.\venv\Scripts\Activate.ps1
python scripts/hydrate_massive.py
```

**Save output to log file:**
```powershell
python scripts/hydrate_full_universe.py 2>&1 | Tee-Object -FilePath logs/hydration.log
```

### MacBook Pro (Terminal)

**Option A: Quick Data Collection (standard)**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
source venv/bin/activate
python src/data_ingestion/collector.py
```

**Option B: Full Universe Hydration (comprehensive)**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
source venv/bin/activate
caffeinate -d python scripts/hydrate_full_universe.py
```
**Note:** `caffeinate -d` prevents Mac from sleeping during long operations.

**Option C: Alpha Vantage Premium Data**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
source venv/bin/activate
caffeinate -d python scripts/hydrate_alpha_vantage.py
```

**Option D: Massive S3 Deep Historical Data**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
source venv/bin/activate
caffeinate -d python scripts/hydrate_massive.py
```

**Save output to log file:**
```bash
python scripts/hydrate_full_universe.py 2>&1 | tee logs/hydration.log
```

---

## 8. RUNNING MODEL TRAINING

### Windows (PowerShell)

**Option A: Standard Training**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"
.\venv\Scripts\Activate.ps1
python src/ml/train_models.py
```

**Option B: Advanced Overnight Training**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"
.\venv\Scripts\Activate.ps1
python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()"
```

**Option C: Massive Parallel Training (Full Universe)**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"
.\venv\Scripts\Activate.ps1
python src/ml/massive_trainer.py
```

**Option D: Agent Training**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"
.\venv\Scripts\Activate.ps1
python -m src.training.agent_trainer --all
```

**Save training output to log:**
```powershell
python src/ml/train_models.py 2>&1 | Tee-Object -FilePath logs/training.log
```

### MacBook Pro (Terminal)

**Option A: Standard Training**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
source venv/bin/activate
caffeinate -d python src/ml/train_models.py
```

**Option B: Advanced Overnight Training**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
source venv/bin/activate
caffeinate -d python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()"
```

**Option C: Massive Parallel Training (Full Universe)**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
source venv/bin/activate
caffeinate -d python src/ml/massive_trainer.py
```

**Option D: Agent Training**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
source venv/bin/activate
caffeinate -d python -m src.training.agent_trainer --all
```

**Save training output to log:**
```bash
caffeinate -d python src/ml/train_models.py 2>&1 | tee logs/training.log
```

---

## 9. STARTING TRADING ENGINE

### Windows (PowerShell)

**Step 1: Ensure IBKR TWS or Gateway is running**
- Paper Trading: Port 7497
- Live Trading: Port 7496

**Step 2: Start the trading engine**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"
.\venv\Scripts\Activate.ps1
python src/trading/execution_engine.py
```

### MacBook Pro (Terminal)

**Step 1: Ensure IBKR TWS or Gateway is running**
- Paper Trading: Port 7497
- Live Trading: Port 7496

**Step 2: Start the trading engine**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
source venv/bin/activate
python src/trading/execution_engine.py
```

---

## 10. MONITORING & LOGS

### Windows (PowerShell)

**View log file (last 50 lines):**
```powershell
Get-Content logs\data_collection.log -Tail 50
```

**Watch log in real-time:**
```powershell
Get-Content logs\training.log -Tail 50 -Wait
```

**Count trained models:**
```powershell
(Get-ChildItem models\*.pkl).Count
```

**List all models with timestamps:**
```powershell
Get-ChildItem models\*.pkl | Sort-Object LastWriteTime -Descending | Select-Object Name, LastWriteTime
```

**Check database row count:**
```powershell
python -c "from src.database.connection import get_engine; from sqlalchemy import text; engine = get_engine(); result = engine.execute(text('SELECT COUNT(*) FROM price_bars')); print(f'Rows: {result.fetchone()[0]:,}')"
```

**Run model dashboard:**
```powershell
python scripts/model_dashboard.py
```

### MacBook Pro (Terminal)

**View log file (last 50 lines):**
```bash
tail -50 logs/data_collection.log
```

**Watch log in real-time:**
```bash
tail -f logs/training.log
```

**Count trained models:**
```bash
ls models/*.pkl 2>/dev/null | wc -l
```

**List all models with timestamps:**
```bash
ls -lt models/*.pkl
```

**Check database row count:**
```bash
python -c "from src.database.connection import get_engine; from sqlalchemy import text; engine = get_engine(); result = engine.execute(text('SELECT COUNT(*) FROM price_bars')); print(f'Rows: {result.fetchone()[0]:,}')"
```

**Run model dashboard:**
```bash
python scripts/model_dashboard.py
```

---

## 11. TROUBLESHOOTING

### Common Issues - Windows

**Issue: "python is not recognized"**
```powershell
# Solution 1: Use full path
C:\Python311\python.exe -m venv venv

# Solution 2: Add Python to PATH
# Search "Environment Variables" in Windows
# Edit PATH to include Python installation folder
```

**Issue: "Execution policy" error**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Issue: "Module not found"**
```powershell
# Make sure venv is activated
.\venv\Scripts\Activate.ps1
# Reinstall requirements
pip install -r requirements.txt
```

**Issue: Database connection fails**
```powershell
# Check .env file exists
Test-Path ".env"
# View .env contents (careful with secrets!)
Get-Content ".env" | Select-String "SQL"
```

### Common Issues - Mac

**Issue: "python: command not found"**
```bash
# Use python3 instead
python3 -m venv venv
python3 src/ml/train_models.py
```

**Issue: "Permission denied" on scripts**
```bash
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

**Issue: "Module not found"**
```bash
# Make sure venv is activated
source venv/bin/activate
# Reinstall requirements
pip install -r requirements.txt
```

**Issue: Mac goes to sleep during training**
```bash
# Prevent display sleep
caffeinate -d python src/ml/train_models.py

# Prevent all sleep
caffeinate -i python src/ml/train_models.py
```

**Issue: ODBC Driver not found on Mac**
```bash
# Install Microsoft ODBC Driver
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
brew update
ACCEPT_EULA=Y brew install msodbcsql17
```

---

## QUICK REFERENCE CHEAT SHEET

### One-Liner Setup Commands

**Windows - Complete Setup:**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"; python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt; Copy-Item "C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env" -Destination ".env"; python scripts/test_db_connection.py
```

**Mac - Complete Setup:**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cp ~/OneDrive/Alpha\ Loop\ LLM/API\ -\ Dec\ 2025.env .env && python scripts/test_db_connection.py
```

### One-Liner Training Commands

**Windows - Full Overnight Training:**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"; .\venv\Scripts\Activate.ps1; python scripts/hydrate_full_universe.py 2>&1 | Tee-Object logs/hydration.log
```

**Mac - Full Overnight Training:**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv && source venv/bin/activate && caffeinate -d python scripts/hydrate_full_universe.py 2>&1 | tee logs/hydration.log
```

---

## DUAL-MACHINE WORKFLOW

### Recommended Setup for Overnight Training

**Windows PC (Primary):**
- Terminal 1: Data Hydration
- Terminal 2: Model Training
- Terminal 3: Monitoring Dashboard

**MacBook Pro (Secondary):**
- Terminal 1: Research Ingestion
- Terminal 2: Sentiment Analysis
- Terminal 3: Backup Training

### Windows Terminals (Open 3)

**Terminal 1 - Data Hydration:**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"
.\venv\Scripts\Activate.ps1
python scripts/hydrate_full_universe.py 2>&1 | Tee-Object logs/hydration.log
```

**Terminal 2 - Model Training:**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"
.\venv\Scripts\Activate.ps1
python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()" 2>&1 | Tee-Object logs/training.log
```

**Terminal 3 - Monitoring:**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\gkv"
.\venv\Scripts\Activate.ps1
python scripts/model_dashboard.py
```

### Mac Terminals (Open 3)

**Terminal 1 - Research Ingestion:**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
source venv/bin/activate
caffeinate -d python scripts/ingest_research.py 2>&1 | tee logs/research.log
```

**Terminal 2 - Backup Training:**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
source venv/bin/activate
caffeinate -d python src/ml/train_models.py 2>&1 | tee logs/training_mac.log
```

**Terminal 3 - Monitoring:**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/gkv
source venv/bin/activate
watch -n 30 'ls -la models/*.pkl | wc -l'
```

---

## KEEP MACHINES AWAKE OVERNIGHT

### Windows

**Via PowerShell:**
```powershell
# Prevent sleep when plugged in
powercfg /change standby-timeout-ac 0

# Check current settings
powercfg /query
```

**Via Settings:**
1. Windows Settings → System → Power & battery
2. Set "Screen timeout" to Never when plugged in
3. Set "Sleep" to Never when plugged in

### MacBook Pro

**Prevent sleep during training:**
```bash
caffeinate -d python src/ml/train_models.py
```

**Prevent sleep indefinitely (background):**
```bash
caffeinate -d &
```

**Via System Settings:**
1. System Settings → Battery → Options
2. Turn ON "Prevent automatic sleeping when display is off"

---

**Built for Alpha Loop Capital - Institutional Grade Trading System**

*This document provides complete natural language instructions for operating the system on both Windows and Mac platforms.*

