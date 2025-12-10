# 🖥️ Cross-Platform Command Reference
## Windows (PowerShell) & MacBook Pro (Terminal) Instructions

> **Natural Language Guide**: Every command explained in plain English with step-by-step instructions for both platforms.

---

## 📋 Table of Contents
1. [Opening Terminal](#1-opening-terminal)
2. [Navigate to Project](#2-navigate-to-project)
3. [Virtual Environment Setup](#3-virtual-environment-setup)
4. [Install Dependencies](#4-install-dependencies)
5. [Environment Configuration](#5-environment-configuration)
6. [Database Operations](#6-database-operations)
7. [Data Collection](#7-data-collection)
8. [Model Training](#8-model-training)
9. [Trading Engine](#9-trading-engine)
10. [Monitoring & Logs](#10-monitoring--logs)
11. [Agent Operations](#11-agent-operations)
12. [Review System](#12-review-system)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Opening Terminal

### What You're Doing
Opening a command-line interface where you can type commands to control the system.

### Windows (PowerShell)
**Option A - Quick Access:**
1. Press `Windows + X` on your keyboard
2. Click "Terminal" or "Windows PowerShell" in the menu
3. A blue/black window opens - you're ready!

**Option B - Search:**
1. Press `Windows` key
2. Type "PowerShell"
3. Click "Windows PowerShell" or "Terminal"

**Option C - Cursor IDE:**
1. In Cursor, press `Ctrl + ~` (backtick)
2. Or go to Terminal menu → New Terminal

```powershell
# You should see a prompt like:
# PS C:\Users\tom>
```

### MacBook Pro (Terminal)
**Option A - Spotlight:**
1. Press `Cmd + Space` to open Spotlight
2. Type "Terminal"
3. Press Enter
4. A white/black window opens - you're ready!

**Option B - Finder:**
1. Open Finder
2. Go to Applications → Utilities → Terminal
3. Double-click Terminal

**Option C - Cursor IDE:**
1. In Cursor, press `Cmd + ~` (backtick)
2. Or go to Terminal menu → New Terminal

```bash
# You should see a prompt like:
# tom@MacBook-Pro ~ %
```

---

## 2. Navigate to Project

### What You're Doing
Changing to the folder where all the code lives. This is always your first step.

### Windows (PowerShell)
```powershell
# Navigate to the Alpha Loop LLM project
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"

# Verify you're in the right place (should show project files)
Get-ChildItem

# Alternative: List files in simplified view
dir
```

**In Plain English:** "Go to the folder called sii inside Alpha-Loop-LLM-1"

### MacBook Pro (Terminal)
```bash
# Navigate to the Alpha Loop LLM project
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii

# Verify you're in the right place (should show project files)
ls -la

# Alternative: Simple list
ls
```

**In Plain English:** "Go to the folder called sii inside Alpha-Loop-LLM-1 in your home directory"

### Understanding Paths
| Platform | Home Directory | Project Path |
|----------|---------------|--------------|
| Windows | `C:\Users\tom\` | `C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii` |
| Mac | `~/` or `/Users/tom/` | `~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii` |

---

## 3. Virtual Environment Setup

### What You're Doing
Creating an isolated Python environment so our packages don't conflict with other projects.

### Windows (PowerShell)

**Step 1: Create the virtual environment**
```powershell
# First, navigate to project
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"

# Create virtual environment (one-time setup)
python -m venv venv
```
**What This Does:** Creates a `venv` folder with a fresh Python installation

**Step 2: Activate the virtual environment**
```powershell
# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```
**What This Does:** Turns on the isolated environment. You'll see `(venv)` before your prompt.

**If You Get an Execution Policy Error:**
```powershell
# Allow script execution (one-time setup, admin not required)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activating again
.\venv\Scripts\Activate.ps1
```

### MacBook Pro (Terminal)

**Step 1: Create the virtual environment**
```bash
# First, navigate to project
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii

# Create virtual environment (one-time setup)
python3 -m venv venv
```
**What This Does:** Creates a `venv` folder with a fresh Python installation

**Step 2: Activate the virtual environment**
```bash
# Activate the virtual environment
source venv/bin/activate
```
**What This Does:** Turns on the isolated environment. You'll see `(venv)` before your prompt.

### Verification
```powershell
# Windows - Check Python location (should show venv path)
Get-Command python
```

```bash
# Mac - Check Python location (should show venv path)
which python
```

---

## 4. Install Dependencies

### What You're Doing
Installing all the Python packages the system needs to run.

### Windows (PowerShell)
```powershell
# Make sure you're in the project and venv is active
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1

# Install all required packages
pip install -r requirements.txt

# Upgrade pip first if you get warnings
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### MacBook Pro (Terminal)
```bash
# Make sure you're in the project and venv is active
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt

# Upgrade pip first if you get warnings
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**What This Does:** Reads the `requirements.txt` file and installs XGBoost, LightGBM, CatBoost, pandas, and all other required packages.

---

## 5. Environment Configuration

### What You're Doing
Setting up your API keys and database credentials securely.

### Windows (PowerShell)
```powershell
# Navigate to project
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"

# Copy the environment file from OneDrive
Copy-Item "C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env" -Destination ".env"

# Verify the file was copied
Test-Path ".env"  # Should return True

# View contents (careful - contains secrets)
Get-Content ".env" | Select-Object -First 5
```

### MacBook Pro (Terminal)
```bash
# Navigate to project
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii

# Copy the environment file from OneDrive
cp ~/OneDrive/Alpha\ Loop\ LLM/API\ -\ Dec\ 2025.env .env

# Or from iCloud
cp ~/Library/Mobile\ Documents/com~apple~CloudDocs/Alpha\ Loop\ LLM/.env .env

# Verify the file was copied
ls -la .env

# View first few lines (careful - contains secrets)
head -5 .env
```

### Required Environment Variables
Your `.env` file should contain:
```env
# Database
DB_SERVER=your-server.database.windows.net
DB_DATABASE=alc_market_data
DB_USERNAME=your-username
DB_PASSWORD=your-password

# API Keys
ALPHA_VANTAGE_API_KEY=your-key
POLYGON_API_KEY=your-key
COINBASE_API_KEY=your-key

# Trading
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # Paper trading
```

---

## 6. Database Operations

### What You're Doing
Testing and managing the Azure SQL database connection.

### Windows (PowerShell)
```powershell
# Navigate and activate
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1

# Test database connection
python scripts/test_db_connection.py

# Setup database schema (if needed)
python scripts/setup_db_schema.py

# Check row count in price_bars table
python -c "from src.database.connection import get_engine; import pandas as pd; print(pd.read_sql('SELECT COUNT(*) as count FROM price_bars', get_engine()))"
```

### MacBook Pro (Terminal)
```bash
# Navigate and activate
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate

# Test database connection
python scripts/test_db_connection.py

# Setup database schema (if needed)
python scripts/setup_db_schema.py

# Check row count in price_bars table
python -c "from src.database.connection import get_engine; import pandas as pd; print(pd.read_sql('SELECT COUNT(*) as count FROM price_bars', get_engine()))"
```

---

## 7. Data Collection

### What You're Doing
Pulling market data from multiple sources into the database.

### Windows (PowerShell)

**Option A: Standard Data Collection**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1

# Start data collector
python src/data_ingestion/collector.py
```

**Option B: Full Throttle - Alpha Vantage Premium**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1

# Hydrate all Alpha Vantage data
python scripts/hydrate_all_alpha_vantage.py 2>&1 | Tee-Object -FilePath logs/alpha_vantage.log
```

**Option C: Full Universe Hydration**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1

# Pull data for entire market universe
python scripts/hydrate_full_universe.py 2>&1 | Tee-Object -FilePath logs/hydration.log
```

### MacBook Pro (Terminal)

**Option A: Standard Data Collection**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate

# Start data collector
python src/data_ingestion/collector.py
```

**Option B: Full Throttle - Alpha Vantage Premium (Keep Mac Awake)**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate

# Hydrate with caffeinate to prevent sleep
caffeinate -d python scripts/hydrate_all_alpha_vantage.py 2>&1 | tee logs/alpha_vantage.log
```

**Option C: Full Universe Hydration (Keep Mac Awake)**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate

# Pull data for entire market universe
caffeinate -d python scripts/hydrate_full_universe.py 2>&1 | tee logs/hydration.log
```

### Data Sources Collected
| Source | Data Type | History |
|--------|-----------|---------|
| Alpha Vantage | Stocks, Fundamentals | 20+ years daily |
| Polygon | 1-minute bars | 2 years |
| Coinbase | Crypto (BTC, ETH) | Full history |
| FRED | Macro indicators | Full history |

---

## 8. Model Training

### What You're Doing
Training machine learning models (XGBoost, LightGBM, CatBoost) on the collected data.

### Windows (PowerShell)

**Standard Training**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1

# Start model training
python src/ml/train_models.py
```

**Advanced Overnight Training**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1

# Full overnight training with logging
python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()" 2>&1 | Tee-Object -FilePath logs/training.log
```

**Full Throttle Setup (3 terminals)**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1

# Start automated full throttle training
.\scripts\start_full_throttle_training.ps1
```

### MacBook Pro (Terminal)

**Standard Training**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate

# Start model training
python src/ml/train_models.py
```

**Advanced Overnight Training (Keep Mac Awake)**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate

# Full overnight training with caffeinate
caffeinate -d python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()" 2>&1 | tee logs/training.log
```

**Full Throttle Setup (3 terminals)**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate

# Start automated full throttle training
bash scripts/start_full_throttle_training.sh
```

### Check Training Progress
```powershell
# Windows - View training logs
Get-Content logs\training.log -Tail 20

# Count trained models
(Get-ChildItem models\*.pkl).Count
```

```bash
# Mac - View training logs
tail -f logs/training.log

# Count trained models
ls models/*.pkl | wc -l
```

---

## 9. Trading Engine

### What You're Doing
Starting the trading execution system that generates signals and places orders.

### Prerequisites
- IBKR TWS or Gateway must be running
- Paper trading: Port 7497
- Live trading: Port 7496

### Windows (PowerShell)
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1

# Start trading engine (run at 9:15 AM ET)
python src/trading/execution_engine.py

# Alternative: Use batch file
.\scripts\START_TRADING_ENGINE.bat
```

### MacBook Pro (Terminal)
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate

# Start trading engine (run at 9:15 AM ET)
python src/trading/execution_engine.py

# Alternative: Use shell script
bash scripts/mac_trading_engine.sh
```

---

## 10. Monitoring & Logs

### What You're Doing
Watching system progress and checking for errors.

### Windows (PowerShell)

**View Logs in Real-Time**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"

# Data collection log
Get-Content logs\data_collection.log -Tail 50 -Wait

# Model training log
Get-Content logs\training.log -Tail 50 -Wait

# Trading engine log
Get-Content logs\trading_engine.log -Tail 50 -Wait

# All logs at once (separate terminals)
Get-Content logs\*.log -Tail 10
```

**Check System Status**
```powershell
# Model dashboard
python scripts/model_dashboard.py

# Training status
python scripts/training_status.py

# Check model grades
.\scripts\CHECK_MODEL_GRADES.bat
```

### MacBook Pro (Terminal)

**View Logs in Real-Time**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii

# Data collection log
tail -f logs/data_collection.log

# Model training log
tail -f logs/training.log

# Trading engine log
tail -f logs/trading_engine.log

# All logs at once (use separate terminals or tmux)
tail -f logs/*.log
```

**Check System Status**
```bash
# Model dashboard
python scripts/model_dashboard.py

# Training status
python scripts/training_status.py

# Check model count
ls -la models/*.pkl | wc -l
```

---

## 11. Agent Operations

### What You're Doing
Working with the 93 AI agents that power the system.

### Windows (PowerShell)

**Start Agent Chat Interface**
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1

# Interactive agent chat
python src/interfaces/agent_chat.py

# Or use batch file
.\scripts\START_AGENT_CHAT.bat
```

**Train All Agents**
```powershell
# Train all 93 agents
.\scripts\TRAIN_ALL_AGENTS.bat
```

### MacBook Pro (Terminal)

**Start Agent Chat Interface**
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate

# Interactive agent chat
python src/interfaces/agent_chat.py
```

**Train All Agents**
```bash
# Train agents with overnight protection
caffeinate -d python src/training/train_all_agents.py
```

---

## 12. Review System

### What You're Doing
Running automated code review across the entire project.

### Windows (PowerShell)
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1

# Run code review
python -m src.review.orchestrator

# Or use batch file
.\scripts\REVIEW_CODE.bat

# Perform specific review
python scripts/perform_review.py
```

### MacBook Pro (Terminal)
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate

# Run code review
python -m src.review.orchestrator

# Perform specific review
python scripts/perform_review.py
```

---

## 13. Troubleshooting

### Common Issues & Solutions

#### "Module not found" Error

**Windows:**
```powershell
# Make sure venv is activated (you should see (venv) in prompt)
.\venv\Scripts\Activate.ps1

# If still failing, reinstall packages
pip install -r requirements.txt
```

**Mac:**
```bash
# Make sure venv is activated (you should see (venv) in prompt)
source venv/bin/activate

# If still failing, reinstall packages
pip install -r requirements.txt
```

#### "Execution Policy" Error (Windows Only)
```powershell
# Run this once to allow scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Verify it worked
Get-ExecutionPolicy -List
```

#### "python: command not found" (Mac Only)
```bash
# Use python3 instead
python3 -m venv venv
python3 src/data_ingestion/collector.py

# Or create an alias in ~/.zshrc or ~/.bashrc
echo "alias python=python3" >> ~/.zshrc
source ~/.zshrc
```

#### Database Connection Fails
```powershell
# Windows - Check .env file exists
Test-Path ".env"

# View DB settings (without passwords)
Get-Content ".env" | Select-String "DB_"
```

```bash
# Mac - Check .env file exists
ls -la .env

# View DB settings (without passwords)
grep "DB_" .env
```

#### Mac Goes to Sleep During Training
```bash
# Use caffeinate with any long-running command
caffeinate -d python src/ml/train_models.py

# Keep running until you manually stop (Ctrl+C)
caffeinate -d

# Keep running for 8 hours (28800 seconds)
caffeinate -t 28800 python src/ml/train_models.py
```

#### Check Python/Package Versions
```powershell
# Windows
python --version
pip list | Select-String "xgboost|lightgbm|catboost|pandas"
```

```bash
# Mac
python --version
pip list | grep -E "xgboost|lightgbm|catboost|pandas"
```

---

## 📋 Quick Reference Cards

### Windows Quick Start
```powershell
# Complete setup in one block
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item "C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env" -Destination ".env"
python scripts/test_db_connection.py
```

### Mac Quick Start
```bash
# Complete setup in one block
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp ~/OneDrive/Alpha\ Loop\ LLM/API\ -\ Dec\ 2025.env .env
python scripts/test_db_connection.py
```

### Command Translation Table

| Action | Windows (PowerShell) | Mac (Terminal) |
|--------|---------------------|----------------|
| Change directory | `cd "path\to\folder"` | `cd ~/path/to/folder` |
| List files | `Get-ChildItem` or `dir` | `ls -la` |
| Copy file | `Copy-Item src -Destination dst` | `cp src dst` |
| View file | `Get-Content file.txt` | `cat file.txt` |
| View last N lines | `Get-Content file -Tail N` | `tail -N file` |
| Follow log | `Get-Content file -Tail 50 -Wait` | `tail -f file` |
| Find text | `Select-String "pattern" file` | `grep "pattern" file` |
| Check if file exists | `Test-Path "file"` | `test -f file && echo yes` |
| Create directory | `New-Item -ItemType Directory -Path dir` | `mkdir -p dir` |
| Delete file | `Remove-Item file` | `rm file` |
| Activate venv | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` |
| Prevent sleep | (Power settings) | `caffeinate -d` |

---

**Built for Alpha Loop Capital - Institutional-Grade Trading System**
**Cross-Platform Compatible: Windows 10/11 & macOS**

