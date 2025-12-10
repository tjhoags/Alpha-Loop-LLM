# Windows Setup Instructions

> **📚 For complete cross-platform reference, see [`CROSS_PLATFORM_COMMANDS.md`](CROSS_PLATFORM_COMMANDS.md)**

## Which Terminal to Use?

**You can use EITHER:**
1. **Local Windows PowerShell** (Windows + X → Terminal)
2. **Cursor's Integrated Terminal** (Terminal menu → New Terminal)

Both work the same way! Use whichever you prefer.

---

## Quick Setup (Windows)

### Step 1: Open Terminal
**In Plain English:** "Open a command window where you can type instructions"

- **Option A:** Press `Windows + X` → Click "Terminal"
- **Option B:** In Cursor, press `Ctrl + ~` or go to Terminal → New Terminal

### Step 2: Navigate to Project
**In Plain English:** "Go to the folder where all the code lives"

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
```

### Step 3: Set Up Environment
**In Plain English:** "Create an isolated Python workspace for this project"

```powershell
# Create virtual environment (one-time setup)
python -m venv venv

# Activate it (do this every time you open a new terminal)
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run this once:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Success:** You'll see `(venv)` at the start of your prompt

### Step 4: Install Packages
**In Plain English:** "Install all required Python packages"

```powershell
pip install -r requirements.txt
```

### Step 5: Copy Environment File
**In Plain English:** "Copy your API keys and database credentials"

```powershell
Copy-Item "C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env" -Destination ".env"
```

### Step 6: Test Database
**In Plain English:** "Make sure we can connect to the database"

```powershell
python scripts/test_db_connection.py
```

---

## Start Training (Windows)

### Terminal 1 - Data Collection
**In Plain English:** "Start pulling market data from all sources"

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1
python src/data_ingestion/collector.py
```

### Terminal 2 - Model Training
**In Plain English:** "Train machine learning models on the collected data"

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1
python src/ml/train_models.py
```

### Terminal 3 - Monitoring
**In Plain English:** "Watch what the system is doing"

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1

# Watch data collection (live updates)
Get-Content logs\data_collection.log -Tail 50 -Wait

# Check how many models have been trained
(Get-ChildItem models\*.pkl).Count
```

---

## Morning Trading (9:15 AM ET)

**In Plain English:** "Start the trading engine before market opens"

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1
python src/trading/execution_engine.py
```

**Prerequisites:** IBKR TWS/Gateway must be running (Paper: port 7497)

---

## Notes for Windows

| Topic | Details |
|-------|---------|
| Shell | Use PowerShell (not CMD) |
| Paths | Use backslashes: `C:\Users\tom\...` |
| Activate | `.\venv\Scripts\Activate.ps1` |
| View logs | `Get-Content file -Tail 50` |
| Live tail | `Get-Content file -Tail 50 -Wait` |
| File exists | `Test-Path "file"` |

---

## Troubleshooting

### "Execution Policy" Error
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Module not found" Error
Make sure venv is activated (you should see `(venv)` in prompt):
```powershell
.\venv\Scripts\Activate.ps1
```

### Database Connection Fails
Check that .env file exists and has correct credentials:
```powershell
Test-Path ".env"
Get-Content ".env" | Select-String "DB_"
```

---

## Quick Reference

```powershell
# Complete setup in one block (copy-paste)
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item "C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env" -Destination ".env"
python scripts/test_db_connection.py
```

