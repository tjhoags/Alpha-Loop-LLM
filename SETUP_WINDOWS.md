# Windows Setup Instructions

## Which Terminal to Use?

**You can use EITHER:**
1. **Local Windows PowerShell** (Windows + X → Terminal)
2. **Cursor's Integrated Terminal** (Terminal menu → New Terminal)

Both work the same way! Use whichever you prefer.

---

## Quick Setup (Windows)

### Step 1: Open Terminal
- **Option A:** Press `Windows + X` → Click "Terminal"
- **Option B:** In Cursor, go to Terminal → New Terminal

### Step 2: Navigate to Project
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\bek"
```

### Step 3: Set Up Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# If execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Install Packages
```powershell
pip install -r requirements.txt
```

### Step 5: Copy Environment File
```powershell
Copy-Item "C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env" -Destination ".env"
```

### Step 6: Test Database
```powershell
python scripts/test_db_connection.py
```

---

## Start Training (Windows)

### Terminal 1 - Data Collection:
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\bek"
.\venv\Scripts\Activate.ps1
python src/data_ingestion/collector.py
```

### Terminal 2 - Model Training:
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\bek"
.\venv\Scripts\Activate.ps1
python src/ml/train_models.py
```

---

## Notes for Windows

- Use PowerShell (not CMD)
- Paths use backslashes: `C:\Users\tom\...`
- Activate script: `.\venv\Scripts\Activate.ps1`
- Use `Get-Content` for viewing logs

