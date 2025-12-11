# Alpha Loop Capital - Setup Guide

## Overview

This guide covers setup for both **Windows (Lenovo)** and **Mac (MacBook Pro)**.

---

## Prerequisites

- Python 3.10+
- Git
- API Keys (stored in Dropbox-synced `.env` file)
- Interactive Brokers TWS/Gateway (for trading)

---

## Windows Setup

### Step 1: Open Terminal
- Press `Windows + X` → Click "Terminal"
- Or in Cursor: Press `Ctrl + ~`

### Step 2: Navigate to Project
```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\hfc"
```

### Step 3: Create Virtual Environment
```powershell
# Create (one-time)
python -m venv venv

# Activate (every session)
.\venv\Scripts\Activate.ps1

# If execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Copy Environment File
```powershell
Copy-Item "C:\Users\tom\Alphaloopcapital Dropbox\ALC Tech Agents\API - Dec 2025.env" -Destination ".env"
```

### Step 6: Test Connection
```powershell
python scripts/test_db_connection.py
```

---

## Mac Setup

### Step 1: Open Terminal
- Press `Cmd + Space` → Type "Terminal"
- Or in Cursor: Press `Cmd + ~`

### Step 2: Navigate to Project
```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/hfc
```

### Step 3: Create Virtual Environment
```bash
# Create (one-time)
python3 -m venv venv

# Activate (every session)
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Copy Environment File
```bash
cp ~/Alphaloopcapital\ Dropbox/ALC\ Tech\ Agents/API\ -\ Dec\ 2025.env .env
```

### Step 6: Test Connection
```bash
python scripts/test_db_connection.py
```

---

## Quick Reference

| Task | Windows | Mac |
|------|---------|-----|
| Activate venv | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` |
| View logs | `Get-Content logs\file.log -Tail 50 -Wait` | `tail -f logs/file.log` |
| Prevent sleep | N/A (power settings) | `caffeinate -d` |
| Path separator | `\` (backslash) | `/` (forward slash) |

---

## Troubleshooting

### "Module not found"
Ensure virtual environment is activated (look for `(venv)` in prompt).

### "Execution policy error" (Windows)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "python3 command not found" (Mac)
Use `python` instead of `python3`, or install Python via Homebrew.

### Database connection fails
1. Verify `.env` file exists: `ls -la .env` or `Test-Path .env`
2. Check credentials: `grep SQL_ .env` or `Get-Content .env | Select-String "SQL_"`

---

*© 2025 Alpha Loop Capital, LLC*

