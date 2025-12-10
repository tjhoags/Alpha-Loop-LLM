# 🖥️ Terminal Commands Reference
## Quick Reference for Windows & Mac Operations

> **For detailed explanations, see [`CROSS_PLATFORM_COMMANDS.md`](CROSS_PLATFORM_COMMANDS.md)**

---

## 📊 Current System Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Data in Azure SQL** | 3,400+ symbols, 1.4M+ rows | Continuously updating |
| **Models Trained** | Check `models/` folder | Retrained hourly |
| **Trading Engine** | Ready | Start at 9:15 AM ET |

---

## 🚀 Quick Start Commands

### Initial Setup (One-Time)

<table>
<tr>
<th>🪟 Windows (PowerShell)</th>
<th>🍎 MacBook Pro (Terminal)</th>
</tr>
<tr>
<td>

```powershell
# Step 1: Open terminal
# Press Windows + X → Terminal

# Step 2: Navigate to project
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"

# Step 3: Create virtual environment
python -m venv venv

# Step 4: Activate it
.\venv\Scripts\Activate.ps1

# Step 5: Install packages
pip install -r requirements.txt

# Step 6: Copy environment file
Copy-Item "C:\Users\tom\OneDrive\Alpha Loop LLM\API - Dec 2025.env" -Destination ".env"

# Step 7: Test database
python scripts/test_db_connection.py
```

</td>
<td>

```bash
# Step 1: Open terminal
# Press Cmd + Space → type "Terminal"

# Step 2: Navigate to project
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii

# Step 3: Create virtual environment
python3 -m venv venv

# Step 4: Activate it
source venv/bin/activate

# Step 5: Install packages
pip install -r requirements.txt

# Step 6: Copy environment file
cp ~/OneDrive/Alpha\ Loop\ LLM/API\ -\ Dec\ 2025.env .env

# Step 7: Test database
python scripts/test_db_connection.py
```

</td>
</tr>
</table>

---

## 🌙 Overnight Training (5+ Terminals)

### Terminal Configuration

| Terminal | Windows | Mac | Purpose |
|----------|---------|-----|---------|
| **T1** | Data Hydration | Data Hydration | Pull market data |
| **T2** | ML Training | ML Training | Train models |
| **T3** | Monitor | Monitor | Dashboard |
| **T4** | - | Research | Mac-specific data |
| **T5** | - | Backup Training | Redundancy |

### Terminal 1: Data Hydration

<table>
<tr>
<th>🪟 Windows</th>
<th>🍎 Mac</th>
</tr>
<tr>
<td>

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1
python scripts/hydrate_full_universe.py 2>&1 | Tee-Object -FilePath logs/hydration.log
```

</td>
<td>

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
caffeinate -d python scripts/hydrate_full_universe.py 2>&1 | tee logs/hydration.log
```

</td>
</tr>
</table>

### Terminal 2: Model Training

<table>
<tr>
<th>🪟 Windows</th>
<th>🍎 Mac</th>
</tr>
<tr>
<td>

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1
python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()" 2>&1 | Tee-Object -FilePath logs/training.log
```

</td>
<td>

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
caffeinate -d python -c "from src.ml.advanced_training import run_overnight_training; run_overnight_training()" 2>&1 | tee logs/training.log
```

</td>
</tr>
</table>

### Terminal 3: Monitor Dashboard

<table>
<tr>
<th>🪟 Windows</th>
<th>🍎 Mac</th>
</tr>
<tr>
<td>

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1

# Check hydration progress
Get-Content logs/hydration.log -Tail 10

# Check training progress
Get-Content logs/training.log -Tail 10

# Model count
(Get-ChildItem models\*.pkl -ErrorAction SilentlyContinue).Count

# Full dashboard
python scripts/model_dashboard.py
```

</td>
<td>

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate

# Check hydration progress
tail -10 logs/hydration.log

# Check training progress
tail -10 logs/training.log

# Model count
ls models/*.pkl 2>/dev/null | wc -l

# Full dashboard
python scripts/model_dashboard.py
```

</td>
</tr>
</table>

### Terminal 4 (Mac Only): Research Ingestion

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
caffeinate -d python scripts/ingest_research.py 2>&1 | tee logs/research.log
```

### Terminal 5 (Mac Only): Backup Training

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
caffeinate -d python src/ml/train_models.py 2>&1 | tee logs/training_backup.log
```

---

## ☀️ Morning Trading (9:15 AM ET)

<table>
<tr>
<th>🪟 Windows</th>
<th>🍎 Mac</th>
</tr>
<tr>
<td>

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"
.\venv\Scripts\Activate.ps1
python src/trading/execution_engine.py
```

</td>
<td>

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii
source venv/bin/activate
python src/trading/execution_engine.py
```

</td>
</tr>
</table>

**Prerequisites:** IBKR TWS/Gateway running (Paper: 7497, Live: 7496)

---

## 📊 Status Check Commands

### Quick Status

<table>
<tr>
<th>🪟 Windows</th>
<th>🍎 Mac</th>
</tr>
<tr>
<td>

```powershell
# How many models trained?
(Get-ChildItem models\*.pkl).Count

# Last 5 log entries
Get-Content logs\training.log -Tail 5

# Database row count
python -c "from src.database.connection import get_engine; import pandas as pd; print(pd.read_sql('SELECT COUNT(*) FROM price_bars', get_engine()))"
```

</td>
<td>

```bash
# How many models trained?
ls models/*.pkl | wc -l

# Last 5 log entries
tail -5 logs/training.log

# Database row count
python -c "from src.database.connection import get_engine; import pandas as pd; print(pd.read_sql('SELECT COUNT(*) FROM price_bars', get_engine()))"
```

</td>
</tr>
</table>

---

## 🔧 Utility Commands

### Data Operations

<table>
<tr>
<th>🪟 Windows</th>
<th>🍎 Mac</th>
</tr>
<tr>
<td>

```powershell
# Test DB connection
python scripts/test_db_connection.py

# Quick data hydration
.\scripts\HYDRATE_QUICK.bat

# Full universe hydration
.\scripts\HYDRATE_FULL_UNIVERSE.bat

# Ingest research
.\scripts\INGEST_RESEARCH.bat
```

</td>
<td>

```bash
# Test DB connection
python scripts/test_db_connection.py

# Quick data hydration
bash scripts/mac_data_collection.sh

# Full overnight training
bash scripts/mac_overnight_training.sh

# Ingest research
python scripts/ingest_research.py
```

</td>
</tr>
</table>

### Agent Operations

<table>
<tr>
<th>🪟 Windows</th>
<th>🍎 Mac</th>
</tr>
<tr>
<td>

```powershell
# Start agent chat
.\scripts\START_AGENT_CHAT.bat

# Train all agents
.\scripts\TRAIN_ALL_AGENTS.bat

# Check model grades
.\scripts\CHECK_MODEL_GRADES.bat
```

</td>
<td>

```bash
# Start agent chat
python src/interfaces/agent_chat.py

# Train all agents
python src/training/train_all_agents.py

# Check model grades
python scripts/training_status.py
```

</td>
</tr>
</table>

### Code Review

<table>
<tr>
<th>🪟 Windows</th>
<th>🍎 Mac</th>
</tr>
<tr>
<td>

```powershell
# Run code review
.\scripts\REVIEW_CODE.bat

# Or directly
python -m src.review.orchestrator

# Issue scanner
python -m src.review.issue_scanner
```

</td>
<td>

```bash
# Run code review
python -m src.review.orchestrator

# Issue scanner
python -m src.review.issue_scanner
```

</td>
</tr>
</table>

---

## 📈 Training Timeline

| Time | Activity | Status |
|------|----------|--------|
| Now | Start hydration + training | 🔄 Running |
| +1 hour | First models complete | ✅ Check |
| +3 hours | 500+ models trained | ✅ Check |
| +6 hours | Most stocks complete | ✅ Check |
| 9:00 AM | Check model grades | 📊 Review |
| 9:15 AM | Start trading engine | 🚀 Go |
| 9:30 AM | Market open | 💰 Trading |

---

## ⚠️ Troubleshooting

### Common Issues

| Problem | Windows Fix | Mac Fix |
|---------|-------------|---------|
| "Module not found" | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` |
| "Execution policy" | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` | N/A |
| "python not found" | Use `python` | Use `python3` |
| Mac sleeps | N/A | `caffeinate -d` |
| DB connection fails | Check `.env` file | Check `.env` file |

### Verify Environment

<table>
<tr>
<th>🪟 Windows</th>
<th>🍎 Mac</th>
</tr>
<tr>
<td>

```powershell
# Check Python version
python --version

# Check venv is active
Get-Command python

# Check .env exists
Test-Path ".env"

# View .env (first 5 lines)
Get-Content ".env" | Select -First 5
```

</td>
<td>

```bash
# Check Python version
python --version

# Check venv is active
which python

# Check .env exists
ls -la .env

# View .env (first 5 lines)
head -5 .env
```

</td>
</tr>
</table>

---

## 🎯 One-Line Commands (Copy-Paste Ready)

### Windows - Start Everything

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\sii"; .\venv\Scripts\Activate.ps1; .\scripts\start_full_throttle_training.ps1
```

### Mac - Start Everything

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/sii && source venv/bin/activate && bash scripts/start_full_throttle_training.sh
```

---

## 📚 Documentation Links

| Document | Description |
|----------|-------------|
| [CROSS_PLATFORM_COMMANDS.md](CROSS_PLATFORM_COMMANDS.md) | Detailed natural language guide |
| [SETUP_WINDOWS.md](SETUP_WINDOWS.md) | Windows-specific setup |
| [SETUP_MAC.md](SETUP_MAC.md) | Mac-specific setup |
| [FULL_THROTTLE_SETUP.md](FULL_THROTTLE_SETUP.md) | Maximum data ingestion |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | ML training details |
| [AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md) | Agent system overview |

---

**Alpha Loop Capital - Institutional-Grade Trading System**
