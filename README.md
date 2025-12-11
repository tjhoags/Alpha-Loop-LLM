# Alpha Loop LLM - Institutional-Grade Algorithmic Trading System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

**Alpha Loop Capital, LLC** - Mission-Critical Production Trading System

---

## 🎯 Overview

This is the **production-ready, institutional-grade** version of the Alpha Loop algorithmic trading system. Built for:

- **Overnight training** - Run ML models while you sleep
- **Market open execution** - Execute trades by 9:30 AM ET
- **Multi-machine support** - Windows + MacBook Pro simultaneous operation
- **83+ AI agents** - Coordinated via ACA (Agent Creating Agents) architecture

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[COMPLETE_COMMAND_REFERENCE.md](COMPLETE_COMMAND_REFERENCE.md)** | Complete natural language guide for all commands |
| [SETUP_WINDOWS.md](SETUP_WINDOWS.md) | Windows-specific setup details |
| [SETUP_MAC.md](SETUP_MAC.md) | MacBook-specific setup details |
| [DUAL_MACHINE_TRAINING.md](DUAL_MACHINE_TRAINING.md) | Running on both machines simultaneously |
| [TERMINAL_COMMANDS.md](TERMINAL_COMMANDS.md) | Quick terminal command reference |
| [COMPREHENSIVE_DATA_GUIDE.md](COMPREHENSIVE_DATA_GUIDE.md) | Maximum data ingestion guide |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | ML model training details |

---

## 🚀 Quick Start

### Step 1: Open Your Terminal

<details>
<summary><b>Windows (PowerShell)</b></summary>

**In Plain English:** "Open a command window where you can type instructions"

1. Press `Windows + X` on your keyboard
2. Click "Terminal" or "Windows PowerShell"
3. A window opens with a prompt like `PS C:\Users\tom>`

**Or in Cursor IDE:** Press `Ctrl + ~` or go to Terminal → New Terminal
</details>

<details>
<summary><b>MacBook Pro (Terminal)</b></summary>

**In Plain English:** "Open a command window where you can type instructions"

1. Press `Cmd + Space` to open Spotlight
2. Type "Terminal" and press Enter
3. A window opens with a prompt like `tom@MacBook-Pro ~ %`

**Or in Cursor IDE:** Press `Cmd + ~` or go to Terminal → New Terminal
</details>

---

### Step 2: Navigate to Project

<details>
<summary><b>Windows (PowerShell)</b></summary>

**In Plain English:** "Go to the folder where all the code lives"

```powershell
# Type this and press Enter
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\upy"

# Verify you're in the right place
dir
```
</details>

<details>
<summary><b>MacBook Pro (Terminal)</b></summary>

**In Plain English:** "Go to the folder where all the code lives"

```bash
# Type this and press Enter
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/upy

# Verify you're in the right place
ls
```
</details>

---

### Step 3: Set Up Virtual Environment

<details>
<summary><b>Windows (PowerShell)</b></summary>

**In Plain English:** "Create an isolated Python workspace for this project"

```powershell
# Create the virtual environment (one-time)
python -m venv venv

# Activate it (do this every time you open a new terminal)
.\venv\Scripts\Activate.ps1

# If you get an "execution policy" error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Success:** You'll see `(venv)` at the start of your prompt
</details>

<details>
<summary><b>MacBook Pro (Terminal)</b></summary>

**In Plain English:** "Create an isolated Python workspace for this project"

```bash
# Create the virtual environment (one-time)
python3 -m venv venv

# Activate it (do this every time you open a new terminal)
source venv/bin/activate
```

**Success:** You'll see `(venv)` at the start of your prompt
</details>

---

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install with development tools:

```bash
pip install -e ".[dev]"
```

---

### Step 5: Configure Environment

<details>
<summary><b>Windows (PowerShell)</b></summary>

**In Plain English:** "Copy your API keys and database credentials to the project"

```powershell
Copy-Item "C:\Users\tom\Alphaloopcapital Dropbox\ALC Tech Agents\API - Dec 2025.env" -Destination ".env"
```
</details>

<details>
<summary><b>MacBook Pro (Terminal)</b></summary>

**In Plain English:** "Copy your API keys and database credentials to the project"

```bash
cp ~/Alphaloopcapital\ Dropbox/ALC\ Tech\ Agents/API\ -\ Dec\ 2025.env .env
```
</details>

---

### Step 6: Test Database Connection

```bash
python scripts/test_db_connection.py
```

---

## 🔄 Daily Operations

### 🌙 Night (10 PM) - Start Data Collection

**In Plain English:** "Start pulling market data from all sources"

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
cd "C:\Users\tom\.cursor\worktrees\Alpha-Loop-LLM-1\upy"
.\venv\Scripts\Activate.ps1
python src/data_ingestion/collector.py
```
</details>

<details>
<summary><b>MacBook Pro (Terminal)</b></summary>

```bash
cd ~/Alpha-Loop-LLM/Alpha-Loop-LLM-1/upy
source venv/bin/activate
caffeinate -d python src/data_ingestion/collector.py
```
**Note:** `caffeinate -d` prevents your Mac from sleeping
</details>

### 🌙 Night (10 PM) - Model Training

**In Plain English:** "Train machine learning models on the collected data"

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
python src/ml/train_models.py
```
</details>

<details>
<summary><b>MacBook Pro (Terminal)</b></summary>

```bash
caffeinate -d python src/ml/train_models.py
```
</details>

**Leave both running overnight!**

---

### ☀️ Morning (9:30 AM ET) - Start Trading

**In Plain English:** "Start the trading engine that will execute trades at market open"

```bash
python src/trading/execution_engine.py
```

**Prerequisites:** TWS/Gateway running (paper port 7497, live port 7496)

---

## 📊 Monitoring

### View Logs

<details>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
# Watch data collection (live updates)
Get-Content logs\data_collection.log -Tail 50 -Wait

# Watch model training (live updates)
Get-Content logs\model_training.log -Tail 50 -Wait

# Watch trading engine (live updates)
Get-Content logs\trading_engine.log -Tail 50 -Wait

# Check how many models have been trained
(Get-ChildItem models\*.pkl).Count
```
</details>

<details>
<summary><b>MacBook Pro (Terminal)</b></summary>

```bash
# Watch data collection (live updates)
tail -f logs/data_collection.log

# Watch model training (live updates)
tail -f logs/model_training.log

# Watch trading engine (live updates)
tail -f logs/trading_engine.log

# Check how many models have been trained
ls models/*.pkl | wc -l
```
</details>

---

## 🏗️ System Architecture

```
alpha-loop-llm/
├── src/
│   ├── agents/          # 83+ AI agents (HOAGS, GHOST, etc.)
│   ├── config/          # Configuration management (Pydantic)
│   ├── core/            # ACA engine, base classes, utilities
│   ├── data_ingestion/  # Multi-source data collection
│   ├── database/        # Azure SQL Server integration
│   ├── ml/              # ML models & feature engineering
│   ├── trading/         # Execution engine & order management
│   ├── risk/            # Risk management & position sizing
│   ├── signals/         # Signal generation modules
│   └── training/        # Agent training utilities
├── scripts/             # Utility scripts & helpers
├── models/              # Trained model files (.pkl)
├── data/                # Market data storage
├── logs/                # System logs
├── tests/               # Test suite
├── .env                 # API keys (NOT COMMITTED)
├── pyproject.toml       # Python project config
└── requirements.txt     # Python dependencies
```

---

## ⚡ Key Features

### Advanced ML Pipeline
- **50+ Features:** Price, technical indicators, volume, volatility, momentum, microstructure
- **Ensemble Models:** XGBoost, LightGBM, CatBoost
- **Time-Series CV:** No data leakage
- **Model Versioning:** Timestamped saves

### Risk Management
- **Kelly Criterion:** Position sizing with confidence weighting
- **Daily Loss Limits:** 2% max daily loss
- **Drawdown Protection:** 5% max drawdown
- **Position Limits:** Max 10 positions, 10% per position

### Data Infrastructure
- **Multi-Source:** Alpha Vantage, Polygon.io, Coinbase
- **Azure SQL:** Centralized data storage
- **Retry Logic:** Exponential backoff
- **Rate Limiting:** Automatic handling

### Trading Execution
- **Interactive Brokers:** Full integration
- **Paper Trading:** Safe testing (Port 7497)
- **Order Management:** Market orders with fill tracking
- **Real-Time Signals:** ML-based with confidence

### Agent System (ACA)
- **83+ Specialized Agents** across Investment & Operations divisions
- **Master Agents:** HOAGS (Tom), GHOST (Autonomous), FRIEDS (Chris)
- **Agent Creating Agents:** System can propose new agents for capability gaps
- **Cross-Machine Learning:** Merge training from multiple machines

---

## ❓ Troubleshooting

### "Module not found"
→ Activate venv: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (Mac)

### "Execution policy error" (Windows)
→ Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Database connection fails
→ Check `.env` file has correct SQL credentials

### "No models found"
→ Make sure model training completed (check `models/` folder)

### API rate limits
→ System handles automatically with retries

### Mac goes to sleep
→ Use `caffeinate -d python src/ml/train_models.py`

---

## ✅ Pre-Trading Checklist

Before starting trading:
- [ ] Virtual environment created and activated
- [ ] All packages installed
- [ ] `.env` file copied to project folder
- [ ] Database connection test passed
- [ ] Data collection ran overnight
- [ ] Models trained (check `models/` folder)
- [ ] Trading engine starts without errors

---

## 🎯 Success Indicators

You'll know everything is working when:

1. Data collection logs show "Collecting data for..." messages
2. Model training logs show "Training XGBoost...", "Training LightGBM..." messages
3. `models/` folder has `.pkl` files after training completes
4. Trading engine logs show "Loaded X models" when starting
5. Trading engine shows "Starting trading engine" at 9:30 AM ET

---

## 📞 Support

For issues or questions:
- **Tom Hogan** - Founder & CIO - tom@alphaloopcapital.com
- **Chris Friedman** - COO - chris@alphaloopcapital.com

---

**Built for Alpha Loop Capital - Institutional-Grade Trading System**

*Version 2.0.0 - December 2025*
